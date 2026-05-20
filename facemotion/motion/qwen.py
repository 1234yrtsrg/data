"""Adapter for Qwen3-14B-Instruct JSON generation."""

from __future__ import annotations

import json
import os
import re
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


DEFAULT_CHAT_MODEL = "Qwen/Qwen3-14B-Instruct"


class QwenJSONDecodeError(ValueError):
    """Raised when Qwen output cannot be parsed as a JSON object."""

    def __init__(self, message: str, raw_output: str):
        super().__init__(f"{message}\nRaw output:\n{raw_output}")
        self.raw_output = raw_output


class QwenChat:
    """Thin chat adapter for Qwen3-14B-Instruct.

    The class is intentionally business-prompt agnostic. Pass an existing model
    wrapper, a callable chat client, or a Hugging Face model/tokenizer pair.
    """

    def __init__(
        self,
        model: Any | None = None,
        tokenizer: Any | None = None,
        model_id: str = DEFAULT_CHAT_MODEL,
        gpu: int = 0,
        device_map: str = "cuda",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.gpu = gpu
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.kwargs = kwargs
        self.last_raw_output: str | None = None

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_new_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Generate and parse one JSON object."""

        raw_output = self._generate_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        self.last_raw_output = raw_output
        return self._parse_json_object(raw_output)

    def _generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        if self.model is None:
            self._load_transformers_model()

        if hasattr(self.model, "generate_json"):
            result = self.model.generate_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            if isinstance(result, dict):
                return json.dumps(result, ensure_ascii=False)
            return self._coerce_result_to_text(result)

        if hasattr(self.model, "chat"):
            result = self.model.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            return self._coerce_result_to_text(result)

        if self.tokenizer is not None and hasattr(self.model, "generate"):
            return self._generate_with_transformers(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )

        if callable(self.model):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                result = self.model(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            except TypeError:
                result = self.model(
                    messages,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            return self._coerce_result_to_text(result)

        raise TypeError(
            "Unsupported QwenChat model adapter. Provide a model with chat(), "
            "generate_json(), generate()+tokenizer, or a callable client."
        )

    def _load_transformers_model(self) -> None:
        """Lazy-load Qwen3-14B-Instruct with Hugging Face Transformers."""

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenChat requires torch and transformers to load Qwen3-14B-Instruct. "
                "Install them or pass an already-loaded model/tokenizer into QwenChat."
            ) from exc

        torch_dtype = self._resolve_torch_dtype(torch)

        if self.device_map not in {"cuda", "auto", "cpu"}:
            raise ValueError("device_map must be one of: 'cuda', 'auto', 'cpu'")

        if self.device_map == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. Use device_map='cpu' or run on a CUDA server."
                )
            device_count = torch.cuda.device_count()
            if self.gpu < 0 or self.gpu >= device_count:
                raise ValueError(
                    f"Invalid GPU id {self.gpu}. This server has {device_count} CUDA device(s)."
                )
            torch.cuda.set_device(self.gpu)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )

        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if self.device_map == "auto":
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self.model.eval()

        if self.device_map == "cuda":
            self.model.to(f"cuda:{self.gpu}")
        elif self.device_map == "cpu":
            self.model.to("cpu")

    def _resolve_torch_dtype(self, torch_module: Any) -> Any:
        if self.torch_dtype == "auto":
            return "auto"
        dtype_map = {
            "bfloat16": torch_module.bfloat16,
            "float16": torch_module.float16,
            "float32": torch_module.float32,
        }
        if self.torch_dtype not in dtype_map:
            raise ValueError("torch_dtype must be one of: 'bfloat16', 'float16', 'float32', 'auto'")
        return dtype_map[self.torch_dtype]

    def _generate_with_transformers(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        if not hasattr(self.tokenizer, "apply_chat_template"):
            raise TypeError("tokenizer must support apply_chat_template for chat generation")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self._apply_chat_template(messages)
        inputs = self.tokenizer([prompt_text], return_tensors="pt")

        try:
            device = getattr(self.model, "device", None) or next(self.model.parameters()).device
            inputs = {key: value.to(device) for key, value in inputs.items()}
        except Exception:
            pass

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature
        if getattr(self.tokenizer, "eos_token_id", None) is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        generation_kwargs.update(self.kwargs.get("generation_kwargs", {}))

        try:
            import torch

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
        except ImportError:
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        input_length = inputs["input_ids"].shape[-1]
        generated_ids = output_ids[0][input_length:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    @staticmethod
    def _coerce_result_to_text(result: Any) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)
        if hasattr(result, "choices"):
            choice = result.choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if content is not None:
                    return str(content)
            text = getattr(choice, "text", None)
            if text is not None:
                return str(text)
        if hasattr(result, "content"):
            return str(result.content)
        if hasattr(result, "text"):
            return str(result.text)
        return str(result)

    @classmethod
    def _parse_json_object(cls, raw_output: str) -> dict[str, Any]:
        cleaned = cls._strip_markdown_code_fence(raw_output)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise QwenJSONDecodeError(
                f"Qwen output is not valid JSON: {exc}",
                raw_output,
            ) from exc

        if not isinstance(parsed, dict):
            raise QwenJSONDecodeError("Qwen output must be a JSON object", raw_output)
        return parsed

    @staticmethod
    def _strip_markdown_code_fence(text: str) -> str:
        stripped = text.strip()
        fenced = re.fullmatch(r"```(?:json|JSON)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
        if fenced:
            return fenced.group(1).strip()
        return stripped
