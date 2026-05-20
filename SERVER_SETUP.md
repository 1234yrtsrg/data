# Facemotion Server Setup

This project uses three model/runtime stacks:

- `Qwen/Qwen3-14B` for text-to-motion-spec and prompt compilation.
- `Qwen/Qwen-Image-Edit-2511` with Lightning LoRA for image editing.
- MediaPipe Face Landmarker for blendshape extraction.

## 1. Create Environment

```bash
conda create -n facemotion python=3.10 -y
conda activate facemotion

python -m pip install -U pip setuptools wheel
```

## 2. Install PyTorch

Check the GPU and CUDA driver first:

```bash
nvidia-smi
```

Install the PyTorch build that matches the server CUDA version. For example,
for CUDA 12.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

If your server uses a different CUDA version, choose the matching command from
the official PyTorch install selector.

## 3. Install Project Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` installs the common Python dependencies, including:

- Transformers / Accelerate for Qwen3.
- Diffusers from source for recent Qwen image-edit pipeline support.
- PEFT for loading the Lightning LoRA.
- MediaPipe for Face Landmarker blendshape extraction.

MediaPipe may also need system OpenGL/EGL runtime libraries. On Ubuntu/Debian
servers or containers, install them before running blendshape extraction:

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0 libegl1 libgles2
```

On older Ubuntu images, the package names may be:

```bash
apt-get install -y libgl1-mesa-glx libegl1-mesa libgles2-mesa
```

If you cannot use `apt-get`, try the conda-forge runtime libraries:

```bash
conda install -c conda-forge libgl libegl libgles -y
```

## 4. Prepare Models

The Qwen models are loaded from Hugging Face by default:

```text
Qwen/Qwen3-14B
Qwen/Qwen-Image-Edit-2511
lightx2v/Qwen-Image-Edit-2511-Lightning
```

MediaPipe requires a local Face Landmarker task model. Download it manually and
put it somewhere like:

```text
models/face_landmarker.task
```

The path is passed with `--model_path`; it is not hard-coded.

## 5. Run The Pipeline

### One Command

Run text-to-prompts, image editing, and MediaPipe blendshape extraction in one
workflow:

```bash
python -m facemotion.cli.run_blendshape_pipeline \
  --text "A person is smiling gently and then glancing sideways." \
  --image portrait.png \
  --model_path models/face_landmarker.task \
  --output_dir outputs/run_smile_glance \
  --gpu 7
```

This writes:

```text
outputs/run_smile_glance/prompts.json
outputs/run_smile_glance/states/000.png
outputs/run_smile_glance/states/001.png
outputs/run_smile_glance/states/002.png
outputs/run_smile_glance/blendshapes.json
```

On V100 32GB, the one-command workflow defaults to:

```text
--device-map auto
--torch-dtype float16
--offload sequential
```

### Step By Step

Generate the structured motion spec and edit prompts:

```bash
python -m facemotion.cli.make_prompts \
  --text "A person is smiling gently and then glancing sideways." \
  --output outputs/prompts.json \
  --gpu 0
```

If the 14B text model does not fit on one GPU, try:

```bash
python -m facemotion.cli.make_prompts \
  --text "A person is smiling gently and then glancing sideways." \
  --output outputs/prompts.json \
  --gpu 0 \
  --device-map auto
```

Generate edited key-state images:

```bash
python -m facemotion.cli.make_images \
  --image input_portrait.png \
  --prompts outputs/prompts.json \
  --output_dir outputs/states \
  --gpu 0
```

Extract MediaPipe blendshapes:

```bash
python -m facemotion.cli.extract_blendshapes \
  --image_dir outputs/states \
  --model_path models/face_landmarker.task \
  --output outputs/blendshapes.json
```

## 6. Memory Notes

For image editing, the default is:

```bash
--offload sequential
```

This uses `enable_sequential_cpu_offload()` to minimize GPU memory usage by
moving modules between CPU and GPU during inference. It is slower but safer on
memory-limited GPUs.

Use this for maximum speed when GPU memory is enough:

```bash
--offload none
```

Use this as a middle ground:

```bash
--offload model
```

## 7. MediaPipe Import Note

This repository may contain a local `mediapipe/` source checkout. Still install
the official Python wheel:

```bash
pip install mediapipe
```

The `extract_blendshapes` CLI is written to prefer the official installed
package for Face Landmarker runtime usage.

## 8. Troubleshooting

If `make_prompts` still tries to download:

```text
Qwen/Qwen3-14B-Instruct
```

then the server copy is stale. The default text model should be:

```text
Qwen/Qwen3-14B
```

Check it from the project root:

```bash
python - <<'PY'
from facemotion.motion.qwen import DEFAULT_CHAT_MODEL
print(DEFAULT_CHAT_MODEL)
PY
```

If it prints the old `Qwen/Qwen3-14B-Instruct`, update the code on the server
or temporarily override the model:

```bash
python -m facemotion.cli.make_prompts \
  --text "A person is smiling gently and then glancing sideways." \
  --output outputs/prompts.json \
  --gpu 7 \
  --model Qwen/Qwen3-14B
```

The warning from `hf auth login` about missing git credential helper is harmless
for downloading public/read-access models. It only affects saving credentials
for git operations such as pushing to the Hub.

If `extract_blendshapes` returns an error like:

```text
libGLESv2.so.2: cannot open shared object file: No such file or directory
```

install the OpenGL/EGL system libraries from step 3, then rerun:

```bash
python -m facemotion.cli.extract_blendshapes \
  --image_dir outputs/states \
  --model_path models/face_landmarker.task \
  --output outputs/blendshapes.json
```
