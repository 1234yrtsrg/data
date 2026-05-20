from facemotion import FacemotionBlendshapeWorkflow


TEXT = "A person is smiling gently and then glancing sideways."
IMAGE_PATH = "portrait.png"
MEDIAPIPE_MODEL_PATH = "models/face_landmarker.task"
OUTPUT_DIR = "outputs/run_smile_glance"
GPU = 7


def main() -> None:
    workflow = FacemotionBlendshapeWorkflow(gpu=GPU)
    result = workflow.run(
        text=TEXT,
        image_path=IMAGE_PATH,
        mediapipe_model_path=MEDIAPIPE_MODEL_PATH,
        output_dir=OUTPUT_DIR,
    )

    print(result.blendshapes_path)


if __name__ == "__main__":
    main()
