import argparse
from pathlib import Path
from diffsynth.pipelines.wan_video_new import ModelConfig

REPO_ROOT = Path(__file__).resolve().parent

MODEL_CONFIGS = [
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Download base model weights for WorldCam.")
    p.add_argument(
        "--local-model-path",
        type=str,
        default=str(REPO_ROOT / "models"),
        help="Directory to store downloaded models (default: ./models).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    for cfg in MODEL_CONFIGS:
        cfg.local_model_path = args.local_model_path
        print(f"Downloading {cfg.model_id} / {cfg.origin_file_pattern} ...")
        cfg.download_if_necessary()
        print(f"  -> {cfg.path}")
    print("Done.")


if __name__ == "__main__":
    main()
