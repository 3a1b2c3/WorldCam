import argparse
import torch
import numpy as np
import os
import json
from pathlib import Path
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models import ModelManager, load_state_dict

# =============================================================================
# Configuration — edit these before running
# =============================================================================

# Paths (relative to the repository root, i.e. the directory containing this file)
REPO_ROOT          = Path(__file__).resolve().parent
TRAINED_MODEL_PATH = str(REPO_ROOT / "weights" / "finetuned_dit.safetensors")
TEST_DATA_DIR      = str(REPO_ROOT / "data")
PROMPT_JSON_PATH   = str(REPO_ROOT / "data" / "captions.json")
SAVE_PATH          = str(REPO_ROOT / "output")

# Which chunks to run (set to None to run all .mp4 files in TEST_DATA_DIR)
CHUNK_IDS = None

# Video settings
VIDEO_HEIGHT   = 480
VIDEO_WIDTH    = 832
COND_FRAMES    = 65    # number of conditioning frames fed to the model
OUTPUT_FPS     = 30
OUTPUT_QUALITY = 4   # 1 (low) – 10 (high)

# Inference settings
CFG_SCALE          = 4
SEED               = 0
GAME_DOMAIN_PREFIX         = "<A first-person shooter CS game> "
LONG_TERM_MEMORY_START      = 30   # AR step at which long-term memory retrieval begins
LONG_TERM_MEMORY_NUM_CLIPS  = 4    # number of memory clips to retrieve per step
LONG_TERM_MEMORY_REF_INDICES = [48, 52, 56, 60]  # extrinsic indices used as retrieval query
NEGATIVE_PROMPT    = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# =============================================================================


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=str, default=None,
                   help="Comma-separated chunk IDs to process (overrides CHUNK_IDS). "
                        "If omitted, runs all .mp4 files in TEST_DATA_DIR.")
    p.add_argument("--save-path", type=str, default=None,
                   help="Override output directory (defaults to SAVE_PATH).")
    return p.parse_args()


def main():
    args = parse_args()
    save_path = args.save_path if args.save_path else SAVE_PATH
    os.makedirs(save_path, exist_ok=True)

    # Load pipeline
    pipe = WanVideoPipeline(torch_dtype=torch.bfloat16, device="cuda")

    model_manager = ModelManager()
    base_model_configs = [
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="Wan2.1_VAE.pth"),
        ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="diffusion_pytorch_model.safetensors"),
    ]
    for cfg in base_model_configs:
        cfg.download_if_necessary()
        model_manager.load_model(cfg.path, device="cpu", torch_dtype=pipe.torch_dtype)

    pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    pipe.vae          = model_manager.fetch_model("wan_video_vae")
    pipe.dit          = model_manager.fetch_model("wan_video_dit")

    tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")
    tokenizer_config.download_if_necessary()
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(tokenizer_config.path)

    # Load fine-tuned DiT weights
    print(f"Loading fine-tuned weights from: {TRAINED_MODEL_PATH}")
    pipe.dit.load_state_dict(load_state_dict(TRAINED_MODEL_PATH, device="cpu"), strict=True)
    print("Fine-tuned weights loaded.")

    pipe.text_encoder.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.vae.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.to(pipe.device, dtype=pipe.torch_dtype)

    # Load captions
    with open(PROMPT_JSON_PATH, "r") as f:
        prompt_entries = json.load(f)
    caption_map = {entry["media_path"]: entry["caption"] for entry in prompt_entries}

    test_data_dir = Path(TEST_DATA_DIR)
    if args.chunks:
        chunk_ids = [c.strip() for c in args.chunks.split(",") if c.strip()]
    elif CHUNK_IDS is not None:
        chunk_ids = CHUNK_IDS
    else:
        chunk_ids = [f.stem for f in sorted(test_data_dir.iterdir()) if f.suffix == ".mp4"]

    for chunk_id in chunk_ids:
        file_path = test_data_dir / f"{chunk_id}.mp4"
        if not file_path.exists():
            print(f"Skipping {chunk_id}: .mp4 not found")
            continue

        intrinsics_path = test_data_dir / f"{chunk_id}_intrinsics_palindrome.npy"
        extrinsics_path = test_data_dir / f"{chunk_id}_poses_palindrome.npy"

        if not intrinsics_path.exists() or not extrinsics_path.exists():
            print(f"Skipping {chunk_id}: pose files not found")
            continue

        caption = caption_map.get(f"{chunk_id}.mp4", "")
        prompt  = GAME_DOMAIN_PREFIX + caption

        intrinsics = torch.from_numpy(np.load(intrinsics_path)).to(pipe.device, dtype=torch.float32)[None, ...]
        extrinsics = torch.from_numpy(np.load(extrinsics_path)).to(pipe.device, dtype=torch.float32)[None, ...]

        cond_video = VideoData(str(file_path), height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        cond_video.length = COND_FRAMES

        print(f"Processing {chunk_id} | prompt: {prompt}")

        video = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            input_video=cond_video,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            cfg_scale=CFG_SCALE,
            seed=SEED,
            tiled=True,
            long_term_memory_start_step=LONG_TERM_MEMORY_START,
            long_term_memory_num_clips=LONG_TERM_MEMORY_NUM_CLIPS,
        )

        output_file = str(Path(save_path) / f"{chunk_id}_output.mp4")
        save_video(video, output_file, fps=OUTPUT_FPS, quality=OUTPUT_QUALITY)
        print(f"Saved -> {output_file}")


if __name__ == "__main__":
    main()
