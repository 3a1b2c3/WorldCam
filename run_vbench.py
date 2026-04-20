import csv
import json
import re
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

SCRIPT_DIR          = Path(__file__).resolve().parent
TRAINED_MODEL_PATH  = str(SCRIPT_DIR / "weights" / "finetuned_dit.safetensors")
_VBENCH_ROOT        = SCRIPT_DIR.parent / "VBench" / "vbench2_beta_i2v" / "vbench2_beta_i2v" / "data"
_DEFAULT_INFO_JSON  = _VBENCH_ROOT / "i2v-bench-info.json"
_DEFAULT_CROP_DIR   = _VBENCH_ROOT / "crop"

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

# WorldCam conditioning: 65 frames → 16 latents (8 cond + 8 generated, hardcoded in pipeline)
COND_FRAMES = 65
# Pose frames: must exceed condition_num*4 + num_ar_steps*4 + 100 pipeline padding
NUM_POSE_FRAMES = 350


def _safe(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", s)[:150]


def _make_identity_poses(num_frames: int, height: int, width: int):
    """Identity extrinsics and pinhole intrinsics for each frame."""
    extrinsics = np.tile(np.eye(4, dtype=np.float32)[None], (num_frames, 1, 1))
    f  = max(height, width) * 0.87
    cx = width  / 2.0
    cy = height / 2.0
    intrinsics = np.tile(
        np.array([f, f, cx, cy], dtype=np.float32)[None],
        (num_frames, 1),
    )
    return intrinsics, extrinsics


def _build_pipeline():
    from diffsynth.models import ModelManager, load_state_dict
    from diffsynth.pipelines.wan_video_new import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline(torch_dtype=torch.bfloat16, device="cuda")
    model_manager = ModelManager()
    for pattern in [
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.1_VAE.pth",
        "diffusion_pytorch_model.safetensors",
    ]:
        cfg = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern=pattern)
        cfg.download_if_necessary()
        model_manager.load_model(cfg.path, device="cpu", torch_dtype=pipe.torch_dtype)

    pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
    pipe.vae          = model_manager.fetch_model("wan_video_vae")
    pipe.dit          = model_manager.fetch_model("wan_video_dit")

    tok_cfg = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*")
    tok_cfg.download_if_necessary()
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(tok_cfg.path)

    from diffsynth.models import load_state_dict
    print(f"[vbench] Loading fine-tuned weights: {TRAINED_MODEL_PATH}")
    pipe.dit.load_state_dict(load_state_dict(TRAINED_MODEL_PATH, device="cpu"), strict=True)

    pipe.text_encoder.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.vae.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.to(pipe.device, dtype=pipe.torch_dtype)
    return pipe


def vbench_batch(
    output_dir: str = "results_vbench/videos",
    num_samples: int = 5,
    seed: int = 42,
    image_types: str = "indoor,scenery",
    vbench_info_json: str = None,
    crop_dir: str = None,
    height: int = 480,
    width: int = 832,
    num_ar_steps: int = 50,
    skip_existing: bool = True,
    cfg_scale: float = 4.0,
    long_term_memory_start: int = 30,
    long_term_memory_num_clips: int = 4,
):
    from diffsynth import save_video

    info_json = Path(vbench_info_json) if vbench_info_json else _DEFAULT_INFO_JSON
    image_dir = (Path(crop_dir) if crop_dir else _DEFAULT_CROP_DIR) / "1-1"
    out_dir   = SCRIPT_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_path = out_dir.parent / "vbench_stats.csv"
    is_new     = not stats_path.exists()
    stats_f    = open(stats_path, "a", newline="", encoding="utf-8")
    stats_w    = csv.writer(stats_f)
    if is_new:
        stats_w.writerow(["task_idx", "prompt", "sample_idx", "duration_s", "ar_steps_per_s", "ram_gb", "vram_gb", "out_path", "status"])

    with open(info_json, encoding="utf-8") as f:
        entries = json.load(f)

    allowed = {t.strip() for t in image_types.split(",") if t.strip()} if image_types else None
    seen, prompts = set(), []
    for e in entries:
        name = e["file_name"]
        if name in seen:
            continue
        if allowed and e.get("type") not in allowed:
            continue
        seen.add(name)
        prompts.append((name, e.get("caption", Path(name).stem)))

    print(f"[vbench] {len(prompts)} prompts × {num_samples} = {len(prompts) * num_samples} total")

    pipe = _build_pipeline()

    # Pre-build constant camera tensors (identity = stationary camera)
    K_np, P_np = _make_identity_poses(NUM_POSE_FRAMES, height, width)
    intrinsics = torch.from_numpy(K_np).unsqueeze(0).cuda()   # (1, N, 4)
    extrinsics = torch.from_numpy(P_np).unsqueeze(0).cuda()   # (1, N, 4, 4)

    skipped = generated = errors = done = 0
    total   = len(prompts) * num_samples
    t_start = time.time()

    for task_idx, (image_name, prompt) in enumerate(prompts):
        image_path = image_dir / image_name
        if not image_path.is_file():
            print(f"[vbench] skip {task_idx}: not found — {image_path}")
            continue

        img        = Image.open(image_path).convert("RGB")
        cond_video = [img] * COND_FRAMES  # tile single image into conditioning clip

        for sample_idx in range(num_samples):
            sample_seed = seed + sample_idx
            out_path    = out_dir / f"{_safe(prompt)}-{sample_idx}-{sample_seed}.mp4"

            if skip_existing and out_path.exists():
                skipped += 1
                done    += 1
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", "", str(out_path), "skipped"])
                stats_f.flush()
                continue

            pct = 100 * done / total if total else 0
            eta = ""
            if done > 0:
                secs = (time.time() - t_start) / done * (total - done)
                eta  = f"  ETA {int(secs//3600):02d}h{int(secs%3600//60):02d}m{int(secs%60):02d}s"
            print(f"[vbench] [{done+1}/{total}  {pct:.0f}%{eta}]  task {task_idx+1}  sample {sample_idx+1}: {prompt[:60]}")

            torch.manual_seed(sample_seed)
            try:
                with torch.inference_mode():
                    st     = time.time()
                    frames = pipe(
                        prompt=prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        input_video=cond_video,
                        intrinsics=intrinsics,
                        extrinsics=extrinsics,
                        cfg_scale=cfg_scale,
                        seed=sample_seed,
                        height=height,
                        width=width,
                        tiled=True,
                        num_ar_steps=num_ar_steps,
                        long_term_memory_start_step=long_term_memory_start,
                        long_term_memory_num_clips=long_term_memory_num_clips,
                        attention_sink_inference=False,
                    )
                    ed = time.time()

                duration    = ed - st
                ar_steps_ps = num_ar_steps / duration
                save_video(frames, str(out_path), fps=30, quality=4)
                ram_gb  = psutil.Process().memory_info().rss / 1024**3
                vram_gb = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
                print(f"[vbench] saved {out_path.name}  ({ar_steps_ps:.2f} AR-steps/s  RAM {ram_gb:.1f}GB  VRAM {vram_gb:.1f}GB)")
                stats_w.writerow([task_idx, prompt, sample_idx, f"{duration:.2f}", f"{ar_steps_ps:.2f}", f"{ram_gb:.2f}", f"{vram_gb:.2f}", str(out_path), "ok"])
                stats_f.flush()
                generated += 1
            except Exception as exc:
                traceback.print_exc()
                print(f"[vbench] ERROR task {task_idx} sample {sample_idx}: {exc}")
                stats_w.writerow([task_idx, prompt, sample_idx, "", "", "", "", str(out_path), "error"])
                stats_f.flush()
                errors += 1
            done += 1

    stats_f.close()
    elapsed = time.time() - t_start
    print(f"\n[vbench] done — generated={generated}  skipped={skipped}  errors={errors}  elapsed={elapsed/60:.1f}m")
    print(f"[vbench] stats → {stats_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir",               default="results_vbench/videos")
    p.add_argument("--num_samples",              type=int,   default=5)
    p.add_argument("--seed",                     type=int,   default=42)
    p.add_argument("--image_types",              default="indoor,scenery")
    p.add_argument("--vbench_info_json",         default=None)
    p.add_argument("--crop_dir",                 default=None)
    p.add_argument("--height",                   type=int,   default=480)
    p.add_argument("--width",                    type=int,   default=832)
    p.add_argument("--num_ar_steps",             type=int,   default=50)
    p.add_argument("--skip_existing",            type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--cfg_scale",                type=float, default=4.0)
    p.add_argument("--long_term_memory_start",   type=int,   default=30)
    p.add_argument("--long_term_memory_num_clips", type=int, default=4)
    args = p.parse_args()
    vbench_batch(**vars(args))
