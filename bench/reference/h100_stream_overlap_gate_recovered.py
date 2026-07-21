#!/usr/bin/env python3
"""Recovered Phase-1 H100 CF++/rolling-TAEHV measurement runner.

This is the exact session-created runner, with the subsequently applied
startup-trim fix folded in. It is preserved as executable evidence and is
intended to be copied into a clean checkout of thu-ml/Causal-Forcing at
8db419e341e5fc52542c0b2c4542728420ddfb4a. It is not a standalone module in
this repository.

The historical runner predates quality-run manifest v1 and therefore does not
record checkpoint, initial-noise, latent, source-diff, or media hashes. A new
quality sweep must wrap or extend it to emit those fields before paid runs.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from demo_utils.memory import gpu
from demo_utils.taehv import TAEHV
from omegaconf import OmegaConf
from pipeline import CausalInferencePipeline
from torchvision.io import write_video
from utils.misc import set_seed


class LatentPassthrough(torch.nn.Module):
    def decode_to_pixel(self, latents, use_cache=False):
        return latents


parser = argparse.ArgumentParser()
parser.add_argument("--step", choices=["1", "2"], required=True)
parser.add_argument("--mode", choices=["serial", "overlap"], required=True)
parser.add_argument("--trials", type=int, default=4)
args = parser.parse_args()
root = Path(__file__).resolve().parent
config = OmegaConf.merge(
    OmegaConf.load(root / "configs" / "default_config.yaml"),
    OmegaConf.load(root / "configs" / f"causal_forcing_dmd_framewise_{args.step}step.yaml"),
)
set_seed(20260719)
torch.set_grad_enabled(False)
pipeline = CausalInferencePipeline(config, device=torch.device("cuda"))
state_dict = torch.load(
    root / "checkpoints" / "causal-forcing++" / f"framewise-{args.step}step.pt",
    map_location="cpu",
)
gen_sd = state_dict["generator_ema"]
try:
    pipeline.generator.load_state_dict(gen_sd)
except RuntimeError:
    fixed = {}
    for key, value in gen_sd.items():
        if key.startswith("model._fsdp_wrapped_module."):
            key = key.replace("model._fsdp_wrapped_module.", "model.", 1)
        fixed[key] = value
    pipeline.generator.load_state_dict(fixed, strict=False)
pipeline.vae = LatentPassthrough()
pipeline = pipeline.to(dtype=torch.bfloat16)
pipeline.text_encoder.to(gpu)
pipeline.generator.to(gpu)
pipeline.eval()
taehv = TAEHV(checkpoint_path=str(root / "checkpoints" / "taew2_1.pth")).to(
    device=gpu, dtype=torch.float16
).eval()

prompt = (
    "A lone red fox runs through a snowy pine forest at sunrise, cinematic "
    "tracking shot, realistic fur, stable anatomy."
)
forward_count = 0


def count_forward(_module, _inputs, _output):
    global forward_count
    forward_count += 1


hook = pipeline.generator.register_forward_hook(count_forward)


def reset_caches():
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(1, torch.bfloat16, gpu)
        pipeline._initialize_crossattn_cache(1, torch.bfloat16, gpu)
        return
    for cache in pipeline.crossattn_cache:
        cache["is_init"] = False
    for cache in pipeline.kv_cache1:
        cache["global_end_index"].zero_()
        cache["local_end_index"].zero_()


results = []
last_video = None
for trial in range(args.trials):
    set_seed(20260719)
    generator = torch.Generator(device="cuda").manual_seed(20260719)
    noise = torch.randn(
        (1, 21, 16, 60, 104),
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    before_forwards = forward_count
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gen_stream = torch.cuda.default_stream()
    vae_stream = gen_stream if args.mode == "serial" else torch.cuda.Stream()
    start_event = torch.cuda.Event(enable_timing=True)
    text_done_event = torch.cuda.Event(enable_timing=True)
    rgb_events = []
    output_chunks = []
    wall_start = time.perf_counter()
    start_event.record(gen_stream)
    with torch.cuda.stream(gen_stream):
        conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
        for key, value in conditional_dict.items():
            conditional_dict[key] = value.to(dtype=torch.bfloat16)
        text_done_event.record(gen_stream)
        reset_caches()
        current_start_frame = 0
        vae_tail = None
        for block_index in range(21):
            noisy_input = noise[:, current_start_frame : current_start_frame + 1]
            current_steps = (
                pipeline.denoising_step_list_first_chunk
                if block_index == 0
                and pipeline.denoising_step_list_first_chunk is not None
                else pipeline.denoising_step_list
            )
            for index, current_timestep in enumerate(current_steps):
                timestep = (
                    torch.ones((1, 1), device=gpu, dtype=torch.int64)
                    * current_timestep
                )
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
                if index < len(current_steps) - 1:
                    next_timestep = current_steps[index + 1]
                    noisy_input = pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep
                        * torch.ones((1,), device=gpu, dtype=torch.long),
                    ).unflatten(0, denoised_pred.shape[:2])
            context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
            pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
            latent_ready = torch.cuda.Event()
            latent_ready.record(gen_stream)
            with torch.cuda.stream(vae_stream):
                vae_stream.wait_event(latent_ready)
                current_latent = denoised_pred.to(torch.float16)
                prior_context_latents = (
                    0 if vae_tail is None else int(vae_tail.shape[1])
                )
                if vae_tail is None:
                    decode_input = current_latent
                else:
                    decode_input = torch.cat((vae_tail, current_latent), dim=1)
                vae_tail = decode_input[:, -3:]
                pixels_untrimmed = taehv.decode_video(
                    decode_input, parallel=True, show_progress_bar=False
                )
                trim_frames = (
                    3 if block_index == 0 else prior_context_latents * 4
                )
                pixels = pixels_untrimmed[:, trim_frames:]
                output_chunks.append(pixels)
                rgb_ready = torch.cuda.Event(enable_timing=True)
                rgb_ready.record(vae_stream)
                rgb_events.append(rgb_ready)
            current_start_frame += 1
    vae_stream.synchronize()
    gen_stream.synchronize()
    wall_s = time.perf_counter() - wall_start
    video = torch.cat(output_chunks, dim=1)
    assert video.shape == (1, 81, 3, 480, 832), video.shape
    text_s = start_event.elapsed_time(text_done_event) / 1000.0
    rgb_ready_s = [start_event.elapsed_time(event) / 1000.0 for event in rgb_events]
    chunk_intervals = [
        rgb_ready_s[index] - rgb_ready_s[index - 1]
        for index in range(1, len(rgb_ready_s))
    ]
    effective_frame_intervals = [interval / 4.0 for interval in chunk_intervals]
    p95_frame_interval = sorted(effective_frame_intervals)[
        int(0.95 * (len(effective_frame_intervals) - 1))
    ]
    row = {
        "trial": trial,
        "warmup": trial == 0,
        "mode": args.mode,
        "pixel_frames": 81,
        "forwards": forward_count - before_forwards,
        "text_encode_s": text_s,
        "first_visible_rgb_s": rgb_ready_s[0],
        "last_rgb_s_cuda": rgb_ready_s[-1],
        "wall_e2e_s": wall_s,
        "e2e_fps": 81 / wall_s,
        "mean_effective_frame_interval_ms": statistics.mean(
            effective_frame_intervals
        )
        * 1000,
        "p95_effective_frame_interval_ms": p95_frame_interval * 1000,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 2**30,
    }
    print("METRIC", json.dumps(row), flush=True)
    results.append(row)
    last_video = video

hook.remove()
steady = results[1:]
summary = {
    "gpu": torch.cuda.get_device_name(),
    "step": int(args.step),
    "mode": args.mode,
    "decoder": "TAEHV taew2_1, rolling 3-latent context, per-block decode",
    "resolution": "480x832",
    "prompt": prompt,
    "trials": results,
    "steady_mean_e2e_fps": statistics.mean(
        result["e2e_fps"] for result in steady
    ),
    "steady_mean_first_visible_rgb_s": statistics.mean(
        result["first_visible_rgb_s"] for result in steady
    ),
    "steady_mean_p95_effective_frame_interval_ms": statistics.mean(
        result["p95_effective_frame_interval_ms"] for result in steady
    ),
}
metrics_path = root.parent / f"h100_cf{args.step}_taehv_{args.mode}_metrics.json"
metrics_path.write_text(json.dumps(summary, indent=2) + "\n")
video_path = root.parent / f"h100_cf{args.step}_taehv_{args.mode}.mp4"
video_u8 = (
    last_video[0]
    .clamp(0, 1)
    .mul(255)
    .round()
    .to(torch.uint8)
    .permute(0, 2, 3, 1)
    .cpu()
)
write_video(
    str(video_path),
    video_u8,
    fps=16,
    video_codec="libx264",
    options={"crf": "18"},
)
print("SUMMARY", json.dumps(summary), flush=True)
print("WROTE", metrics_path, video_path, flush=True)
