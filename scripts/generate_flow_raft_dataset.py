"""
Generate a synthetic RAFT fine-tuning dataset for LEED.

Outputs:
  data/flow_raft20000/
    train/images/{id:04d}_0.png, {id:04d}_1.png
    train/flows/{id:04d}.npy
    val/images/...
    val/flows/...
    test/images/...
    test/flows/...

We generate 10k pairs (20k images) split 0.8/0.1/0.1 by id ranges:
  0000–7999 train, 8000–8999 val, 9000–9999 test.
Each id uses seed=id for reproducibility and one of several presets.
"""

from pathlib import Path
import shutil
import random
from dataclasses import replace
import sys

# Ensure repo root is on PYTHONPATH
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from trackleed.ai.synth import SynthConfig, generate_sequence


DATA_ROOT = Path("data/flow_raft20000")
TMP_ROOT = DATA_ROOT / "tmp_gen"

# Train/val/test ranges by id
SPLITS = {
    "train": range(0, 8000),
    "val": range(8000, 9000),
    "test": range(9000, 10000),
}


def preset_typical(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.5)
    cfg.radial_expansion_per_frame = random.uniform(0.0, 0.5)
    cfg.bg_grad_amplitude = random.uniform(0.0, 0.02)
    cfg.noise_level = random.uniform(0.01, 0.03)
    return cfg


def preset_static(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.0, 0.2)
    cfg.radial_expansion_per_frame = 0.0
    cfg.noise_level = random.uniform(0.005, 0.02)
    return cfg


def preset_dimmed(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.0)
    cfg.noise_level = random.uniform(0.03, 0.06)
    cfg.intensity_range = (0.3, 0.8)
    cfg.blink_period = random.choice([8, 12, 16])
    cfg.blink_duty_cycle = random.uniform(0.3, 0.6)
    return cfg


def preset_moire(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.0)
    cfg.grid_mode = random.choice(["stripe", "checker"])
    cfg.grid_amplitude = random.uniform(0.05, 0.12)
    cfg.grid_period_x = random.uniform(6.0, 12.0)
    cfg.grid_period_y = random.uniform(6.0, 12.0)
    return cfg


def preset_halo(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.5)
    cfg.bg_spot_amplitude = random.uniform(0.02, 0.08)
    cfg.bg_spot_count = random.randint(2, 8)
    cfg.bg_quad_amplitude = random.uniform(0.0, 0.02)
    return cfg


def preset_large_motion(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(1.5, 3.0)
    cfg.radial_expansion_per_frame = random.uniform(0.5, 1.0)
    cfg.noise_level = random.uniform(0.01, 0.03)
    return cfg


def preset_intensity_change(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.0)
    cfg.intensity_wave = random.choice(["sin", "triangle"])
    cfg.intensity_wave_period = random.uniform(20.0, 60.0)
    cfg.blink_period = None
    cfg.blink_duty_cycle = 0.5
    return cfg


def preset_satellites(base: SynthConfig) -> SynthConfig:
    cfg = replace(base)
    cfg.drift_per_frame = random.uniform(0.5, 1.0)
    cfg.satellite_probability = random.uniform(0.3, 0.6)
    cfg.satellite_intensity_ratio_range = (0.2, 0.5)
    cfg.satellite_radius_offset = (5.0, 15.0)
    cfg.satellite_angle_offset = (-0.3, 0.3)
    return cfg


PRESETS = [
    preset_typical,
    preset_static,
    preset_dimmed,
    preset_moire,
    preset_halo,
    preset_large_motion,
    preset_intensity_change,
    preset_satellites,
]


def make_base_config(seed: int) -> SynthConfig:
    return SynthConfig(
        frames=2,
        frame_size=256,
        spots_per_ring=8,
        num_rings=1,
        ring_radius=80.0,
        drift_per_frame=0.5,
        radial_expansion_per_frame=0.2,
        noise_level=0.02,
        seed=seed,
        bg_base_level=0.05,
        bg_wave_amplitude=0.0,
        bg_wave_period=200,
        bg_wave_mode="none",
        bg_grad_amplitude=0.0,
        bg_quad_amplitude=0.0,
        bg_spot_amplitude=0.0,
        bg_spot_count=0,
        grid_mode="none",
        grid_amplitude=0.05,
        grid_period_x=16.0,
        grid_period_y=16.0,
        intensity_range=(0.8, 1.2),
    )


def generate_pair(sample_id: int):
    # pick preset deterministically from id
    random.seed(sample_id)
    preset_fn = PRESETS[sample_id % len(PRESETS)]
    cfg = preset_fn(make_base_config(sample_id))
    tmp_dir = TMP_ROOT / f"sample_{sample_id:04d}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    generate_sequence(
        config=cfg,
        output_dir=tmp_dir,
        save_mask=False,
        save_yolo=False,
        save_flow=True,
        image_format="png",
    )
    # pick split
    split = None
    for name, rng in SPLITS.items():
        if sample_id in rng:
            split = name
            break
    if split is None:
        split = "train"
    split_img_dir = DATA_ROOT / split / "images"
    split_flow_dir = DATA_ROOT / split / "flows"
    split_img_dir.mkdir(parents=True, exist_ok=True)
    split_flow_dir.mkdir(parents=True, exist_ok=True)

    img0 = tmp_dir / "synthetic_0000.png"
    img1 = tmp_dir / "synthetic_0001.png"
    flow = tmp_dir / "flows" / "synthetic_0000.npy"
    shutil.move(img0, split_img_dir / f"{sample_id:04d}_0.png")
    shutil.move(img1, split_img_dir / f"{sample_id:04d}_1.png")
    shutil.move(flow, split_flow_dir / f"{sample_id:04d}.npy")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    all_ids = list(range(0, 10000))
    for sid in all_ids:
        generate_pair(sid)
        if sid % 50 == 0:
            print(f"{sid+1}/10000 generated...")
    print(f"Generated {len(all_ids)} pairs into {DATA_ROOT}")


if __name__ == "__main__":
    main()
