"""
RAFT optical flow wrapper with forward–backward consistency.

Requires the RAFT code on PYTHONPATH (e.g., cloned repo with /core on sys.path).
"""

from dataclasses import dataclass
from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
import importlib.util
import cv2


@dataclass
class RAFTConfig:
    model_path: Path
    device: str = "cuda"
    resize_short_side: int = 512
    fwd_bwd_tau: float = 1.5
    flow_conf_threshold: float = 0.0
    max_step_px: Optional[float] = None
    small: bool = True
    alternate_corr: bool = False
    mixed_precision: bool = False
    dropout: float = 0.0


class RAFTWrapper:
    """Lazy RAFT loader and flow computer."""

    def __init__(self, cfg: RAFTConfig):
        self.cfg = cfg
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RAFT checkpoint not found: {model_path}")

        # Auto-detect small/full from filename if not explicitly set
        name_lower = model_path.name.lower()
        if "small" in name_lower:
            self.cfg.small = True
            self.cfg.alternate_corr = False
        else:
            self.cfg.small = False
            self.cfg.alternate_corr = True
        # Disable alternate_corr if extension not available
        if self.cfg.alternate_corr:
            if importlib.util.find_spec("alt_cuda_corr") is None:
                print("[RAFT] alt_cuda_corr not found, disabling alternate_corr.")
                self.cfg.alternate_corr = False

        try:
            # Try to auto-append common RAFT locations if not already on path
            extra_paths = [
                os.environ.get("RAFT_CORE_PATH"),
                Path("third_party/RAFT/core"),
                Path("RAFT/core"),
                Path.cwd() / "third_party" / "RAFT" / "core",
                Path.cwd() / "RAFT" / "core",
            ]
            for p in extra_paths:
                if p and Path(p).exists() and str(p) not in sys.path:
                    sys.path.append(str(p))
            from raft import RAFT  # type: ignore
        except Exception as exc:
            raise ImportError(
                "RAFT code not found on PYTHONPATH. Set RAFT_CORE_PATH to .../RAFT/core or clone RAFT under third_party/RAFT."
            ) from exc

        args = Namespace(
            small=self.cfg.small,
            mixed_precision=self.cfg.mixed_precision,
            dropout=self.cfg.dropout,
            alternate_corr=self.cfg.alternate_corr,
        )
        model = RAFT(args)
        raw = torch.load(model_path, map_location="cpu")
        if "model" in raw:
            state = raw["model"]
        elif "state_dict" in raw:
            state = raw["state_dict"]
        else:
            state = raw
        state = {k.replace("module.", ""): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[RAFT] Missing keys when loading: {missing}")
        if unexpected:
            print(f"[RAFT] Unexpected keys ignored: {unexpected}")
        model.to(self.cfg.device)
        model.eval()
        self.model = model
        return self.model

    @torch.no_grad()
    def compute_flow(self, image1: np.ndarray, image2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dense flow between image1 and image2.
        Returns (flow_hw2, conf_mask) in numpy, flow in pixels relative to original resolution.
        """
        model = self.load()
        dev = self.cfg.device
        # 1) Wspólny crop do min rozmiaru, żeby uniknąć off-by-one
        h0, w0 = image1.shape[:2]
        h1, w1 = image2.shape[:2]
        h = min(h0, h1)
        w = min(w0, w1)
        image1 = image1[:h, :w]
        image2 = image2[:h, :w]

        # 2) Wspólny resize krótszego boku (opcjonalny)
        scale = 1.0
        if self.cfg.resize_short_side and min(h, w) != self.cfg.resize_short_side:
            scale = float(self.cfg.resize_short_side) / float(min(h, w))
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            image1 = cv2.resize(image1, (new_w, new_h), interpolation=cv2.INTER_AREA)
            image2 = cv2.resize(image2, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            new_h, new_w = h, w

        # 3) Tensory + padder
        t1 = self._prep_image(image1, dev)
        t2 = self._prep_image(image2, dev)
        try:
            from utils.utils import InputPadder
        except Exception:
            from raft.utils.utils import InputPadder  # depending on repo layout
        padder = InputPadder(t1.shape)
        t1, t2 = padder.pad(t1, t2)

        with torch.no_grad():
            flow12, _ = model(t1, t2, iters=20, test_mode=True)
            flow21, _ = model(t2, t1, iters=20, test_mode=True)

        flow12 = padder.unpad(flow12)[0]  # (2,H,W)
        flow21 = padder.unpad(flow21)[0]

        # Forward-backward consistency
        try:
            fb_mask = self._forward_backward_mask(flow12, flow21, tau=self.cfg.fwd_bwd_tau)
        except Exception as exc:
            print(f"[RAFT] FB check failed ({exc}), using all-valid mask.")
            fb_mask = torch.ones_like(flow12[0], dtype=torch.bool, device=flow12.device)

        # 4) Do numpy
        flow_np = flow12.permute(1, 2, 0).cpu().numpy()  # (H,W,2)
        fb_mask_np = fb_mask.cpu().numpy()

        # 5) Jeśli był resize, przeskaluj flow do oryginalnego (po crop) rozmiaru
        if (new_h, new_w) != (h, w):
            flow_np = cv2.resize(flow_np, (w, h), interpolation=cv2.INTER_LINEAR)
            flow_np[..., 0] *= (w / float(new_w))
            flow_np[..., 1] *= (h / float(new_h))
            fb_mask_np = cv2.resize(fb_mask_np.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        if self.cfg.max_step_px:
            mag = np.linalg.norm(flow_np, axis=2)
            fb_mask_np &= mag <= float(self.cfg.max_step_px)

        return flow_np, fb_mask_np

    def _prep_image(self, img: np.ndarray, device: str):
        """Normalize to [0,1], force 3ch, add batch dim."""
        if img.ndim == 2:
            img = img[..., None]
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img_float = img.astype(np.float32)
        img_float = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-6)
        t = torch.from_numpy(img_float).permute(2, 0, 1)[None].to(device)
        return t

    def _forward_backward_mask(self, flow12: torch.Tensor, flow21: torch.Tensor, tau: float):
        """
        Basic forward-backward consistency check.
        flow12: (2,H,W) from I1->I2
        flow21: (2,H,W) from I2->I1
        """
        _, H, W = flow12.shape
        # create grid
        y, x = torch.meshgrid(torch.arange(H, device=flow12.device), torch.arange(W, device=flow12.device), indexing="ij")
        coords = torch.stack([x, y], dim=0).float()  # (2,H,W)
        # forward coords
        coords_fwd = coords + flow12
        # normalize grid for sampling
        grid = torch.stack(
            [
                2.0 * coords_fwd[0] / max(W - 1, 1) - 1.0,
                2.0 * coords_fwd[1] / max(H - 1, 1) - 1.0,
            ],
            dim=-1,
        )
        try:
            flow21_sampled = F.grid_sample(flow21[None], grid[None], mode="bilinear", padding_mode="border", align_corners=True)[0]
            fb_sum = flow12 + flow21_sampled
            fb_err = torch.linalg.norm(fb_sum, dim=0)
            mask = fb_err <= tau
            return mask
        except Exception as exc:
            print(f"[RAFT] grid_sample failed in FB check: {exc}")
            return torch.ones_like(flow12[0], dtype=torch.bool, device=flow12.device)


def load_raft(cfg: RAFTConfig) -> RAFTWrapper:
    """Factory for RAFTWrapper."""
    return RAFTWrapper(cfg)
