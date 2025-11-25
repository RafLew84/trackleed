"""
Scaffolding for RAFT optical flow integration.

This module will load a RAFT checkpoint and expose a compute_flow helper.
Full logic (forward–backward consistency, resizing, LK fallback) will be added in later steps.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch


@dataclass
class RAFTConfig:
    model_path: Path
    device: str = "cuda"
    resize_short_side: int = 512
    fwd_bwd_tau: float = 1.5
    flow_conf_threshold: float = 0.0
    max_step_px: Optional[float] = None


class RAFTWrapper:
    """Lazy RAFT loader; actual inference will be implemented in later steps."""

    def __init__(self, cfg: RAFTConfig):
        self.cfg = cfg
        self.model = None

    def load(self):
        if self.model is not None:
            return self.model
        model_path = Path(self.cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"RAFT checkpoint not found: {model_path}")
        # Placeholder: actual RAFT architecture import to be added.
        raise NotImplementedError("RAFT model loading not yet implemented.")

    def compute_flow(self, image1, image2) -> Tuple:
        """
        Compute dense flow between image1 and image2.
        Will return (flow, confidence_mask) once implemented.
        """
        raise NotImplementedError("RAFT flow computation not yet implemented.")


def load_raft(cfg: RAFTConfig) -> RAFTWrapper:
    """Factory for RAFTWrapper."""
    return RAFTWrapper(cfg)
