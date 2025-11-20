"""Deep-learning spot detection helper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PyQt6.QtCore import QRect
from skimage import measure
from skimage.color import rgb2gray
from scipy import ndimage as ndi

from trackleed.core.data_models import SpotROI


class DLSpotDetectionError(RuntimeError):
    """Raised when the DL spot detector cannot run."""


class ResidualDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class UNetV2(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 48):
        super().__init__()
        self.down1 = ResidualDoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ResidualDoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ResidualDoubleConv(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = ResidualDoubleConv(base_ch * 4, base_ch * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            ResidualDoubleConv(base_ch * 8, base_ch * 16),
            nn.Dropout2d(p=0.1),
        )

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ResidualDoubleConv(base_ch * 16, base_ch * 8)
        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ResidualDoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ResidualDoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ResidualDoubleConv(base_ch * 2, base_ch)
        self.out_conv = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.down1(x)
        c2 = self.down2(self.pool1(c1))
        c3 = self.down3(self.pool2(c2))
        c4 = self.down4(self.pool3(c3))
        c5 = self.bottleneck(self.pool4(c4))

        u4 = torch.cat([self.up4(c5), c4], dim=1)
        c4 = self.dec4(u4)
        u3 = torch.cat([self.up3(c4), c3], dim=1)
        c3 = self.dec3(u3)
        u2 = torch.cat([self.up2(c3), c2], dim=1)
        c2 = self.dec2(u2)
        u1 = torch.cat([self.up1(c2), c1], dim=1)
        c1 = self.dec1(u1)
        return self.out_conv(c1)


_MODEL_CACHE: Dict[str, nn.Module] = {}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_spots_with_model(
    image: np.ndarray,
    metadata_mask: List[QRect],
    settings: dict,
) -> List[SpotROI]:
    model_path = settings.get("model_path")
    if not model_path:
        raise DLSpotDetectionError("No model selected for deep-learning spot detection.")

    model = _load_model(Path(model_path))
    model.eval()

    prepared = _prepare_image(image)
    padded, pad_info = _pad_to_multiple(prepared, multiple=16)
    tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(_DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    probs = _unpad(probs, pad_info)

    conf = float(settings.get("confidence", 0.5))
    binary_mask = probs >= conf

    if metadata_mask:
        binary_mask = _apply_metadata_mask(binary_mask, metadata_mask)

    dilate_radius = int(settings.get("dilate_radius_px", 0))
    if dilate_radius > 0:
        structure = ndi.generate_binary_structure(2, 1)
        for _ in range(dilate_radius):
            binary_mask = ndi.binary_dilation(binary_mask, structure=structure)

    labeled, num = ndi.label(binary_mask)
    if num == 0:
        return []

    min_area = int(settings.get("min_area_px", 1))
    max_area = int(settings.get("max_area_px", 100000))
    bbox_margin = int(settings.get("bbox_margin_px", 0))

    height, width = binary_mask.shape
    rois: List[SpotROI] = []
    for region in measure.regionprops(labeled, intensity_image=probs):
        if region.area < min_area or region.area > max_area:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        x0 = max(0, min_col - bbox_margin)
        y0 = max(0, min_row - bbox_margin)
        x1 = min(width, max_col + bbox_margin)
        y1 = min(height, max_row + bbox_margin)
        rect = QRect(x0, y0, x1 - x0, y1 - y0)
        cy, cx = region.weighted_centroid if region.weighted_centroid is not None else region.centroid
        intensity = float(region.max_intensity) if region.max_intensity is not None else 0.0
        spot = SpotROI(
            id=len(rois) + 1,
            roi_rect=rect,
            center_x=float(cx),
            center_y=float(cy),
            intensity=intensity,
            source="dl",
        )
        rois.append(spot)

    return rois


def _prepare_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = rgb2gray(image)
    arr = image.astype(np.float32)
    min_v = float(arr.min())
    max_v = float(arr.max())
    if max_v > min_v:
        arr = (arr - min_v) / (max_v - min_v)
    else:
        arr.fill(0.0)
    return arr


def _apply_metadata_mask(mask: np.ndarray, rects: List[QRect]) -> np.ndarray:
    out = mask.copy()
    for rect in rects:
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(mask.shape[1], x + w)
        y1 = min(mask.shape[0], y + h)
        out[y0:y1, x0:x1] = False
    return out


def _load_model(model_path: Path) -> nn.Module:
    key = str(model_path.resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    if not model_path.exists():
        raise DLSpotDetectionError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location="cpu")
    except Exception as exc:
        raise DLSpotDetectionError(f"Failed to load model: {exc}") from exc

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise DLSpotDetectionError("Unsupported checkpoint format.")

    base_ch = _infer_base_channels(state_dict)
    model = UNetV2(in_ch=1, out_ch=1, base_ch=base_ch)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        raise DLSpotDetectionError(f"Model state dict mismatch: {exc}") from exc

    model.to(_DEVICE)
    _MODEL_CACHE[key] = model
    return model


def _infer_base_channels(state_dict: Dict[str, torch.Tensor]) -> int:
    for key in ("down1.block.0.weight", "down1.block.0.weight"):
        tensor = state_dict.get(key)
        if tensor is not None:
            return int(tensor.shape[0])
    # fallback: try common bases
    for ch in (64, 48, 32):
        if any(v.shape[0] == ch for v in state_dict.values() if v.dim() >= 4):
            return ch
    return 48


def _pad_to_multiple(image: np.ndarray, multiple: int = 16):
    h, w = image.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
    return padded, (pad_h, pad_w)


def _unpad(image: np.ndarray, pad_info) -> np.ndarray:
    pad_h, pad_w = pad_info
    if pad_h == 0 and pad_w == 0:
        return image
    h = image.shape[0] - pad_h if pad_h > 0 else image.shape[0]
    w = image.shape[1] - pad_w if pad_w > 0 else image.shape[1]
    return image[:h, :w]
