"""Declared static backgrounds; randomness never participates in scene geometry."""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
import tifffile

from trackleed.synthetic.storage import pixel_descriptor

from trackleed.synthetic.apparatus import Apparatus, ApparatusImage, prepare_apparatus


@dataclass(frozen=True)
class Glow:
    center_px: tuple[float, float]
    sigma_px: tuple[float, float]
    amplitude: float


@dataclass(frozen=True)
class ProceduralBackground:
    gradient: tuple[float, float] = (0.0, 0.0)
    glow: Glow | None = None
    texture_amplitude: float = 0.0
    texture_seed: int = 0
    kind: str = "procedural"


@dataclass(frozen=True)
class ImportTransform:
    crop_xywh: tuple[int, int, int, int] | None = None
    scale: float = 1.0
    offset_px: tuple[float, float] = (0.0, 0.0)
    fill: float = 0.0


@dataclass(frozen=True)
class ImportedBackground:
    path: str
    gain: float
    offset: float = 0.0
    screen_mask: str | None = None
    occluder_mask: str | None = None
    transform: ImportTransform | None = None
    kind: str = "import"


BackgroundModel = ProceduralBackground | ImportedBackground


@dataclass(frozen=True)
class BackgroundSource:
    role: str
    content: bytes
    pixels: NDArray
    format: str

    @property
    def package_path(self) -> str:
        return f"inputs/{self.role}.source"

    def descriptor(self) -> dict[str, Any]:
        return {"package_path": self.package_path, "sha256": sha256(self.content).hexdigest(),
                **pixel_descriptor(self.pixels), "format": self.format,
                "range": [float(self.pixels.min()), float(self.pixels.max())]}


@dataclass(frozen=True)
class PreparedBackground:
    apparatus: ApparatusImage
    sources: tuple[BackgroundSource, ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        values = self.apparatus.background
        if not np.isfinite(values).all() or (np.abs(values) > np.finfo(np.float32).max).any():
            raise ValueError("Prepared background must fit finite float32")


def parse_background(value: Any) -> BackgroundModel:
    if isinstance(value, dict) and value.get("kind") == "import":
        return _parse_import(value)
    values = background_object(value, {"kind", "gradient", "glow", "texture_amplitude", "texture_seed"})
    if values.get("kind") != "procedural":
        raise ValueError("background_model.kind must be procedural or import")
    gradient = background_pair(values.get("gradient", [0, 0]), "gradient")
    glow = None
    if values.get("glow") is not None:
        raw = background_object(values["glow"], {"center_px", "sigma_px", "amplitude"})
        center = background_pair(raw.get("center_px"), "glow.center_px")
        sigma = background_pair(raw.get("sigma_px"), "glow.sigma_px")
        amplitude = background_number(raw.get("amplitude"), "glow.amplitude")
        if min(sigma) <= 0 or amplitude < 0:
            raise ValueError("background glow requires positive sigma and nonnegative amplitude")
        glow = Glow(center, sigma, amplitude)
    texture = background_number(values.get("texture_amplitude", 0), "texture_amplitude")
    seed = values.get("texture_seed", 0)
    if texture < 0 or type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("background texture needs nonnegative amplitude and a uint64 seed")
    return ProceduralBackground(gradient, glow, texture, seed)


def background_object(value: Any, fields: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - fields:
        raise ValueError(f"background object permits only {sorted(fields)}")
    return dict(value)


def background_number(value: Any, label: str) -> float:
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"background.{label} must be finite")
    return float(value)


def background_pair(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"background.{label} must contain two coordinates")
    return background_number(value[0], label), background_number(value[1], label)


def prepare_background(
    model: BackgroundModel | None, apparatus: Apparatus | None,
    image_size: tuple[int, int], level: float, base_dir: Path,
) -> PreparedBackground:
    result = prepare_apparatus(apparatus, image_size, level)
    if isinstance(model, ImportedBackground):
        return _prepare_import(model, result, image_size, base_dir)
    metadata: dict[str, Any] = {"kind": "constant" if model is None else "procedural",
                                "additional_noise": "none", "sources": {}}
    if model is None:
        return PreparedBackground(result, (), metadata)
    width, height = image_size
    y, x = np.ogrid[:height, :width]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        field = np.full((height, width), level, dtype=np.float64)
        field += model.gradient[0] * (x / max(width - 1, 1) - .5)
        field += model.gradient[1] * (y / max(height - 1, 1) - .5)
        if model.glow is not None:
            glow = model.glow
            distance = ((x - glow.center_px[0]) / glow.sigma_px[0])**2
            distance = distance + ((y - glow.center_px[1]) / glow.sigma_px[1])**2
            field += glow.amplitude * np.exp(-.5 * distance)
        if model.texture_amplitude:
            rng = np.random.Generator(np.random.PCG64(model.texture_seed))
            field += rng.uniform(-1, 1, size=field.shape) * model.texture_amplitude
    # Generated apparatus appearance is separate from the useful-screen field.
    result.background[result.signal_region] = field[result.signal_region]
    if not np.isfinite(result.background).all():
        raise ValueError("Prepared background must be finite")
    metadata["texture"] = {"stream": "background_texture", "algorithm": "PCG64",
                           "distribution": "uniform[-amplitude,amplitude)",
                           "seed": model.texture_seed, "static": True}
    return PreparedBackground(result, (), metadata)


def _parse_import(value: Any) -> ImportedBackground:
    values = background_object(value, {"kind", "path", "gain", "offset", "screen_mask", "occluder_mask", "transform"})
    for key in ("path", "screen_mask", "occluder_mask"):
        path = values.get(key)
        if path is None and key != "path":
            continue
        if not isinstance(path, str) or not path.strip() or "\0" in path:
            raise ValueError(f"background.{key} must be a nonempty path")
    gain = background_number(values.get("gain"), "gain")
    offset = background_number(values.get("offset", 0), "offset")
    if gain <= 0:
        raise ValueError("background.gain must be positive")
    return ImportedBackground(values["path"], gain, offset,
                              values.get("screen_mask"), values.get("occluder_mask"),
                              _parse_transform(values["transform"]) if values.get("transform") is not None else None)


def _read_source(path: Path, role: str) -> BackgroundSource:
    content = path.read_bytes()
    stream = BytesIO(content)
    if content[:2] in (b"II", b"MM"):
        with tifffile.TiffFile(stream) as image:
            if len(image.pages) != 1:
                raise ValueError(f"Background input must contain one image: {path}")
            page = image.pages[0]
            if not isinstance(page, tifffile.TiffPage):
                raise ValueError(f"TIFF requires a standalone image page: {path}")
            gray = page.photometric == tifffile.PHOTOMETRIC.MINISBLACK and page.samplesperpixel == 1
            rgb = (page.photometric == tifffile.PHOTOMETRIC.RGB and page.samplesperpixel == 3
                   and page.planarconfig == tifffile.PLANARCONFIG.CONTIG)
            if not (gray or rgb) or page.imagedepth != 1:
                raise ValueError(f"TIFF requires MINISBLACK grayscale or contiguous RGB samples: {path}")
            pixels = image.asarray()
            if (gray and pixels.ndim != 2) or (rgb and (pixels.ndim != 3 or pixels.shape[-1] != 3)):
                raise ValueError(f"TIFF sample shape disagrees with grayscale/RGB encoding: {path}")
        format_name = "TIFF"
    else:
        with Image.open(stream) as image:
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError(f"Background input must contain one image: {path}")
            if image.format != "PNG" or image.mode not in ("L", "RGB", "I", "I;16"):
                raise ValueError(f"Background input must be grayscale/RGB PNG or TIFF: {path}")
            pixels = np.array(image)
        format_name = "PNG"
    if (pixels.ndim not in (2, 3) or (pixels.ndim == 3 and pixels.shape[2] != 3)
            or pixels.dtype.kind not in "uif" or not pixels.size or not np.isfinite(pixels).all()):
        raise ValueError(f"Background input must be finite numeric grayscale or RGB: {path}")
    return BackgroundSource(role, content, pixels, format_name)


def _prepare_import(
    model: ImportedBackground, apparatus: ApparatusImage,
    image_size: tuple[int, int], base_dir: Path,
) -> PreparedBackground:
    source = _read_source(base_dir / model.path, "background")
    pixels = source.pixels.astype(np.float64)
    if pixels.ndim == 3:
        pixels = .299 * pixels[:, :, 0] + .587 * pixels[:, :, 1] + .114 * pixels[:, :, 2]
    transform = _resolved_transform(model.transform, pixels.shape, image_size)
    with np.errstate(over="ignore", invalid="ignore"):
        background = pixels * model.gain + model.offset
    if not np.isfinite(background).all():
        raise ValueError("Prepared background must be finite")
    background, covered = _warp(background, transform, image_size)
    sources = [source]
    screen, occluders = apparatus.screen & covered, apparatus.occluders.copy()
    for role in ("screen_mask", "occluder_mask"):
        path = getattr(model, role)
        if path is None:
            continue
        mask = _read_source(base_dir / path, role)
        if (mask.pixels.shape != pixels.shape or mask.pixels.dtype != np.uint8
                or not np.isin(mask.pixels, [0, 255]).all()):
            raise ValueError(f"{role} must be a matching binary uint8 0/255 image")
        sources.append(mask)
        warped, _ = _warp(mask.pixels, transform, image_size, mask=True)
        if role == "screen_mask":
            screen &= warped != 0
        else:
            occluders |= warped != 0
    metadata = {"kind": "import", "rgb_to_gray": [.299, .587, .114],
                "gain": model.gain, "offset": model.offset, "additional_noise": "none",
                "transform": asdict(transform),
                "interpolation": {"background": "bilinear_edge_clamp", "masks": "nearest"},
                "pixel_mapping": "output+0.5 = offset + scale*(source-crop_origin+0.5)",
                "sources": {s.role: s.descriptor() for s in sources},
                "source_paths": "input.json:condition.background_model",
                "geometry_coordinates": "scene and procedural apparatus use output pixel centers",
                "mask_composition": "screen intersection; occluder union; background unchanged"}
    return PreparedBackground(ApparatusImage(screen, occluders, background), tuple(sources), metadata)


def _parse_transform(value: Any) -> ImportTransform:
    values = background_object(value, {"crop_xywh", "scale", "offset_px", "fill"})
    crop = values.get("crop_xywh")
    if crop is not None:
        if (not isinstance(crop, (list, tuple)) or len(crop) != 4
                or any(type(v) is not int or v < 0 for v in crop) or min(crop[2:]) == 0):
            raise ValueError("background crop_xywh needs integer x/y>=0, width/height>0")
        crop = tuple(crop)
    scale = background_number(values.get("scale", 1), "transform.scale")
    if scale <= 0:
        raise ValueError("background transform.scale must be positive and uniform")
    offset = background_pair(values.get("offset_px", [0, 0]), "transform.offset_px")
    fill = background_number(values.get("fill", 0), "transform.fill")
    return ImportTransform(crop, scale, offset, fill)


def _resolved_transform(
    transform: ImportTransform | None, shape: tuple[int, ...], image_size: tuple[int, int],
) -> ImportTransform:
    height, width = shape
    if transform is None:
        if (width, height) != image_size:
            raise ValueError("Background dimensions require an explicit transform")
        transform = ImportTransform()
    crop = transform.crop_xywh or (0, 0, width, height)
    x, y, w, h = crop
    if x + w > width or y + h > height:
        raise ValueError("background crop_xywh exceeds source dimensions")
    return replace(transform, crop_xywh=crop)


def _warp(
    values: NDArray, transform: ImportTransform, image_size: tuple[int, int], *, mask: bool = False,
) -> tuple[NDArray, NDArray[np.bool_]]:
    assert transform.crop_xywh is not None
    x, y, width, height = transform.crop_xywh
    values = values[y:y + height, x:x + width]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        u = (np.arange(image_size[0], dtype=np.float64) + .5 - transform.offset_px[0]) / transform.scale - .5
        v = (np.arange(image_size[1], dtype=np.float64) + .5 - transform.offset_px[1]) / transform.scale - .5
    if not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("background transform produces non-finite coordinates")
    coverage = ((v[:, None] >= -.5) & (v[:, None] < height - .5)
                & (u >= -.5) & (u < width - .5))
    u, v = np.clip(u, 0, width - 1), np.clip(v, 0, height - 1)
    if mask:
        result = values[np.ix_(np.floor(v + .5).astype(int), np.floor(u + .5).astype(int))]
        return np.where(coverage, result, 0), coverage
    left, top = np.floor(u).astype(int), np.floor(v).astype(int)
    right, bottom = np.minimum(left + 1, width - 1), np.minimum(top + 1, height - 1)
    fx, fy = u - left, (v - top)[:, None]
    upper = (1 - fx) * values[np.ix_(top, left)] + fx * values[np.ix_(top, right)]
    lower = (1 - fx) * values[np.ix_(bottom, left)] + fx * values[np.ix_(bottom, right)]
    result = (1 - fy) * upper + fy * lower
    return np.where(coverage, result, transform.fill), coverage
