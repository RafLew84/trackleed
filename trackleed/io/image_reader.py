# stal/io/image_reader.py

from pathlib import Path
from functools import lru_cache
import numpy as np
import tifffile
from PIL import Image

CACHE_SIZE = 15
TIFF_SUFFIXES = {".tif", ".tiff"}
PIL_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}

@lru_cache(maxsize=CACHE_SIZE)
def load_tiff(filepath: Path) -> np.ndarray:
    """
    Loads an image file and returns its content as a NumPy array.

    Args:
        filepath: The path to the image file.

    Returns:
        A NumPy array containing the image data.
    """
    filepath = Path(filepath)
    print(f"CACHE MISS: Reading from disk: {filepath.name}")
    try:
        suffix = filepath.suffix.lower()
        if suffix in PIL_SUFFIXES:
            with Image.open(filepath) as im:
                return np.array(im.convert("L"))
        if suffix in TIFF_SUFFIXES or not suffix:
            return tifffile.imread(filepath)
        with Image.open(filepath) as im:
            return np.array(im.convert("L"))
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return np.array([])
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        try:
            with Image.open(filepath) as im:
                return np.array(im.convert("L"))
        except Exception as e2:
            print(f"Pillow fallback also failed for {filepath}: {e2}")
            return np.array([])
