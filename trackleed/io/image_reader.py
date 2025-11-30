# stal/io/image_reader.py

from pathlib import Path
from functools import lru_cache
import numpy as np
import tifffile
from PIL import Image

CACHE_SIZE = 15
VERBOSE = False

@lru_cache(maxsize=CACHE_SIZE)
def load_tiff(filepath: Path) -> np.ndarray:
    """
    Loads a .tiff file and returns its content as a NumPy array.

    Args:
        filepath: The path to the .tiff file.

    Returns:
        A NumPy array containing the image data.
    """
    if VERBOSE:
        print(f"CACHE MISS: Reading from disk: {filepath.name}")
    try:
        return tifffile.imread(filepath)
    except FileNotFoundError:
        if VERBOSE:
            print(f"Error: File not found at {filepath}")
        return np.array([])
    except Exception as e:
        if VERBOSE:
            print(f"An error occurred while reading {filepath}: {e}")
        # Fallback to Pillow for non-TIFF formats (e.g., PNG from GIF extraction)
        try:
            with Image.open(filepath) as im:
                return np.array(im.convert("L"))
        except Exception as e2:
            if VERBOSE:
                print(f"Pillow fallback also failed for {filepath}: {e2}")
            return np.array([])
