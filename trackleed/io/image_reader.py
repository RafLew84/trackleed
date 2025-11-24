# stal/io/image_reader.py

from pathlib import Path
from functools import lru_cache
import numpy as np
import tifffile
from PIL import Image

CACHE_SIZE = 15

@lru_cache(maxsize=CACHE_SIZE)
def load_tiff(filepath: Path) -> np.ndarray:
    """
    Loads a .tiff file and returns its content as a NumPy array.

    Args:
        filepath: The path to the .tiff file.

    Returns:
        A NumPy array containing the image data.
    """
    print(f"CACHE MISS: Reading from disk: {filepath.name}")
    try:
        return tifffile.imread(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        # Return an empty array or raise the exception, depending on desired error handling
        return np.array([])
    except Exception as e:
        print(f"An error occurred while reading {filepath}: {e}")
        # Fallback to Pillow for non-TIFF formats (e.g., PNG from GIF extraction)
        try:
            with Image.open(filepath) as im:
                return np.array(im.convert("L"))
        except Exception as e2:
            print(f"Pillow fallback also failed for {filepath}: {e2}")
            return np.array([])
