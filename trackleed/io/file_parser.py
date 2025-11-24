# stal/io/file_parser.py

import re
from pathlib import Path
from typing import List

from trackleed.core.data_models import LeedFile

ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.png'}
# Allow arbitrary-length numeric prefixes/suffixes so generated frames (e.g., GIF extractions) are accepted.
START_DIGIT_PATTERN = re.compile(r'^(0*\d+)')
END_DIGIT_PATTERN = re.compile(r'(0*\d+)$')

# The pattern now only needs to match the data files.
DATA_FILE_PATTERN = re.compile(
    r"^([a-zA-Z]+)(\d+)([a-zA-Z])_"  # Element, Surface, Mode (e.g., Ru0001G)
    r"(\d+)_"                       # Time (e.g., 0195)
    r"T(\d+)_"                      # Temperature (e.g., T0700)
    r"E(\d+)_"                      # Energy (e.g., E0077)
    r"C(\d+)"                       # Coverage (e.g., C0000)
    r"\.tiff?$", re.IGNORECASE      # Extension .tif or .tiff
)

def parse_directory(directory: Path) -> List[LeedFile]:
    """
    Parses a directory for LEED data files based on a specific filename convention.

    This function now ignores background files, which are expected to be
    loaded manually by the user.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A list of LeedFile objects for the found data files, sorted by their ID.
    """
    data_files: List[LeedFile] = []

    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        filename = filepath.name

        # Try to match the data file pattern. Background files will be ignored.
        data_match = DATA_FILE_PATTERN.match(filename)
        if data_match:
            element, surface, mode, id_str, temp, energy, coverage = data_match.groups()
            data_files.append(
                LeedFile(
                    filepath=filepath,
                    filename=filename,
                    is_background=False,
                    element=element,
                    surface=int(surface),
                    mode=mode,
                    id=int(id_str),
                    temperature=float(temp),
                    energy=float(energy),
                    coverage=float(coverage) / 100.0
                )
            )
            continue

        # Fallback: accept numeric prefixes or suffixes (e.g., 1.tif, sample_001.png)
        stem = filepath.stem
        fallback_id = None
        start_match = START_DIGIT_PATTERN.match(stem)
        if start_match:
            fallback_id = int(start_match.group(1))
        else:
            end_match = END_DIGIT_PATTERN.search(stem)
            if end_match:
                fallback_id = int(end_match.group(1))

        if fallback_id is not None:
            data_files.append(
                LeedFile(
                    filepath=filepath,
                    filename=filename,
                    is_background=False,
                    element="GEN",
                    surface=0,
                    mode="U",
                    id=fallback_id,
                    temperature=0.0,
                    energy=0.0,
                    coverage=None,
                )
            )

    # Sort the data files by their ID to ensure a consistent order.
    data_files.sort(key=lambda f: f.id)

    return data_files

# # stal/io/file_parser.py

# import re
# from pathlib import Path
# from typing import List, Optional, Tuple

# from stal.core.data_models import LeedFile

# # We pre-compile the regular expressions for efficiency.
# # This pattern matches the data files, e.g., Ru0001G_0195_T0700_E0077_C0000.tiff
# DATA_FILE_PATTERN = re.compile(
#     r"^([a-zA-Z]+)(\d+)([a-zA-Z])_"  # Element, Surface, Mode (e.g., Ru0001G)
#     r"(\d+)_"                       # Time (e.g., 0195)
#     r"T(\d+)_"                      # Temperature (e.g., T0700)
#     r"E(\d+)_"                      # Energy (e.g., E0077)
#     r"C(\d+)"                       # Coverage (e.g., C0000)
#     r"\.tiff?$", re.IGNORECASE      # Extension .tif or .tiff, case-insensitive
# )

# # This pattern matches the background file, e.g., Ru0001RGB_E0077_T0700_BG.tiff
# BG_FILE_PATTERN = re.compile(
#     r"^([a-zA-Z]+)(\d+)([a-zA-Z]+)_" # Element, Surface, Mode (e.g., Ru0001RGB)
#     r"E(\d+)_"                      # Energy (e.g., E0077)
#     r"T(\d+)_"                      # Temperature (e.g., T0700)
#     r"BG"                           # Background identifier
#     r"\.tiff?$", re.IGNORECASE      # Extension .tif or .tiff, case-insensitive
# )


# def parse_directory(directory: Path) -> Tuple[List[LeedFile], Optional[LeedFile]]:
#     """
#     Parses a directory for LEED .tiff files based on specific filename conventions.

#     Args:
#         directory: The path to the directory to scan.

#     Returns:
#         A tuple containing:
#         - A list of LeedFile objects for the data files.
#         - A single LeedFile object for the background file, or None if not found.
#     """
#     data_files: List[LeedFile] = []
#     background_file: Optional[LeedFile] = None

#     for filepath in directory.iterdir():
#         if not (filepath.is_file() and filepath.suffix.lower() in ['.tif', '.tiff']):
#             continue

#         filename = filepath.name

#         # First, try to match the background file pattern
#         bg_match = BG_FILE_PATTERN.match(filename)
#         if bg_match:
#             element, surface, mode, energy, temp = bg_match.groups()
#             background_file = LeedFile(
#                 filepath=filepath,
#                 filename=filename,
#                 is_background=True,
#                 element=element,
#                 surface=int(surface),
#                 mode=mode,
#                 temperature=float(temp),
#                 energy=float(energy),
#                 id=None,        # No time info in BG file
#                 coverage=None,    # No coverage info in BG file
#             )
#             continue # Move to the next file

#         # If not a background file, try to match the data file pattern
#         data_match = DATA_FILE_PATTERN.match(filename)
#         if data_match:
#             element, surface, mode, id_str, temp, energy, coverage = data_match.groups()
#             data_files.append(
#                 LeedFile(
#                     filepath=filepath,
#                     filename=filename,
#                     is_background=False,
#                     element=element,
#                     surface=int(surface),
#                     mode=mode,
#                     id=int(id_str),
#                     temperature=float(temp),
#                     energy=float(energy),
#                     coverage=float(coverage) / 100.0 # Assuming C0150 means 1.50
#                 )
#             )

#     # Sort the data files, for example by time, to ensure a consistent order
#     data_files.sort(key=lambda f: f.id)

#     return data_files, background_file
