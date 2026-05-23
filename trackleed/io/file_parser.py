# stal/io/file_parser.py

import re
from pathlib import Path
from typing import Dict, List, Tuple

from trackleed.core.data_models import LeedFile

ALLOWED_EXTENSIONS = {'.tif', '.tiff', '.png'}
# Allow arbitrary-length numeric prefixes/suffixes so generated frames (e.g., GIF extractions) are accepted.
START_DIGIT_PATTERN = re.compile(r'^(0*\d+)')
END_DIGIT_PATTERN = re.compile(r'(0*\d+)$')
FRAME_TOKEN_PATTERN = re.compile(r'_(0*\d+)(?:_|$)')
METADATA_ID_PATTERNS = (
    ("temperature", re.compile(r'(?:^|_)T(0*\d+)(?:_|$)', re.IGNORECASE)),
    ("coverage", re.compile(r'(?:^|_)C(0*\d+)(?:_|$)', re.IGNORECASE)),
    ("energy", re.compile(r'(?:^|_)E(0*\d+)(?:_|$)', re.IGNORECASE)),
)
ID_REPAIR_SOURCE_ORDER = ("frame", "temperature", "coverage", "energy")

# The pattern now only needs to match the data files.
DATA_FILE_PATTERN = re.compile(
    r"^([a-zA-Z][a-zA-Z0-9]*?)"     # Element (e.g., Ru, Cu, Cu3Au), non-greedy before surface
    r"(?:\((\d+)\)|(\d+))"          # Surface: digits in parentheses or plain digits
    r"(?:[-_]\d+)?"                 # Optional surface/domain suffix, e.g. Ag111-3G
    r"([a-zA-Z])_"                  # Mode (e.g., G)
    r"(\d+)_"                       # Time (e.g., 0195)
    r"T(\d+)_"                      # Temperature (e.g., T0700)
    r"E(\d+)_"                      # Energy (e.g., E0077)
    r"C(\d+)"                       # Coverage (e.g., C0000)
    r"(?:_[A-Za-z][A-Za-z0-9+-]*)*" # Optional suffix tags, e.g., _R-33
    r"\.tiff?$", re.IGNORECASE      # Extension .tif or .tiff
)


def _parse_coverage_code(raw_code: str) -> float:
    """
    Parse coverage token from filename.

    Current datasets use 4-digit fixed-point encoding:
    - C1000 -> 1.000
    - C0920 -> 0.920
    so the divisor is 1000.

    Backward compatibility:
    - For 3-digit codes use divisor 100.
    - For shorter codes, return plain numeric value.
    """
    code = (raw_code or "").strip()
    if not code:
        return 0.0
    value = float(code)
    n_digits = len(code)
    if n_digits >= 4:
        return value / 1000.0
    if n_digits == 3:
        return value / 100.0
    return value


def _metadata_id_candidates(stem: str) -> Dict[str, int]:
    candidates: Dict[str, int] = {}
    for name, pattern in METADATA_ID_PATTERNS:
        match = pattern.search(stem)
        if match:
            candidates[name] = int(match.group(1))
    return candidates


def _filename_id_candidates(stem: str) -> Dict[str, int]:
    candidates: Dict[str, int] = {}
    frame_token_match = FRAME_TOKEN_PATTERN.search(stem)
    if frame_token_match:
        candidates["frame"] = int(frame_token_match.group(1))
    candidates.update(_metadata_id_candidates(stem))
    return candidates


def _first_metadata_id_candidate(candidates: Dict[str, int]) -> int | None:
    for source_name in ("temperature", "coverage", "energy"):
        value = candidates.get(source_name)
        if value is not None:
            return value
    return None


def _ids_are_unique(files: List[LeedFile]) -> bool:
    ids = [file.id for file in files if file.id is not None]
    return len(ids) == len(files) and len(set(ids)) == len(ids)


def _repair_duplicate_ids_from_metadata_tokens(
    parsed_entries: List[Tuple[LeedFile, Dict[str, int]]]
) -> str | None:
    files = [file for file, _candidates in parsed_entries]
    if _ids_are_unique(files):
        return None

    for source_name in ID_REPAIR_SOURCE_ORDER:
        values = []
        for _file, candidates in parsed_entries:
            value = candidates.get(source_name)
            if value is None:
                values = []
                break
            values.append(value)

        if values and len(set(values)) == len(values):
            for file, candidates in parsed_entries:
                file.id = candidates[source_name]
            return source_name

    return None


def repair_duplicate_frame_ids(files: List[LeedFile]) -> str | None:
    """
    Repair duplicate in-memory frame IDs using filename metadata tokens.

    Older parser versions could assign the same ID to many frames, especially when
    the trailing C token was constant. Detection linking needs unique IDs, so this
    function tries the explicit frame token first, then the deterministic metadata
    fallback order: T token, then C token, then E token.
    """
    parsed_entries = []
    for file in files:
        if file.id is None:
            continue
        parsed_entries.append((file, _filename_id_candidates(Path(file.filename).stem)))

    return _repair_duplicate_ids_from_metadata_tokens(parsed_entries)


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
    parsed_entries: List[Tuple[LeedFile, Dict[str, int]]] = []

    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue
        if filepath.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        filename = filepath.name

        # Try to match the data file pattern. Background files will be ignored.
        data_match = DATA_FILE_PATTERN.match(filename)
        if data_match:
            element, surface_paren, surface_plain, mode, id_str, temp, energy, coverage = data_match.groups()
            surface = surface_paren or surface_plain
            leed_file = LeedFile(
                filepath=filepath,
                filename=filename,
                is_background=False,
                element=element,
                surface=int(surface),
                mode=mode,
                id=int(id_str),
                temperature=float(temp),
                energy=float(energy) / 10.0,
                coverage=_parse_coverage_code(coverage),
            )
            data_files.append(leed_file)
            parsed_entries.append(
                (
                    leed_file,
                    {
                        "frame": int(id_str),
                        "temperature": int(temp),
                        "coverage": int(coverage),
                        "energy": int(energy),
                    },
                )
            )
            continue

        # Fallback: accept numeric prefixes or suffixes (e.g., 1.tif, sample_001.png)
        stem = filepath.stem
        metadata_candidates = _filename_id_candidates(stem)
        fallback_id = None
        start_match = START_DIGIT_PATTERN.match(stem)
        if start_match:
            fallback_id = int(start_match.group(1))
        else:
            frame_token_match = FRAME_TOKEN_PATTERN.search(stem)
            if frame_token_match:
                fallback_id = int(frame_token_match.group(1))
            else:
                metadata_id = _first_metadata_id_candidate(metadata_candidates)
                if metadata_id is not None:
                    fallback_id = metadata_id
                else:
                    end_match = END_DIGIT_PATTERN.search(stem)
                    if end_match:
                        fallback_id = int(end_match.group(1))

        if fallback_id is not None:
            leed_file = LeedFile(
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
            data_files.append(leed_file)
            parsed_entries.append((leed_file, metadata_candidates))

    # Sort the data files by their ID to ensure a consistent order.
    _repair_duplicate_ids_from_metadata_tokens(parsed_entries)
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
