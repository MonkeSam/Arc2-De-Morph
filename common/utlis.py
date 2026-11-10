import re,argparse
from pathlib import Path

def add_common_args(parser,output_dir: str = ".", output_help: str = "Output directory"):
    """Common argument  in demorph.py and distance.py."""
    def two_digits(value: str) -> str:
        '''Algorithm code format check.'''
        if not re.fullmatch(r"\d{2}", value):
            raise argparse.ArgumentTypeError(
                f"'{value}' invalid value: only 2 digits are needed"
            )
        return value

    parser.add_argument("--morphs", type=str, required=True, help="Directory of the morphed images")
    parser.add_argument("--original", type=str, required=True, help="Directory of the original images")
    parser.add_argument("--output", type=str, required=True, default=output_dir, help=output_help)
    parser.add_argument(
        "--algorithm", type=two_digits, required=True,
        help="Select images generated with a specific algorithm, must be algorithm integer code for 'CXX'",
    )
    return parser

def algorithm_code(n: int) -> str:
    return f"C{n}"

def morph_pattern(algorithm: str) -> re.Pattern:
    """M_<criminal>_<accomplice>_C<XX>_<...>.png  (gruppi: stem, criminal, accomplice)."""
    return re.compile(
        rf"^(?P<stem>M_(?P<criminal>[^_]+)_(?P<accomplice>[^_]+)_{algorithm}_.+)\.png$"
    )


def subject_id(raw: str) -> str:
    """Get face id,splitting id from face pose."""
    return raw.split("-")[0]

def original_name(subject: str, pose: str) -> str:
    """Get face name in specific pose"""
    return f"{subject_id(subject)}-{pose}.jpg"

def inv_name(morph_stem: str, removed_stem: str) -> str:
    """Return demorph filename."""
    return f"INV_{morph_stem}_REMOVED_{removed_stem}.png"


INV_PATTERN = re.compile(
    r"^INV_(?P<stem>M_.+)_REMOVED_(?P<subject>[^_-]+)-(?P<pose>[^_]+)\.png$"
)

def stem_of(path) -> str:
    return Path(path).stem

def extracted_dir(base, algorithm: str, interp_mode: str) -> Path:
    """demorph.py output directory."""
    return Path(base) / algorithm / interp_mode
