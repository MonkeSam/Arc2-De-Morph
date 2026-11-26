POSES = ("11", "12", "13")
THRESHOLD = 0.5143
LABELS = ("A_vs_C", "M_vs_C", "M_vs_A", "DC_vs_A", "DC_vs_C", "DA_vs_C", "DA_vs_A")

def column(label: str, kind: str = "distance") -> str:
    """Column label header."""
    return f"{label}_{kind}"
FIELDS = ["morph", "pose", "criminal", "accomplice"] + [
    column(label, kind) for label in LABELS for kind in ("distance", "verified")
]

DEFAULT_DEMORPH_PATH = "./invslerp"

# Single root directory for every pipeline output, and its subfolders
DEFAULT_OUTPUT_ROOT = "Arc2DeMorph"
DEMORPH_SUBDIR = "invslerp"
STATS_SUBDIR = "distance_stats"
GATE_LOG_NAME = "gate_results.log"
