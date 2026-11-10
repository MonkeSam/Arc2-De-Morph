POSES = ("11", "12", "13")
THRESHOLD = 0.5143
LABELS = ("A_vs_C", "M_vs_C", "M_vs_A", "DC_vs_A", "DC_vs_C", "DA_vs_C", "DA_vs_A")

def column(label: str, kind: str = "distance") -> str:
    """Column label header."""
    return f"{label}_{kind}"
FIELDS = ["morph", "pose", "criminal", "accomplice"] + [
    column(label, kind) for label in LABELS for kind in ("distance", "verified")
]
