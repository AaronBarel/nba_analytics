"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
POSITION_GROUPS = {"G": ["PG", "SG"], "F": ["SF", "PF"], "C": ["C"]}


def season_label(end_year: int) -> str:
    """2026 -> '2025-26' (Basketball-Reference names seasons by their end year)."""
    return f"{end_year - 1}-{str(end_year)[-2:]}"
