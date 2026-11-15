"""Shared settings for the Kerala bird EBV project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
RES = ROOT / "outputs" / "results"
for p in (RAW, PROC, FIG, RES):
    p.mkdir(parents=True, exist_ok=True)

# Study region and grid
KERALA_GEOJSON = RAW / "kerala.geojson"
CELL = 0.05            # grid size in degrees (~5.5 km)
MIN_LAND_FRAC = 0.5    # keep cells with >= 50% area inside Kerala

# GBIF query (Aves, human observations, 2000-2025)
GBIF_API = "https://api.gbif.org/v1"
AVES_CLASS_KEY = 212
YEARS = "2000,2025"

# Cells with fewer records than this are treated as under-sampled
MIN_RECORDS = 200

SEED = 42
