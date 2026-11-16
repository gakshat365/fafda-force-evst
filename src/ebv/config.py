"""Project-wide settings. Every tunable number of the analysis lives here."""
from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "outputs" / "figures"
TABLES = ROOT / "outputs" / "tables"

# --- Study region and grid ---------------------------------------------------
STATES = ("Kerala", "Tamil Nadu", "Karnataka", "Goa", "Puducherry")
MAX_LONGITUDE = 81.0          # drops Yanam, the Puducherry exclave in Andhra Pradesh
CELL_DEG = 0.1                # grid cell size in degrees (~11 km)
MIN_REGION_FRACTION = 0.5     # keep cells with at least half their area in the region
ADMIN_URL = ("https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/"
             "gbOpen/IND/ADM1/geoBoundaries-IND-ADM1.geojson")

# --- GBIF occurrence query -----------------------------------------------------
GBIF_API = "https://api.gbif.org/v1"
AVES_CLASS_KEY = 212
YEARS = (2000, 2025)
REQUEST_INTERVAL_S = 0.35     # GBIF rate-limits heavy API use; stay polite and serial
THREATENED_OR_NT = ("NT", "VU", "EN", "CR")   # IUCN Red List codes counted as "threatened"

# --- Biodiversity targets ------------------------------------------------------
RAREFACTION_SIZE = 200        # richness = species expected in this many records (Hurlbert 1971)
TARGET_COVERAGE = 0.95        # coverage-based alternative (Chao & Jost 2012), tested for comparison
ABSENCE_MIN_RECORDS = 1000    # a cell can give an "inferred absence" only above this effort
MIN_PRESENCE_CELLS = 40       # focal species need at least this many presence cells

# Focal species for the occurrence EBV: IUCN Near Threatened or worse, at least
# MIN_PRESENCE_CELLS presence cells, and together covering every major habitat.
FOCAL_SPECIES = {  # GBIF scientific name: (display name, habitat)
    "Buceros bicornis": ("Great Hornbill", "Large evergreen forest"),
    "Ocyceros griseus": ("Malabar Grey Hornbill", "Western Ghats forest (endemic)"),
    "Anthracoceros coronatus": ("Malabar Pied Hornbill", "Riverine and dry forest"),
    "Trochalopteron fairbanki": ("Kerala Laughingthrush", "High-altitude shola (endemic)"),
    "Myiomela albiventris": ("White-bellied Blue Robin", "High-altitude shola (endemic)"),
    "Sterna aurantia": ("River Tern", "Rivers and reservoirs"),
    "Sterna acuticauda": ("Black-bellied Tern", "Rivers and reservoirs"),
    "Pelecanus philippensis": ("Spot-billed Pelican", "Lakes and irrigation tanks"),
    "Ciconia episcopus": ("Woolly-necked Stork", "Wetlands and farmland"),
    "Ammomanes phoenicura": ("Rufous-tailed Lark", "Dry open plains"),
    "Cursorius coromandelicus": ("Indian Courser", "Dry grassland and scrub"),
    "Gyps bengalensis": ("White-rumped Vulture", "Dry forest (Nilgiri-Moyar)"),
}

# --- Validation and models -------------------------------------------------------
BLOCK_DEG = 0.5               # spatial cross-validation block size (~55 km)
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_TUNING_ITER = 12
SEED = 42
