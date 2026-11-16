"""Step 1: study region (Kerala, Tamil Nadu, Karnataka, Goa, Puducherry) and its 0.1 degree grid."""
from ebv import config
from ebv.region import build_grid, load_admin_boundaries, study_states

states = study_states(load_admin_boundaries())
grid = build_grid(states)
config.PROCESSED.mkdir(parents=True, exist_ok=True)
grid.to_csv(config.PROCESSED / "grid.csv", index=False)

print(f"{len(grid):,} cells of {config.CELL_DEG} degrees")
print(grid.state.value_counts().to_string())
