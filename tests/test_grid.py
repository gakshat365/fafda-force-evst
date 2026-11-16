import numpy as np
import pandas as pd
from shapely.geometry import box

from ebv.environment import per_cell, to_cells
from ebv.region import GridSpec, build_grid


def test_grid_snaps_outwards_to_cell_edges():
    spec = GridSpec.around(box(76.03, 10.01, 76.27, 10.18), cell=0.1)
    assert spec.bounds == (76.0, 10.0, 76.3, 10.2)
    assert (spec.n_cols, spec.n_rows) == (3, 2)


def test_build_grid_keeps_majority_cells_and_labels_state():
    states = {"A": box(0, 0, 0.2, 0.1), "B": box(0.2, 0, 0.26, 0.1)}
    grid = build_grid(states)
    # third cell is only 60% covered by B -> kept; cells are labelled by their main state
    assert list(grid.cell_id) == ["0.00_0.00", "0.10_0.00", "0.20_0.00"]
    assert list(grid.state) == ["A", "A", "B"]


def test_per_cell_groups_pixels_by_cell():
    spec = GridSpec(0.0, 0.0, 0.2, 0.1, cell=0.1)            # one row, two columns
    pixels = np.array([[1, 1, 5, 5],
                       [1, 1, 5, 5]])                          # k = 2 pixels per cell side
    blocks = per_cell(pixels, spec, k=2)
    assert blocks.shape == (1, 2, 4)
    cells = pd.DataFrame({"lon0": [0.1, 0.0], "lat0": [0.0, 0.0]})
    assert list(to_cells(blocks.mean(axis=-1), spec, cells)) == [5, 1]
