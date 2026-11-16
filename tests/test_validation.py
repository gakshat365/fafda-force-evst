import numpy as np
import pandas as pd

from ebv.features import select_uncorrelated
from ebv.validation import outer_splits, spatial_blocks, state_groups, true_skill_statistic

RNG = np.random.default_rng(1)
N = 600
CELLS = pd.DataFrame({
    "lon": RNG.uniform(74, 80, N), "lat": RNG.uniform(8, 18, N),
    "state": RNG.choice(["Kerala", "Tamil Nadu", "Karnataka", "Goa", "Puducherry"], N),
})


def test_spatial_folds_never_share_a_block():
    blocks = spatial_blocks(CELLS)
    for train, test in outer_splits(CELLS, "spatial"):
        assert not set(blocks[train]) & set(blocks[test])


def test_every_cell_is_tested_exactly_once():
    for scheme in ("random", "spatial", "state"):
        tested = np.concatenate([test for _, test in outer_splits(CELLS, scheme)])
        assert sorted(tested) == list(range(N))


def test_state_scheme_merges_small_states():
    groups = set(state_groups(CELLS))
    assert groups == {"Kerala", "Tamil Nadu", "Karnataka"}
    assert len(outer_splits(CELLS, "state")) == 3


def test_stratified_spatial_folds_keep_both_classes():
    y = (RNG.uniform(size=N) < 0.15).astype(int)
    for _, test in outer_splits(CELLS, "spatial", y, classification=True):
        assert 0 < y[test].sum() < len(test)


def test_collinear_candidate_is_dropped_in_favour_of_earlier_one():
    a = RNG.normal(size=200)
    frame = pd.DataFrame({"a": a, "b": a * 2 + RNG.normal(scale=0.01, size=200), "c": RNG.normal(size=200)})
    kept, dropped = select_uncorrelated(frame, ["a", "b", "c"])
    assert kept == ["a", "c"]
    assert dropped.iloc[0]["dropped"] == "b" and dropped.iloc[0]["kept_instead"] == "a"


def test_tss_is_one_for_perfect_and_zero_for_constant_scores():
    y = np.array([0, 0, 1, 1])
    assert true_skill_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert true_skill_statistic(y, np.full(4, 0.5)) == 0.0
