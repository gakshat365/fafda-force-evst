import numpy as np
import pytest

from ebv.diversity import (coverage_richness, estimated_coverage, expected_coverage,
                           rarefied_richness, sample_size_for_coverage)

RNG = np.random.default_rng(0)
COUNTS = RNG.geometric(0.02, size=150)     # a realistic long-tailed set of species counts


def test_rarefaction_hand_worked_example():
    # counts (2, 1, 1), n = 4, m = 2: 5/6 + 1/2 + 1/2
    assert rarefied_richness([2, 1, 1], 2) == pytest.approx(11 / 6)


def test_rarefaction_end_points():
    assert rarefied_richness(COUNTS, 1) == pytest.approx(1.0)
    assert rarefied_richness(COUNTS, COUNTS.sum()) == pytest.approx(len(COUNTS))


def test_rarefaction_mask_splits_total():
    mask = np.arange(len(COUNTS)) % 3 == 0
    m = 300
    total = rarefied_richness(COUNTS, m)
    assert rarefied_richness(COUNTS, m, mask) + rarefied_richness(COUNTS, m, ~mask) == pytest.approx(total)


def test_rarefaction_rejects_bad_sample_size():
    with pytest.raises(ValueError):
        rarefied_richness([3, 2], 6)


def test_coverage_hand_worked_example():
    # 1 - [2/4 * C(2,2)/C(3,2) + 2 * 1/4 * C(3,2)/C(3,2)] = 1 - (1/6 + 1/2)
    assert expected_coverage([2, 1, 1], 2) == pytest.approx(1 / 3)


def test_single_species_is_fully_covered():
    assert expected_coverage([50], 10) == pytest.approx(1.0)
    assert estimated_coverage([50]) == 1.0


def test_no_singletons_means_full_coverage():
    assert estimated_coverage([5, 3, 2, 2]) == 1.0


def test_coverage_increases_with_sample_size():
    values = [expected_coverage(COUNTS, m) for m in range(1, int(COUNTS.sum()), 97)]
    assert np.all(np.diff(values) >= -1e-12)


def test_sample_size_is_the_first_to_reach_target():
    target = 0.9
    m = sample_size_for_coverage(COUNTS, target)
    assert expected_coverage(COUNTS, m) >= target
    assert expected_coverage(COUNTS, m - 1) < target


def test_unreachable_coverage_gives_nan():
    sparse = [1] * 20 + [2]
    assert sample_size_for_coverage(sparse, 0.95) is None
    assert np.isnan(coverage_richness(sparse, 0.95))
