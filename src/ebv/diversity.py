"""Richness estimators that put grid cells with very different sampling effort on an equal footing.

Every function takes `counts`: the number of GBIF records of each species seen in one
cell (species with zero records are left out). Records are treated as individuals.

- Hurlbert (1971) rarefaction: expected species in a random sub-sample of `m` records.
- Chao & Jost (2012) coverage: the share of the cell's bird "community" (by records)
  that belongs to species already seen. Comparing cells at the same coverage, rather
  than the same number of records, is fairer when cells differ in diversity.
"""
import numpy as np
from scipy.special import gammaln


def _log_choose(a, b):
    """log of the binomial coefficient C(a, b), element-wise; -inf where b > a (C = 0)."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    out = np.full(a.shape, -np.inf)
    ok = b <= a
    out[ok] = gammaln(a[ok] + 1) - gammaln(b[ok] + 1) - gammaln(a[ok] - b[ok] + 1)
    return out


def rarefied_richness(counts, m: int, mask=None) -> float:
    """Expected number of species in a random sample of `m` of the `n` records (m <= n).

    With a boolean `mask`, only the species where mask is True are counted (e.g. the
    expected number of threatened species in the sample).
    """
    counts = np.asarray(counts, float)
    n = counts.sum()
    if not 1 <= m <= n:
        raise ValueError(f"sample size m={m} must be between 1 and n={n:.0f}")
    p_absent = np.exp(_log_choose(n - counts, m) - _log_choose(n, m))
    p_present = 1.0 - p_absent
    if mask is not None:
        p_present = p_present[np.asarray(mask, bool)]
    return float(p_present.sum())


def estimated_coverage(counts) -> float:
    """Chao & Jost (2012) estimate of sample coverage for the full sample of n records."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    f1, f2 = np.sum(counts == 1), np.sum(counts == 2)
    if f1 == 0:
        return 1.0
    if f2 > 0:
        correction = (n - 1) * f1 / ((n - 1) * f1 + 2 * f2)
    else:
        correction = (n - 1) * (f1 - 1) / ((n - 1) * (f1 - 1) + 2)
    return float(1 - f1 / n * correction)


def expected_coverage(counts, m: int) -> float:
    """Expected sample coverage of a random sample of m records (Chao & Jost 2012, eq. 4a)."""
    counts = np.asarray(counts, float)
    n = counts.sum()
    if m >= n:
        return estimated_coverage(counts)
    ratio = np.exp(_log_choose(n - counts, m) - _log_choose(n - 1, m))
    return float(1 - np.sum(counts / n * ratio))


def sample_size_for_coverage(counts, target: float) -> int | None:
    """Smallest number of records whose expected coverage reaches `target`.

    Returns None when even the full sample falls short of the target: estimating richness
    there would need extrapolation, so such cells are treated as under-sampled.
    """
    counts = np.asarray(counts, float)
    n = int(counts.sum())
    if n == 0 or estimated_coverage(counts) < target:
        return None
    low, high = 1, n          # coverage rises with m, so binary search the first m >= target
    while low < high:
        mid = (low + high) // 2
        if expected_coverage(counts, mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def coverage_richness(counts, target: float, mask=None) -> float:
    """Expected species (optionally only `mask` species) at the given sample coverage."""
    m = sample_size_for_coverage(counts, target)
    return np.nan if m is None else rarefied_richness(counts, m, mask)
