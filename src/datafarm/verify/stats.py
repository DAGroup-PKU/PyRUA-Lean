"""Small, dependency-free statistics used by reviewer calibration and sim/real analysis.

Every function returns ``None`` instead of a number when the statistic is undefined
(e.g. zero variance), so callers cannot mistake "undefined" for "zero agreement".
"""

from __future__ import annotations

import math
from typing import Hashable, Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys):
        raise ValueError("pearson needs sequences of equal length")
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    return max(-1.0, min(1.0, sxy / math.sqrt(sxx * syy)))


def cohen_kappa(pairs: Sequence[tuple[Hashable, Hashable]]) -> float | None:
    """Cohen's kappa for two raters over the categories they used.

    ``None`` when there are no pairs or the expected agreement is 1 (both raters used a
    single identical category), where kappa is undefined.
    """

    n = len(pairs)
    if n == 0:
        return None
    categories = sorted({a for a, _ in pairs} | {b for _, b in pairs}, key=repr)
    observed = sum(1 for a, b in pairs if a == b) / n
    expected = 0.0
    for category in categories:
        first = sum(1 for a, _ in pairs if a == category) / n
        second = sum(1 for _, b in pairs if b == category) / n
        expected += first * second
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def wilson_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float] | None:
    """Wilson score interval for a binomial proportion (``None`` when ``total`` is 0)."""

    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("successes must lie in [0, total]")
    z = _normal_quantile(0.5 + confidence / 2)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _normal_quantile(probability: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation, |error| < 1.2e-9)."""

    if not 0 < probability < 1:
        raise ValueError("probability must be in (0, 1)")
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    low = 0.02425
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if probability > 1 - low:
        return -_normal_quantile(1 - probability)
    q = probability - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (
        ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1
    )


def quantile(sorted_values: Sequence[float], fraction: float) -> float | None:
    """Linear-interpolation quantile of already sorted values."""

    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

