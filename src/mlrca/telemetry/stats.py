"""Small, dependency-free statistics used by tools and the heuristic baseline."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from statistics import mean, pstdev


def parse_ts(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    v = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(v)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def fmt_ts(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def psi(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    """Population Stability Index between two histograms (counts or proportions).

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    if len(expected) != len(actual):
        raise ValueError("histograms must have the same number of bins")
    se, sa = sum(expected) or 1.0, sum(actual) or 1.0
    total = 0.0
    for e, a in zip(expected, actual):
        pe, pa = max(e / se, eps), max(a / sa, eps)
        total += (pa - pe) * math.log(pa / pe)
    return round(total, 4)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(mean(values), 4),
        "std": round(pstdev(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def detect_mean_shift(points: list[tuple[datetime, float]], min_segment: int = 6) -> dict | None:
    """Single change point by maximizing the between-segment mean difference (scaled by pooled std).

    Returns ``None`` if the series is too short. ``score`` is roughly a t-like statistic;
    > 5 is a strong, sustained shift for hourly ML quality metrics.
    """
    n = len(points)
    if n < 2 * min_segment:
        return None
    vals = [v for _, v in points]
    best: dict | None = None
    for i in range(min_segment, n - min_segment + 1):
        left, right = vals[:i], vals[i:]
        ml, mr = mean(left), mean(right)
        pooled = math.sqrt((pstdev(left) ** 2 + pstdev(right) ** 2) / 2) or 1e-9
        score = abs(mr - ml) / pooled
        if best is None or score > best["score"]:
            best = {
                "index": i,
                "timestamp": fmt_ts(points[i][0]),
                "before_mean": round(ml, 4),
                "after_mean": round(mr, 4),
                "delta": round(mr - ml, 4),
                "score": round(score, 2),
            }
    return best
