from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

from ..models.types import IntervalMetrics


_EFFORT_RE = re.compile(r"^(Z[1-7])_(\d+)m$")


def _parse_effort_key(effort_key: Optional[str]) -> Tuple[Optional[str], Optional[int]]:
    if not effort_key:
        return None, None
    m = _EFFORT_RE.match(effort_key)
    if not m:
        return None, None
    zone = m.group(1)
    minutes = int(m.group(2))
    return zone, minutes


def _zone_to_category(zone: Optional[str], minutes: Optional[int]) -> str:
    if not zone:
        return "uncategorized"
    if zone == "Z1":
        return "recovery"
    if zone == "Z2":
        return "z2"
    if zone == "Z3":
        return "tempo"
    if zone == "Z4":
        return "thr"
    if zone == "Z5":
        if minutes is not None and minutes <= 6:
            return "vo2-short"
        return "vo2-long"
    if zone == "Z6":
        return "anaerobic"
    if zone == "Z7":
        return "sprint"
    return "uncategorized"


def _primary_effort(metrics: Iterable[IntervalMetrics]) -> Optional[str]:
    work_by_effort: dict[str, float] = {}
    dur_by_effort: dict[str, float] = {}
    for m in metrics:
        ek = getattr(m, "effort_key", None)
        if not ek:
            continue
        work_by_effort[ek] = work_by_effort.get(ek, 0.0) + (m.work_kj or 0.0)
        dur_by_effort[ek] = dur_by_effort.get(ek, 0.0) + (m.interval.duration_s or 0.0)

    if not work_by_effort and not dur_by_effort:
        return None

    def _is_non_z1(k: str) -> bool:
        z, _ = _parse_effort_key(k)
        return z is not None and z != "Z1"

    candidates = [k for k in work_by_effort.keys() if _is_non_z1(k)] or list(work_by_effort.keys())
    if candidates:
        return max(candidates, key=lambda k: (work_by_effort.get(k, 0.0), dur_by_effort.get(k, 0.0)))

    candidates = [k for k in dur_by_effort.keys() if _is_non_z1(k)] or list(dur_by_effort.keys())
    if candidates:
        return max(candidates, key=lambda k: dur_by_effort.get(k, 0.0))
    return None


def categorize_workout(metrics: List[IntervalMetrics]) -> Tuple[Optional[str], Optional[str]]:
    """Return (category, ride_label) using dominant effort_key.

    Label pattern: "<category> <minutes>m" when minutes parsed; else category + raw key.
    """
    if not metrics:
        return None, None
    primary = _primary_effort(metrics)
    if not primary:
        return None, None
    zone, minutes = _parse_effort_key(primary)
    category = _zone_to_category(zone, minutes)
    label = f"{category} {minutes}m" if minutes else f"{category} {primary}"
    return category, label

