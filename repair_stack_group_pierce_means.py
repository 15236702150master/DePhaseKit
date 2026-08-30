#!/usr/bin/env python3
"""
Repair stack group pierce mean coordinates using the correct phase cache.

For qualifying groups:
  - t5+t9 -> recompute average from sP member pierce cache
  - t6+t8 -> recompute average from pP member pierce cache

Writes:
  - sidecar geometry.pierce_lon_mean / pierce_lat_mean
  - result-package meta.json pierce_lon_mean / pierce_lat_mean
  - result-package meta.json preview_pierce_phase
  - TSV audit report of mismatches/fixes
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

# Make the DePhaseKit package importable so PROJECT_ROOT resolves relative to the
# script location instead of being hardcoded to an absolute path.
_DPK_DIR = os.path.dirname(os.path.abspath(__file__))
if _DPK_DIR not in sys.path:
    sys.path.insert(0, _DPK_DIR)

from pierce_point_cache import PROJECT_ROOT  # noqa: E402

STACK_ANALYSIS_ROOT = PROJECT_ROOT / "data" / "output" / "stack" / "analysis"
PIERCE_CACHE_ROOT = PROJECT_ROOT / "data" / "output" / "pierce_points"
REPORT_PATH = PROJECT_ROOT / "data" / "output" / "stack" / "analysis" / "stack_group_pierce_mean_audit.tsv"
DATASETS = ("pick_other", "pick_jandy")
TOL = 1e-6


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number in (-12345.0, 12345.0):
        return None
    return number


def _pair_phase(markers: dict) -> tuple[str, str] | None:
    t5 = _finite(markers.get("t5"))
    t6 = _finite(markers.get("t6"))
    t8 = _finite(markers.get("t8"))
    t9 = _finite(markers.get("t9"))
    if t5 is not None and t9 is not None:
        return ("t5+t9", "sP")
    if t6 is not None and t8 is not None:
        return ("t6+t8", "pP")
    return None


def _members_file(payload: dict) -> Path | None:
    result_package_dir = str(payload.get("result_package_dir") or "").strip()
    if not result_package_dir:
        return None
    path = Path(result_package_dir).expanduser().resolve() / "members.txt"
    return path if path.exists() else None


def _used_wave_names(members_path: Path) -> list[str]:
    names: list[str] = []
    with members_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("status") or "").strip().lower() != "used":
                continue
            name = str(row.get("wave_name") or "").strip()
            if name:
                names.append(name)
    return names


def _load_pierce_points(path: Path) -> dict[str, tuple[float, float]]:
    records: dict[str, tuple[float, float]] = {}
    if not path.exists():
        return records
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        lon = _finite(parts[1])
        lat = _finite(parts[2])
        if lon is None or lat is None:
            continue
        records[parts[0]] = (lon, lat)
    return records


def _phase_cache_file(dataset: str, event: str, phase: str) -> Path:
    return PIERCE_CACHE_ROOT / dataset / event / f"pierce_points_{phase}_35.0km_iasp91.txt"


def _same(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= TOL


def main() -> int:
    phase_cache: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = {}
    rows: list[dict[str, str]] = []
    scanned = 0
    fixed = 0
    mismatched = 0

    for dataset in DATASETS:
        dataset_dir = STACK_ANALYSIS_ROOT / dataset
        if not dataset_dir.is_dir():
            continue
        for event_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for sidecar_path in sorted(event_dir.glob("stack_group*.stack.json")):
                sidecar = _load_json(sidecar_path)
                rule = _pair_phase(sidecar.get("markers", {}) or {})
                if rule is None:
                    continue
                scanned += 1
                pair_kind, phase_kind = rule
                members_path = _members_file(sidecar)
                if members_path is None:
                    continue
                wave_names = _used_wave_names(members_path)
                if not wave_names:
                    continue
                cache_key = (dataset, event_dir.name, phase_kind)
                if cache_key not in phase_cache:
                    phase_cache[cache_key] = _load_pierce_points(_phase_cache_file(dataset, event_dir.name, phase_kind))
                records = phase_cache[cache_key]
                lons = []
                lats = []
                for wave_name in wave_names:
                    record = records.get(wave_name)
                    if record is None:
                        continue
                    lons.append(record[0])
                    lats.append(record[1])
                if not lons or not lats:
                    continue
                correct_lon = sum(lons) / len(lons)
                correct_lat = sum(lats) / len(lats)

                geometry = sidecar.setdefault("geometry", {})
                old_lon = _finite(geometry.get("pierce_lon_mean"))
                old_lat = _finite(geometry.get("pierce_lat_mean"))
                result_package_dir = Path(str(sidecar.get("result_package_dir") or "")).expanduser().resolve()
                meta_path = result_package_dir / "meta.json"
                meta = _load_json(meta_path) if meta_path.exists() else None
                meta_old_lon = _finite(meta.get("pierce_lon_mean")) if meta else None
                meta_old_lat = _finite(meta.get("pierce_lat_mean")) if meta else None
                meta_old_phase = str(meta.get("preview_pierce_phase") or "") if meta else ""

                mismatch = (
                    not _same(old_lon, correct_lon)
                    or not _same(old_lat, correct_lat)
                    or (meta is not None and (not _same(meta_old_lon, correct_lon) or not _same(meta_old_lat, correct_lat) or meta_old_phase != phase_kind))
                )
                if mismatch:
                    mismatched += 1
                    geometry["pierce_lon_mean"] = correct_lon
                    geometry["pierce_lat_mean"] = correct_lat
                    _write_json(sidecar_path, sidecar)
                    if meta is not None:
                        meta["pierce_lon_mean"] = correct_lon
                        meta["pierce_lat_mean"] = correct_lat
                        meta["preview_pierce_phase"] = phase_kind
                        _write_json(meta_path, meta)
                    fixed += 1

                rows.append(
                    {
                        "dataset": dataset,
                        "event": event_dir.name,
                        "group_name": str(sidecar.get("group_name") or ""),
                        "pair_kind": pair_kind,
                        "phase_kind": phase_kind,
                        "sidecar_path": str(sidecar_path),
                        "members_used": str(len(lons)),
                        "old_lon": "" if old_lon is None else f"{old_lon:.9f}",
                        "old_lat": "" if old_lat is None else f"{old_lat:.9f}",
                        "correct_lon": f"{correct_lon:.9f}",
                        "correct_lat": f"{correct_lat:.9f}",
                        "meta_old_lon": "" if meta_old_lon is None else f"{meta_old_lon:.9f}",
                        "meta_old_lat": "" if meta_old_lat is None else f"{meta_old_lat:.9f}",
                        "meta_old_phase": meta_old_phase,
                        "changed": "yes" if mismatch else "no",
                    }
                )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "dataset",
            "event",
            "group_name",
            "pair_kind",
            "phase_kind",
            "sidecar_path",
            "members_used",
            "old_lon",
            "old_lat",
            "correct_lon",
            "correct_lat",
            "meta_old_lon",
            "meta_old_lat",
            "meta_old_phase",
            "changed",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"scanned={scanned}")
    print(f"mismatched={mismatched}")
    print(f"fixed={fixed}")
    print(REPORT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
