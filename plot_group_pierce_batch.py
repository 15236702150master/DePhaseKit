#!/usr/bin/env python3
"""
Batch-render stack-group pierce-point maps, one figure per event.

Rule:
  - keep a stack group only if it has a complete `t5+t9` pair or `t6+t8` pair
  - `t5+t9` groups use `sP` pierce points
  - `t6+t8` groups use `pP` pierce points

The base map reuses the existing standard pierce GMT renderer so the output
keeps the same South Sandwich bathymetry style as the std export.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pierce_point_cache import PROJECT_ROOT, load_pierce_points


DPK_ROOT = PROJECT_ROOT / "opt" / "dephasekit"
STACK_ANALYSIS_ROOT = PROJECT_ROOT / "data" / "output" / "stack" / "analysis"
PHASE_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "phases"
PIERCE_CACHE_ROOT = PROJECT_ROOT / "data" / "output" / "pierce_points"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "stack" / "pierce_group_event_maps"
STANDARD_MAP_SCRIPT = DPK_ROOT / "plot_standard_pierce_map.sh"
DATASETS = ("pick_other", "pick_jandy")
MAP_REGION = "-32/-23/-61/-55"
GROUP_PALETTE = [
    "#d62728",
    "#1f77b4",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
    "#b22222",
    "#006400",
    "#8b008b",
    "#ff1493",
    "#008b8b",
    "#b8860b",
    "#4169e1",
    "#228b22",
]


@dataclass
class GroupSelection:
    dataset: str
    event: str
    group_number: int
    group_name: str
    sidecar_path: Path
    result_package_dir: Path
    pair_kind: str
    phase: str
    align_marker: str
    geometry_gcarc: float | None
    event_depth_km: float | None
    event_lon: float | None
    event_lat: float | None


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


def _group_color(group_number: int) -> str:
    return GROUP_PALETTE[(group_number - 1) % len(GROUP_PALETTE)]


def _group_number_from_name(group_name: str) -> int | None:
    text = str(group_name or "").strip().lower()
    if not text.startswith("group"):
        return None
    try:
        return int(text[5:])
    except ValueError:
        return None


def _pair_rule(markers: dict) -> tuple[str, str, str] | None:
    t5 = _finite(markers.get("t5"))
    t6 = _finite(markers.get("t6"))
    t8 = _finite(markers.get("t8"))
    t9 = _finite(markers.get("t9"))
    if t5 is not None and t9 is not None:
        return ("t5+t9", "sP", "t5")
    if t6 is not None and t8 is not None:
        return ("t6+t8", "pP", "t6")
    return None


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_members(members_path: Path) -> list[str]:
    members: list[str] = []
    with members_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("status") or "").strip().lower() != "used":
                continue
            wave_name = str(row.get("wave_name") or "").strip()
            if wave_name:
                members.append(wave_name)
    return members


def _phase_file_for_event(dataset: str, event: str, phase: str) -> Path:
    return PIERCE_CACHE_ROOT / dataset / event / f"pierce_points_{phase}_35.0km_iasp91.txt"


def _theory_summary_file(dataset: str, event: str) -> Path:
    phase_summary = PHASE_OUTPUT_ROOT / dataset / event / "theory_time_summary_iasp91.json"
    stack_summary = STACK_ANALYSIS_ROOT / dataset / event / "theory_time_summary_iasp91.json"
    if phase_summary.exists():
        return phase_summary
    return stack_summary


def collect_qualifying_groups(dataset: str, event: str) -> list[GroupSelection]:
    event_dir = STACK_ANALYSIS_ROOT / dataset / event
    if not event_dir.is_dir():
        return []

    groups: list[GroupSelection] = []
    for sidecar_path in sorted(event_dir.glob("stack_group*.stack.json")):
        payload = _load_json(sidecar_path)
        rule = _pair_rule(payload.get("markers", {}) or {})
        if rule is None:
            continue
        pair_kind, phase, align_marker = rule
        group_name = str(payload.get("group_name") or "").strip()
        group_number = _group_number_from_name(group_name)
        result_package_dir = Path(str(payload.get("result_package_dir") or "")).expanduser()
        event_info = payload.get("event", {}) or {}
        geometry = payload.get("geometry", {}) or {}
        if group_number is None or not result_package_dir.is_dir():
            continue
        groups.append(
            GroupSelection(
                dataset=dataset,
                event=event,
                group_number=group_number,
                group_name=group_name,
                sidecar_path=sidecar_path,
                result_package_dir=result_package_dir,
                pair_kind=pair_kind,
                phase=phase,
                align_marker=align_marker,
                geometry_gcarc=_finite(geometry.get("gcarc_mean")),
                event_depth_km=_finite(event_info.get("evdp")),
                event_lon=_finite(event_info.get("evlo")),
                event_lat=_finite(event_info.get("evla")),
            )
        )
    return sorted(groups, key=lambda item: item.group_number)


def build_event_points(groups: list[GroupSelection]) -> tuple[list[tuple[float, float, str, str]], OrderedDict[int, str]]:
    if not groups:
        return [], OrderedDict()

    cache_by_phase: dict[str, dict] = {}
    points: list[tuple[float, float, str, str]] = []
    legend: OrderedDict[int, str] = OrderedDict()

    for group in groups:
        phase_file = _phase_file_for_event(group.dataset, group.event, group.phase)
        if phase_file.exists():
            cache_by_phase.setdefault(group.phase, load_pierce_points(phase_file))
        else:
            cache_by_phase.setdefault(group.phase, {})
        members_path = group.result_package_dir / "members.txt"
        if not members_path.exists():
            continue
        color = _group_color(group.group_number)
        legend[group.group_number] = color
        record_map = cache_by_phase[group.phase]
        for wave_name in _read_members(members_path):
            record = record_map.get(wave_name)
            if record is None:
                continue
            points.append((float(record.longitude), float(record.latitude), color, "0"))

    return points, legend


def write_point_table(points: list[tuple[float, float, str, str]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for lon, lat, color, is_flip in points:
            handle.write(f"{lon:.6f} {lat:.6f} {color} {is_flip}\n")


def render_base_map(
    point_file: Path,
    output_prefix: Path,
    event_lon: float | None,
    event_lat: float | None,
) -> Path:
    env = os.environ.copy()
    env["INPUT_FILE"] = str(point_file)
    env["OUTPUT_PREFIX"] = str(output_prefix)
    env["REGION"] = MAP_REGION
    if event_lon is not None:
        env["EVENT_LON"] = f"{event_lon:.6f}"
    if event_lat is not None:
        env["EVENT_LAT"] = f"{event_lat:.6f}"
    subprocess.run(
        ["bash", str(STANDARD_MAP_SCRIPT)],
        cwd=str(DPK_ROOT),
        check=True,
        env=env,
    )
    return output_prefix.with_suffix(".png")


def _header_lines(dataset: str, event: str, groups: list[GroupSelection]) -> list[str]:
    first = groups[0]
    theory_path = _theory_summary_file(dataset, event)
    ppp = None
    spp = None
    if theory_path.exists():
        theory = _load_json(theory_path)
        ppp = _finite(theory.get("pP-P_mean"))
        spp = _finite(theory.get("sP-P_mean"))
    gcarc_values = [value for value in (group.geometry_gcarc for group in groups) if value is not None]
    pair_summary = ", ".join(f"grp{group.group_number}:{group.pair_kind}" for group in groups)
    gcarc_text = f"{sum(gcarc_values) / len(gcarc_values):.2f}°" if gcarc_values else "N/A"
    evdp_text = f"{first.event_depth_km:.1f} km" if first.event_depth_km is not None else "N/A"
    ppp_text = f"{ppp:.2f}s" if ppp is not None else "N/A"
    spp_text = f"{spp:.2f}s" if spp is not None else "N/A"
    return [
        f"Event {event}  |  Depth {evdp_text}  |  qualified groups:{len(groups)}",
        f"mod:iasp91  |  gcarc(avg):{gcarc_text}  |  pP-P:{ppp_text}  |  sP-P:{spp_text}",
        f"pairs: {pair_summary}",
    ]


def decorate_map(
    base_map_path: Path,
    output_path: Path,
    header_lines: list[str],
    legend: OrderedDict[int, str],
) -> None:
    image = mpimg.imread(base_map_path)
    fig = plt.figure(figsize=(10.3, 11.8))
    ax = fig.add_axes([0.06, 0.05, 0.78, 0.90])
    ax.imshow(image)
    ax.axis("off")

    legend_ax = fig.add_axes([0.86, 0.20, 0.09, 0.60])
    legend_ax.axis("off")
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    count = max(len(legend), 1)
    for idx, (group_number, color) in enumerate(legend.items()):
        y = 1.0 - (idx + 0.5) / count
        legend_ax.text(
            0.5,
            y,
            str(group_number),
            ha="center",
            va="center",
            fontsize=11,
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 1.1,
                "alpha": 0.96,
            },
        )

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_event(dataset: str, event: str, output_root: Path) -> Path | None:
    groups = collect_qualifying_groups(dataset, event)
    if not groups:
        return None
    points, legend = build_event_points(groups)
    if not points:
        return None

    event_output_dir = output_root / dataset / event
    event_output_dir.mkdir(parents=True, exist_ok=True)
    point_file = event_output_dir / "qualified_group_points.txt"
    write_point_table(points, point_file)

    event_lon = groups[0].event_lon
    event_lat = groups[0].event_lat
    base_prefix = event_output_dir / "pierce_group_base"
    base_map_path = render_base_map(point_file, base_prefix, event_lon, event_lat)
    final_output = event_output_dir / "pierce_group_event.png"
    decorate_map(base_map_path, final_output, _header_lines(dataset, event, groups), legend)
    return final_output


def iter_events(dataset: str, event: str | None) -> list[str]:
    if event:
        return [event]
    dataset_dir = STACK_ANALYSIS_ROOT / dataset
    if not dataset_dir.is_dir():
        return []
    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch plot stack-group pierce maps per event.")
    parser.add_argument("--dataset", choices=DATASETS, help="limit to one dataset")
    parser.add_argument("--event", help="limit to one event name")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_ROOT), help="output root directory")
    parser.add_argument("--show-first", action="store_true", help="print the first produced path only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.out).expanduser().resolve()
    datasets = [args.dataset] if args.dataset else list(DATASETS)
    produced: list[Path] = []

    for dataset in datasets:
        for event in iter_events(dataset, args.event):
            output_path = render_event(dataset, event, output_root)
            if output_path is None:
                continue
            produced.append(output_path)
            print(f"[ok] {dataset}/{event} -> {output_path}")

    print(f"Produced {len(produced)} event map(s).")
    if args.show_first and produced:
        print(produced[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
