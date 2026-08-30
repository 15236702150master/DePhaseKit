#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone batch renderer for the DePhaseKit "TopDist std epicentral-distance" plot.

For each qualifying group-stacked SAC file it reproduces the project's own
WaveFigure TopDist standard export (stack trace on top + member traces ordered
by gcarc, with phase annotations), by driving the real WaveFigure methods.

Selection rule (per stacked file, read from its SAC t-phase headers):
  - has t5 AND t9  -> plot, aligned on t5
  - has t6 AND t8  -> plot, aligned on t6
  - otherwise      -> skip (no complete pair)

Phases drawn: all of t2..t9 (t0, t1 excluded), like the GUI "Std" export with
everything selected except the phase legend.
"""
import os
import sys
import math
import re
import argparse
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import matplotlib.pyplot as plt
import numpy as np

# Make the DePhaseKit package importable.
DPK_DIR = os.path.dirname(os.path.abspath(__file__))
if DPK_DIR not in sys.path:
    sys.path.insert(0, DPK_DIR)

import obspy  # noqa: E402
from WaveFigure import WaveFigure, init_standard_wave_figure, plot_standard_waves, set_standard_wave_axis  # noqa: E402
from stack_system import stack_output_dir_for_runtime  # noqa: E402
from stack_crustal_thickness import (  # noqa: E402
    fetch_obspy_ray_parameter,
    calculate_single_trace_thickness,
    reverse_station_coord,
)
from pierce_point_cache import PROJECT_ROOT  # noqa: E402

STACK_FILES_ROOT = os.path.join(PROJECT_ROOT, "data", "output", "stack", "stack_files")

# Phases to annotate: everything except t0 (P) and t1 (PcP).
PHASE_KEYS = ["2", "3", "4", "5", "6", "7", "8", "9"]


def _finite(header_value):
    try:
        value = float(header_value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    if value in (-12345.0, 12345.0):
        return None
    return value


def classify_stack(sac_path):
    """Return (align_marker_key, pair_label) or None if no complete pair."""
    sac = obspy.read(sac_path, headonly=True)[0].stats.sac
    t5 = _finite(sac.get("t5"))
    t6 = _finite(sac.get("t6"))
    t8 = _finite(sac.get("t8"))
    t9 = _finite(sac.get("t9"))
    if t5 is not None and t9 is not None:
        return "5", "t5t9"
    if t6 is not None and t8 is not None:
        return "6", "t6t8"
    return None


def build_wavefigure(stack_event_dir):
    """Construct a headless WaveFigure for a stack event workspace."""
    # ta_tb/xlim_preview are only used to build preview_modes, which we bypass
    # by calling the export helpers directly. Pass a minimal valid config.
    wf = WaveFigure(
        wavepath=stack_event_dir,
        xlim=[-50, 30],
        tmarker="t6",
        suffix=".sac",
        ta_tb="t6",
        xlim_preview=[-50, 30],
        axis_mode="absolute",
    )
    # Determine dt from one of the stack SAC files.
    stack_sacs = [f for f in os.listdir(stack_event_dir) if f.startswith("stack_") and f.endswith(".sac")]
    if not stack_sacs:
        raise RuntimeError(f"No stack SAC files in {stack_event_dir}")
    tr = obspy.read(os.path.join(stack_event_dir, stack_sacs[0]))[0]
    wf.dt = float(tr.stats.delta)
    # TopDist display: stack trace placed at top, members below by gcarc.
    wf.preview_stack_display_mode = "top"
    wf.preview_amplitude_scale = getattr(wf, "preview_amplitude_scale", 1.0) or 1.0
    return wf


def render_one(stack_event_dir, stack_wave_name, align_marker_key, output_dir, tag):
    wf = build_wavefigure(stack_event_dir)
    # _build_standard_preview_evtdata reads the active stack wave name off plotfig.
    wf.plotfig = SimpleNamespace(_stack_preview_wave_name=stack_wave_name)

    sidecar = wf._stack_sidecar_for_wave(stack_wave_name)
    window = sidecar.get("window")
    if not window or len(window) != 2:
        raise RuntimeError(f"No window in sidecar for {stack_wave_name}")
    x1, x2 = float(window[0]), float(window[1])

    evtdata = wf._build_standard_preview_evtdata(align_marker_key, x1, x2, order="gcarc")
    if evtdata is None:
        raise RuntimeError(f"Empty evtdata for {stack_wave_name}")

    export_options = dict(wf.standard_export_options)
    export_options["phase_legend"] = False
    export_options["export_gcarc"] = True
    export_options["export_az"] = False
    export_options["export_pierce"] = False
    export_options["export_pierce_group"] = False

    os.makedirs(output_dir, exist_ok=True)
    path, visible = save_topdist_std_with_stack_annotations(
        wf,
        evtdata,
        stack_wave_name,
        align_marker_key,
        output_dir,
        tag,
        PHASE_KEYS,
        export_options=export_options,
    )
    return path, visible


def _group_number_from_stack_name(stack_wave_name):
    base_name = os.path.basename(str(stack_wave_name or ""))
    for token in base_name.replace("\\", "/").split("/"):
        match = re.match(r"stack_group(\d+)(?:_|\.|$)", token)
        if match:
            return int(match.group(1))
    return None


def save_topdist_std_with_stack_annotations(wf, evtdata, stack_wave_name, align_marker_key, output_dir, timestamp_tag, phase_keys, export_options=None):
    if evtdata is None or getattr(evtdata, "sta_num", 0) == 0:
        return None, 0
    stack_group_number = _group_number_from_stack_name(stack_wave_name)
    stack_index = 0
    stack_trace = evtdata.wave_ori[0]
    for idx, trace in enumerate(getattr(evtdata, "wave_ori", [])):
        if getattr(trace.stats, "dpk_stack_preview_role", "") == "stack":
            stack_index = idx
            stack_trace = trace
            break
    stack_wave_name_from_evt = getattr(stack_trace.stats, "dpk_wave_name", "")
    if stack_wave_name_from_evt:
        stack_wave_name = stack_wave_name_from_evt
    sidecar = wf._stack_sidecar_for_wave(stack_wave_name)
    sidecar_markers = dict(sidecar.get("markers", {}) or {})

    fig, ax = init_standard_wave_figure()
    axis_values, y_ticks, y_ticklabels, ylabel = wf._preview_y_axis_config(evtdata, order="gcarc")
    colors = []
    linewidths = []
    for tr in evtdata.wave_ori:
        wave_name = getattr(tr.stats, "dpk_wave_name", "")
        color, linewidth = wf._preview_standard_wave_style(tr, wave_name)
        colors.append(color)
        linewidths.append(linewidth)
    wave_lines = plot_standard_waves(ax, evtdata, axis_values, colors, enf=wf.preview_amplitude_scale, linewidths=linewidths)
    for marker_key in PHASE_KEYS:
        value = _finite(sidecar_markers.get(f"t{marker_key}", math.nan))
        if value is None:
            continue
        stack_trace.stats.sac[f"t{marker_key}"] = float(value)
        wf.markers.setdefault(str(marker_key), {})[stack_wave_name] = float(value)
    set_standard_wave_axis(
        ax,
        evtdata,
        axis_values,
        xlabel=f"Time after {wf._phase_display_label(align_marker_key)} (s)",
        ylabel=ylabel,
        y_mode="gcarc",
        y_ticks=y_ticks,
        y_ticklabels=y_ticklabels,
    )
    visible_phase_count = wf._draw_standard_phase_annotations(
        ax,
        evtdata,
        axis_values,
        align_marker_key,
        phase_keys,
        reference_times=wf._preview_reference_times_from_evtdata(evtdata),
        wave_lines=wave_lines,
    )
    header_lines = wf._standard_export_header_lines(evtdata, export_options)
    if header_lines:
        fig.subplots_adjust(top=0.93)
        axes_box = ax.get_position()
        title_x = axes_box.x0 + axes_box.width * 0.5
        fig.text(
            title_x,
            0.975,
            '\n'.join(header_lines),
            fontsize=14,
            ha='center',
            va='top',
            linespacing=1.18,
        )
    else:
        fig.subplots_adjust(top=0.965)

    stack_y = float(axis_values[stack_index])
    if stack_group_number is not None:
        ax.annotate(
            str(stack_group_number),
            xy=(evtdata.x1 + 2.0, stack_y),
            xytext=(-16, 0),
            textcoords='offset points',
            ha='right',
            va='center',
            fontsize=16,
            color='red',
            fontweight='bold',
            bbox={
                'boxstyle': 'circle,pad=0.35',
                'facecolor': 'white',
                'edgecolor': 'red',
                'linewidth': 1.4,
                'alpha': 0.95,
            },
            zorder=10,
        )

    crust_text = _stack_crustal_text(wf, stack_wave_name)
    if crust_text:
        ax.text(
            evtdata.x1 + 16.0,
            stack_y + 0.35,
            crust_text,
            fontsize=14,
            color='#111111',
            fontweight='bold',
            ha='left',
            va='center',
            zorder=10,
        )

    output_path = os.path.join(output_dir, f"gcarc_{timestamp_tag}.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return output_path, visible_phase_count


def _stack_crustal_text(wf, stack_wave_name):
    sidecar = wf._stack_sidecar_for_wave(stack_wave_name)
    geometry = sidecar.get('geometry', {}) or {}
    event_info = sidecar.get('event', {}) or {}
    markers = dict(sidecar.get('markers', {}) or {})
    gcarc = _finite(geometry.get('gcarc_mean'))
    az = _finite(geometry.get('az_mean'))
    evdp = _finite(event_info.get('evdp'))
    evla = _finite(event_info.get('evla'))
    evlo = _finite(event_info.get('evlo'))
    if None in (gcarc, az, evdp, evla, evlo):
        return ''
    # stack 无单台坐标，用事件坐标 + gcarc_mean + az_mean 反推台站平均经纬度。
    try:
        stla, stlo = reverse_station_coord(evla, evlo, az, gcarc)
    except Exception:
        return ''

    t5 = _finite(markers.get('t5'))
    t6 = _finite(markers.get('t6'))
    t8 = _finite(markers.get('t8'))
    t9 = _finite(markers.get('t9'))

    # 与 calucate_xmP.py 一致：obspy 路径首点 ray param，Vp/Vs 写死 5.8/3.2。
    if t6 is not None and t8 is not None:
        try:
            ray_param = fetch_obspy_ray_parameter(evdp, evla, evlo, stla, stlo, phase='pP')
            thickness = calculate_single_trace_thickness(t6 - t8, ray_param, 'p')
            if not math.isnan(thickness) and thickness > 0.0:
                return f'H≈{thickness:.2f}km'
        except Exception:
            pass

    if t5 is not None and t9 is not None:
        try:
            ray_param = fetch_obspy_ray_parameter(evdp, evla, evlo, stla, stlo, phase='sP')
            thickness = calculate_single_trace_thickness(t5 - t9, ray_param, 's')
            if not math.isnan(thickness) and thickness > 0.0:
                return f'H≈{thickness:.2f}km'
        except Exception:
            pass
    return ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", help="event dir name under stack_files (debug single event)")
    ap.add_argument("--out", default=os.path.join(PROJECT_ROOT, "data", "output", "stack", "topdist_std_plots"))
    ap.add_argument("--show", action="store_true", help="print the produced png path only")
    args = ap.parse_args()

    datasets = [
        ("pick_other", os.path.join(STACK_FILES_ROOT, "pick_other")),
        ("pick_jandy", os.path.join(STACK_FILES_ROOT, "pick_jandy")),
    ]
    skip_events = {"2011_03_06_14_32_36", "2002_02_10_01_47_07"}

    produced = []
    for dataset, root in datasets:
        if not os.path.isdir(root):
            continue
        for event in sorted(os.listdir(root)):
            if dataset == "pick_jandy" and event in skip_events:
                continue
            if args.event and event != args.event:
                continue
            event_dir = os.path.join(root, event)
            if not os.path.isdir(event_dir):
                continue
            for sac_name in sorted(os.listdir(event_dir)):
                if not (sac_name.startswith("stack_group") and sac_name.endswith(".sac")):
                    continue
                sac_path = os.path.join(event_dir, sac_name)
                decision = classify_stack(sac_path)
                if decision is None:
                    print(f"[skip] {dataset}/{event}/{sac_name} (no complete t5/t9 or t6/t8 pair)")
                    continue
                align_key, pair = decision
                tag = f"{event}_{os.path.splitext(sac_name)[0]}_{pair}_t{align_key}"
                out_dir = os.path.join(args.out, dataset, event)
                try:
                    path, visible = render_one(event_dir, sac_name, align_key, out_dir, tag)
                    produced.append(path)
                    print(f"[ok]   {dataset}/{event}/{sac_name} pair={pair} align=t{align_key} phases={visible} -> {path}")
                except Exception as exc:
                    print(f"[err]  {dataset}/{event}/{sac_name}: {exc}")

    print(f"\nProduced {len(produced)} plots.")
    if args.show and produced:
        print(produced[0])


if __name__ == "__main__":
    main()
