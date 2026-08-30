#!/usr/bin/env python3
"""DSM 拟合对比 —— 数据准备逻辑（dephasekit 自包含，无外部脚本依赖）。

把原先通过 importlib 按路径加载的外部脚本
``codes/process/waveform_fit_compare/plot_observed_vs_synthetic.py`` 里的配对/对齐/
滤波/归一逻辑搬进 dephasekit 包内，使拟合窗/组总览窗不再依赖包外文件。

``build_pairs(args)`` 一次性产出已对齐/滤波/归一的 ``WaveformPair`` 列表：
``observed_t/y``、``synthetic_t/y`` 已相对对齐震相（x=0 即震相到时）。
绘图（分页叠绘 / 整组剖面）由 ``dsm_fit_compare_dialog.py`` 负责，本模块只管数据。
"""

from __future__ import annotations

import csv
import math
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from obspy import read
from obspy.taup import TauPyModel


KM_PER_DEG = 111.19
OBSERVED_PATTERNS = ("*.sac", "*.SAC")
SYNTHETIC_PATTERNS = ("*.bhz", "*.BHZ")
MISFIT_MODE_LINEAR = "linear"
MISFIT_MODE_SHAPE = "shape"


@dataclass
class WaveformPair:
    station_key: str
    distance_deg: float
    azimuth_deg: float | None
    align_time_s: float
    observed_path: Path
    synthetic_path: Path
    observed_t: np.ndarray
    observed_y: np.ndarray
    synthetic_t: np.ndarray
    synthetic_y: np.ndarray
    synthetic_align_time_s: float = 0.0
    target_phase: str = ""
    observed_target_delta_s: float = math.nan
    synthetic_target_delta_s: float = math.nan
    target_delta_residual_s: float = math.nan
    time_shift_s: float = 0.0
    amplitude_factor: float = 1.0
    cross_corr_max: float = 0.0
    misfit_mode: str = MISFIT_MODE_SHAPE
    misfit_value: float = 1.0
    misfit_cc: float = 1.0
    shape_misfit: float = 1.0
    variance_reduction: float = 0.0
    sample_rate_used_hz: float = 0.0
    window_npts: int = 0


@dataclass
class CrossCorrelationResult:
    time_grid: np.ndarray
    observed_y: np.ndarray
    synthetic_y: np.ndarray
    time_shift_s: float
    amplitude_factor: float
    cross_corr_max: float
    misfit_mode: str
    misfit_value: float
    misfit_cc: float
    shape_misfit: float
    variance_reduction: float
    sample_rate_used_hz: float
    window_npts: int


THEORETICAL_PHASE_SLOTS = {
    "P": "t0",
    "pP": "t2",
    "sP": "t3",
}
THEORETICAL_SLOT_PHASES = {
    "t0": "P",
    "t2": "pP",
    "t3": "sP",
}
ACTUAL_PHASE_SLOTS = {
    "P": "t7",
    "pP": "t6",
    "sP": "t5",
}
ACTUAL_SLOT_PHASES = {
    "t7": "P",
    "t6": "pP",
    "t5": "sP",
}
OBSERVED_MANUAL_PHASE_SLOTS = ACTUAL_PHASE_SLOTS
OBSERVED_MANUAL_SLOT_PHASES = ACTUAL_SLOT_PHASES


def station_key(path: Path) -> str:
    parts = path.name.split(".")
    if len(parts) < 2:
        return path.stem
    return f"{parts[0]}.{parts[1]}"


def discover_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path not in seen and path.is_file():
                files.append(path)
                seen.add(path)
    return files


def build_station_map(root: Path, patterns: tuple[str, ...]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in discover_files(root, patterns):
        mapping.setdefault(station_key(path), path)
    return mapping


def get_sac_value(sac, key: str) -> float | None:
    value = getattr(sac, key, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or value <= -12344:
        return None
    return value


def get_distance_deg(sac) -> float | None:
    gcarc = get_sac_value(sac, "gcarc")
    if gcarc is not None:
        return gcarc
    dist_km = get_sac_value(sac, "dist")
    if dist_km is None:
        return None
    return dist_km / KM_PER_DEG


def get_first_arrival(model: TauPyModel, depth_km: float, distance_deg: float, phase_name: str) -> float | None:
    arrivals = model.get_travel_times(
        source_depth_in_km=float(depth_km),
        distance_in_degree=float(distance_deg),
        phase_list=[phase_name],
    )
    if not arrivals:
        return None
    return float(arrivals[0].time)


def get_header_phase_time(sac, phase_name: str) -> float | None:
    target = phase_name.strip()
    if not target:
        return None
    lower_target = target.lower()
    if lower_target in {f"t{idx}" for idx in range(10)}:
        return get_sac_value(sac, lower_target)
    for idx in range(10):
        time_value = get_sac_value(sac, f"t{idx}")
        label = getattr(sac, f"kt{idx}", None)
        if time_value is None or label is None:
            continue
        if str(label).strip() == target:
            return time_value
    return None


def semantic_phase_for_alignment(phase_name: str) -> str:
    """Return the physical phase represented by a UI alignment token.

    In the user's picked SAC convention, ``t7/t6/t5`` are actual
    ``P/pP/sP`` picks. Theoretical TauP references live separately in
    ``t0/t2/t3`` and should not replace actual picks for metrics.
    """
    phase = phase_name.strip()
    lower = phase.lower()
    if lower in ACTUAL_SLOT_PHASES:
        return ACTUAL_SLOT_PHASES[lower]
    if lower in THEORETICAL_SLOT_PHASES:
        return THEORETICAL_SLOT_PHASES[lower]
    for canonical in ACTUAL_PHASE_SLOTS:
        if lower == canonical.lower():
            return canonical
    return phase


def observed_manual_slot_for_phase(phase_name: str) -> str | None:
    phase = semantic_phase_for_alignment(phase_name)
    lower = phase.lower()
    for canonical, slot in ACTUAL_PHASE_SLOTS.items():
        if lower == canonical.lower() or phase_name.strip().lower() == slot:
            return slot
    return None


def get_observed_manual_phase_time(sac, phase_name: str) -> float | None:
    slot = observed_manual_slot_for_phase(phase_name)
    if slot is None:
        return None
    return get_sac_value(sac, slot)


def get_actual_phase_time(sac, phase_name: str) -> float | None:
    return get_observed_manual_phase_time(sac, phase_name)


def _snap_time_to_nearest_sample(trace, time_s: float) -> float:
    """Return the absolute time of the nearest native sample without resampling."""
    sac = getattr(trace.stats, "sac", None)
    begin_s = get_sac_value(sac, "b") if sac is not None else None
    if begin_s is None:
        begin_s = 0.0
    delta = float(trace.stats.delta)
    if delta <= 0:
        raise ValueError(f"{trace.id or 'trace'} has invalid sample interval")
    sample_index = int(round((float(time_s) - begin_s) / delta))
    sample_index = max(0, min(sample_index, int(trace.stats.npts) - 1))
    return float(begin_s + sample_index * delta)


def extract_window(
    trace,
    align_time_s: float,
    time_min: float,
    time_max: float,
    *,
    snap_align_to_sample: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        raise ValueError(f"{trace.id or 'trace'} has no SAC header")
    begin_s = get_sac_value(sac, "b")
    if begin_s is None:
        begin_s = 0.0
    if snap_align_to_sample:
        align_time_s = _snap_time_to_nearest_sample(trace, align_time_s)
    rel_t = trace.times() + begin_s - align_time_s
    mask = (rel_t >= time_min) & (rel_t <= time_max)
    return rel_t[mask], trace.data[mask].astype(np.float64)


def detrend_mean(data: np.ndarray) -> np.ndarray:
    if data.size == 0:
        return data
    return data - float(np.mean(data))


def maybe_filter_trace(trace, freqmin: float | None, freqmax: float | None) -> None:
    if freqmin is None and freqmax is None:
        return
    trace.detrend("demean")
    trace.detrend("linear")
    trace.taper(max_percentage=0.05, type="cosine")
    if freqmin is not None and freqmax is not None:
        trace.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    elif freqmin is not None:
        trace.filter("highpass", freq=freqmin, corners=4, zerophase=True)
    elif freqmax is not None:
        trace.filter("lowpass", freq=freqmax, corners=4, zerophase=True)


def normalize_pair(
    observed_y: np.ndarray,
    synthetic_y: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    observed_y = detrend_mean(observed_y)
    synthetic_y = detrend_mean(synthetic_y)

    if mode == "separate":
        obs_scale = np.max(np.abs(observed_y)) if observed_y.size else 0.0
        syn_scale = np.max(np.abs(synthetic_y)) if synthetic_y.size else 0.0
        if obs_scale > 0:
            observed_y = observed_y / obs_scale
        if syn_scale > 0:
            synthetic_y = synthetic_y / syn_scale
        return observed_y, synthetic_y

    joined_max = max(
        float(np.max(np.abs(observed_y))) if observed_y.size else 0.0,
        float(np.max(np.abs(synthetic_y))) if synthetic_y.size else 0.0,
    )
    if joined_max > 0:
        observed_y = observed_y / joined_max
        synthetic_y = synthetic_y / joined_max
    return observed_y, synthetic_y


def _median_sample_interval(times: np.ndarray) -> float:
    if times.size < 2:
        raise ValueError("not enough samples for cross-correlation")
    diffs = np.diff(times)
    if np.any(diffs <= 0):
        raise ValueError("metric window time axis is not strictly increasing")
    return float(np.median(diffs))


def _validate_native_metric_grid(
    observed_t: np.ndarray,
    observed_y: np.ndarray,
    synthetic_t: np.ndarray,
    synthetic_y: np.ndarray,
    time_min: float,
    time_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if observed_t.size < 2 or synthetic_t.size < 2:
        raise ValueError("not enough samples for cross-correlation")

    observed_dt = _median_sample_interval(observed_t)
    synthetic_dt = _median_sample_interval(synthetic_t)
    dt = 0.5 * (observed_dt + synthetic_dt)
    tolerance = max(
        dt,
        observed_dt,
        synthetic_dt,
    ) * 1.5
    if observed_t[0] > time_min + tolerance or observed_t[-1] < time_max - tolerance:
        raise ValueError("observed window does not cover requested metric window")
    if synthetic_t[0] > time_min + tolerance or synthetic_t[-1] < time_max - tolerance:
        raise ValueError("synthetic window does not cover requested metric window")

    dt_tolerance = max(abs(observed_dt), abs(synthetic_dt)) * 1e-4
    if abs(observed_dt - synthetic_dt) > dt_tolerance:
        raise ValueError(
            f"sampling rates differ before metrics: observed={1.0 / observed_dt:.6g} Hz, "
            f"synthetic={1.0 / synthetic_dt:.6g} Hz"
        )

    regularity_tolerance = max(dt_tolerance, dt * 1e-4)
    if np.max(np.abs(np.diff(observed_t) - observed_dt)) > regularity_tolerance:
        raise ValueError("observed metric window is not evenly sampled")
    if np.max(np.abs(np.diff(synthetic_t) - synthetic_dt)) > regularity_tolerance:
        raise ValueError("synthetic metric window is not evenly sampled")

    boundary_tolerance = dt * 0.5
    observed_mask = (
        (observed_t >= time_min - boundary_tolerance)
        & (observed_t <= time_max + boundary_tolerance)
    )
    synthetic_mask = (
        (synthetic_t >= time_min - boundary_tolerance)
        & (synthetic_t <= time_max + boundary_tolerance)
    )
    observed_t = observed_t[observed_mask]
    observed_y = observed_y[observed_mask]
    synthetic_t = synthetic_t[synthetic_mask]
    synthetic_y = synthetic_y[synthetic_mask]
    if observed_t.size < 2 or synthetic_t.size < 2:
        raise ValueError("not enough samples inside requested metric window")
    if observed_t.size != synthetic_t.size:
        raise ValueError(
            "observed and synthetic metric windows must have the same sample count before metrics"
        )

    relative_tolerance = dt * 0.25
    if np.max(np.abs(observed_t - synthetic_t)) > relative_tolerance:
        raise ValueError(
            "observed and synthetic metric windows are not on the same relative time grid"
        )

    time_grid = observed_t.astype(np.float64, copy=True)
    observed_grid = observed_y.astype(np.float64, copy=True)
    synthetic_grid = synthetic_y.astype(np.float64, copy=True)
    return time_grid, observed_grid, synthetic_grid


def _shift_by_integer_lag(data: np.ndarray, lag_samples: int) -> np.ndarray:
    shifted = np.zeros_like(data, dtype=np.float64)
    if lag_samples > 0:
        shifted[lag_samples:] = data[:-lag_samples]
    elif lag_samples < 0:
        shifted[:lag_samples] = data[-lag_samples:]
    else:
        shifted[:] = data
    return shifted


def _overlap_for_lag(observed: np.ndarray, synthetic: np.ndarray, lag_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_samples > 0:
        return observed[lag_samples:], synthetic[:-lag_samples]
    if lag_samples < 0:
        return observed[:lag_samples], synthetic[-lag_samples:]
    return observed, synthetic


def normalize_misfit_mode(mode: str | None) -> str:
    if mode == MISFIT_MODE_LINEAR:
        return MISFIT_MODE_LINEAR
    return MISFIT_MODE_SHAPE


def misfit_values_from_cc(cross_corr: float, mode: str | None = MISFIT_MODE_SHAPE) -> tuple[float, float, float, str]:
    """Return selected, linear, and paper-style shape misfits for a CC value."""
    selected_mode = normalize_misfit_mode(mode)
    cc = float(np.clip(cross_corr, -1.0, 1.0))
    linear_misfit = 1.0 - cc
    shape_misfit = 1.0 - cc * cc
    selected = linear_misfit if selected_mode == MISFIT_MODE_LINEAR else shape_misfit
    return selected, linear_misfit, shape_misfit, selected_mode


def compute_cross_correlation_result(
    observed_t: np.ndarray,
    observed_y: np.ndarray,
    synthetic_t: np.ndarray,
    synthetic_y: np.ndarray,
    time_min: float,
    time_max: float,
    tau_max_s: float = 10.0,
    misfit_mode: str = MISFIT_MODE_SHAPE,
) -> CrossCorrelationResult:
    """Compute normalized cross-correlation metrics on a native common grid.

    This function intentionally does not interpolate or resample. It assumes the
    caller has already made the observed and synthetic windows scientifically
    comparable: same component/unit convention, same preprocessing band, same
    sample interval, same sample count, and the same relative time grid.

    ``time_shift_s > 0`` means the synthetic waveform should be moved later
    (rightward) to match the observed waveform.
    """
    time_grid, observed_grid, synthetic_grid = _validate_native_metric_grid(
        observed_t,
        observed_y,
        synthetic_t,
        synthetic_y,
        time_min,
        time_max,
    )
    if time_grid.size < 2:
        raise ValueError("metric grid has fewer than two samples")

    dt = float(time_grid[1] - time_grid[0])
    observed_zero = detrend_mean(observed_grid)
    synthetic_zero = detrend_mean(synthetic_grid)
    denominator = float(np.linalg.norm(observed_zero) * np.linalg.norm(synthetic_zero))
    if denominator <= 0:
        raise ValueError("zero-energy waveform window")

    corr = np.correlate(observed_zero, synthetic_zero, mode="full")
    center = len(synthetic_zero) - 1
    max_lag = int(round(abs(float(tau_max_s)) / dt))
    max_lag = min(max_lag, len(synthetic_zero) - 1)
    lo = max(0, center - max_lag)
    hi = min(len(corr), center + max_lag + 1)
    search = corr[lo:hi]
    lag_samples = int(np.argmax(search) + lo - center)
    time_shift_s = float(lag_samples * dt)
    cross_corr_max = float(corr[center + lag_samples] / denominator)
    misfit_value, misfit_cc, shape_misfit, selected_mode = misfit_values_from_cc(cross_corr_max, misfit_mode)

    observed_overlap, synthetic_overlap = _overlap_for_lag(observed_zero, synthetic_zero, lag_samples)
    if observed_overlap.size < 2 or synthetic_overlap.size < 2:
        raise ValueError("not enough overlapping samples after time shift")
    shifted_overlap_zero = detrend_mean(synthetic_overlap)
    observed_overlap_zero = detrend_mean(observed_overlap)
    amp_den = float(np.sum(shifted_overlap_zero * shifted_overlap_zero))
    amplitude_factor = (
        float(np.sum(observed_overlap_zero * shifted_overlap_zero) / amp_den)
        if amp_den > 0
        else 1.0
    )
    resid = observed_overlap_zero - amplitude_factor * shifted_overlap_zero
    vr_den = float(np.sum(observed_overlap_zero * observed_overlap_zero))
    variance_reduction = 1.0 - float(np.sum(resid * resid) / vr_den) if vr_den > 0 else 0.0

    shifted_synthetic = _shift_by_integer_lag(synthetic_zero, lag_samples)
    aligned_synthetic = amplitude_factor * shifted_synthetic

    return CrossCorrelationResult(
        time_grid=time_grid,
        observed_y=observed_zero,
        synthetic_y=aligned_synthetic,
        time_shift_s=time_shift_s,
        amplitude_factor=amplitude_factor,
        cross_corr_max=cross_corr_max,
        misfit_mode=selected_mode,
        misfit_value=misfit_value,
        misfit_cc=misfit_cc,
        shape_misfit=shape_misfit,
        variance_reduction=variance_reduction,
        sample_rate_used_hz=float(1.0 / dt),
        window_npts=int(time_grid.size),
    )


def _resolve_observed_alignment_time(args: Namespace, sac, model: TauPyModel, depth_km: float, distance_deg: float) -> float | None:
    align_time_s: float | None = None
    if args.align_source in {"header", "header_then_taup"}:
        align_time_s = get_header_phase_time(sac, args.align_phase)
    if align_time_s is None and args.align_source in {"taup", "header_then_taup"}:
        align_time_s = get_first_arrival(model, depth_km, distance_deg, semantic_phase_for_alignment(args.align_phase))
    return align_time_s


def _resolve_synthetic_alignment_time(args: Namespace, sac, model: TauPyModel, depth_km: float, distance_deg: float) -> float | None:
    align_time_s: float | None = None
    if args.align_source in {"header", "header_then_taup"}:
        align_time_s = get_header_phase_time(sac, args.align_phase)
    if align_time_s is None and args.align_source in {"taup", "header_then_taup"}:
        align_time_s = get_first_arrival(model, depth_km, distance_deg, semantic_phase_for_alignment(args.align_phase))
    return align_time_s


def _resolve_actual_alignment_time(args: Namespace, sac) -> float | None:
    manual_slot = observed_manual_slot_for_phase(args.align_phase)
    if manual_slot is None:
        return None
    return get_sac_value(sac, manual_slot)


def _resolve_interval_phase_time(
    args: Namespace,
    sac,
    model: TauPyModel,
    depth_km: float,
    distance_deg: float,
    phase_name: str,
) -> float | None:
    if getattr(args, "use_observed_manual_picks", False):
        return get_actual_phase_time(sac, phase_name)

    align_source = getattr(args, "align_source", "header_then_taup")
    phase_time: float | None = None
    if align_source in {"header", "header_then_taup"}:
        phase_time = get_header_phase_time(sac, phase_name)
    if phase_time is None and align_source in {"taup", "header_then_taup"}:
        phase_time = get_first_arrival(model, depth_km, distance_deg, semantic_phase_for_alignment(phase_name))
    return phase_time


def _target_phase_delta(
    args: Namespace,
    observed_sac,
    synthetic_sac,
    model: TauPyModel,
    depth_km: float,
    distance_deg: float,
    synthetic_depth_km: float,
    synthetic_distance_deg: float,
) -> tuple[str, float, float, float]:
    target_phase = str(getattr(args, "target_phase", "") or "").strip()
    if not target_phase:
        return "", math.nan, math.nan, math.nan

    observed_anchor = _resolve_interval_phase_time(args, observed_sac, model, depth_km, distance_deg, args.align_phase)
    synthetic_anchor = _resolve_interval_phase_time(
        args,
        synthetic_sac,
        model,
        synthetic_depth_km,
        synthetic_distance_deg,
        args.align_phase,
    )
    observed_target = _resolve_interval_phase_time(args, observed_sac, model, depth_km, distance_deg, target_phase)
    synthetic_target = _resolve_interval_phase_time(
        args,
        synthetic_sac,
        model,
        synthetic_depth_km,
        synthetic_distance_deg,
        target_phase,
    )
    if None in {observed_anchor, synthetic_anchor, observed_target, synthetic_target}:
        return target_phase, math.nan, math.nan, math.nan

    observed_delta = float(observed_target - observed_anchor)
    synthetic_delta = float(synthetic_target - synthetic_anchor)
    return target_phase, observed_delta, synthetic_delta, observed_delta - synthetic_delta


def build_pairs(args: Namespace) -> tuple[list[WaveformPair], list[str]]:
    """配对观测/理论波形，按震相对齐、滤波、归一，返回 (pairs, skipped)。"""
    observed_map = build_station_map(args.observed_dir, OBSERVED_PATTERNS)
    synthetic_map = build_station_map(args.synthetic_dir, SYNTHETIC_PATTERNS)
    # 场景A：把观测侧限定为 dephasekit 当前可见集合（NET.STA），只配对"可见 ∩ 有合成"的台站。
    filter_keys = getattr(args, "observed_station_keys", None)
    if filter_keys:
        observed_map = {k: v for k, v in observed_map.items() if k in filter_keys}
    common_keys = sorted(set(observed_map) & set(synthetic_map))
    model = TauPyModel(model=args.taup_model)

    pairs: list[WaveformPair] = []
    skipped: list[str] = []

    for key in common_keys:
        observed_path = observed_map[key]
        synthetic_path = synthetic_map[key]

        observed_trace = read(str(observed_path))[0]
        synthetic_trace = read(str(synthetic_path))[0]
        maybe_filter_trace(observed_trace, args.bandpass_freqmin, args.bandpass_freqmax)
        maybe_filter_trace(synthetic_trace, args.bandpass_freqmin, args.bandpass_freqmax)
        observed_sac = getattr(observed_trace.stats, "sac", None)
        synthetic_sac = getattr(synthetic_trace.stats, "sac", None)
        if observed_sac is None or synthetic_sac is None:
            skipped.append(f"{key}: missing SAC header")
            continue

        distance_deg = get_distance_deg(observed_sac)
        depth_km = get_sac_value(observed_sac, "evdp")
        if distance_deg is None:
            skipped.append(f"{key}: missing observed distance")
            continue
        if depth_km is None:
            skipped.append(f"{key}: missing observed depth")
            continue
        if args.distance_min is not None and distance_deg < args.distance_min:
            continue
        if args.distance_max is not None and distance_deg > args.distance_max:
            continue

        synthetic_distance_deg = get_distance_deg(synthetic_sac)
        synthetic_depth_km = get_sac_value(synthetic_sac, "evdp")
        if synthetic_distance_deg is None:
            synthetic_distance_deg = distance_deg
        if synthetic_depth_km is None:
            synthetic_depth_km = depth_km

        observed_align_time_s = _resolve_observed_alignment_time(args, observed_sac, model, depth_km, distance_deg)
        synthetic_align_time_s = _resolve_synthetic_alignment_time(
            args,
            synthetic_sac,
            model,
            synthetic_depth_km,
            synthetic_distance_deg,
        )

        if observed_align_time_s is None or synthetic_align_time_s is None:
            skipped.append(f"{key}: no usable {args.align_phase} alignment time")
            continue

        observed_t, observed_y = extract_window(observed_trace, observed_align_time_s, args.time_min, args.time_max)
        synthetic_t, synthetic_y = extract_window(synthetic_trace, synthetic_align_time_s, args.time_min, args.time_max)
        if observed_t.size == 0 or synthetic_t.size == 0:
            skipped.append(f"{key}: empty plotting window")
            continue

        target_phase, observed_target_delta_s, synthetic_target_delta_s, target_delta_residual_s = _target_phase_delta(
            args,
            observed_sac,
            synthetic_sac,
            model,
            depth_km,
            distance_deg,
            synthetic_depth_km,
            synthetic_distance_deg,
        )

        time_shift_s = 0.0
        amplitude_factor = 1.0
        cross_corr_max = 0.0
        misfit_mode = normalize_misfit_mode(getattr(args, "misfit_mode", MISFIT_MODE_SHAPE))
        misfit_value = 1.0
        misfit_cc = 1.0
        shape_misfit = 1.0
        variance_reduction = 0.0
        sample_rate_used_hz = 0.0
        window_npts = 0

        if getattr(args, "use_crosscorr_align", False):
            metric_observed_t, metric_observed_y = extract_window(
                observed_trace,
                observed_align_time_s,
                args.time_min,
                args.time_max,
                snap_align_to_sample=True,
            )
            metric_synthetic_t, metric_synthetic_y = extract_window(
                synthetic_trace,
                synthetic_align_time_s,
                args.time_min,
                args.time_max,
                snap_align_to_sample=True,
            )
            if getattr(args, "use_observed_manual_picks", False):
                observed_metric_align_time_s = _resolve_actual_alignment_time(args, observed_sac)
                synthetic_metric_align_time_s = _resolve_actual_alignment_time(args, synthetic_sac)
                if observed_metric_align_time_s is None or synthetic_metric_align_time_s is None:
                    skipped.append(f"{key}: missing actual pick for {args.align_phase} metrics")
                    continue
                metric_observed_t, metric_observed_y = extract_window(
                    observed_trace,
                    observed_metric_align_time_s,
                    args.time_min,
                    args.time_max,
                    snap_align_to_sample=True,
                )
                metric_synthetic_t, metric_synthetic_y = extract_window(
                    synthetic_trace,
                    synthetic_metric_align_time_s,
                    args.time_min,
                    args.time_max,
                    snap_align_to_sample=True,
                )
                if metric_observed_t.size == 0 or metric_synthetic_t.size == 0:
                    skipped.append(f"{key}: empty actual-pick metric window")
                    continue
            try:
                result = compute_cross_correlation_result(
                    metric_observed_t,
                    metric_observed_y,
                    metric_synthetic_t,
                    metric_synthetic_y,
                    args.time_min,
                    args.time_max,
                    tau_max_s=float(getattr(args, "crosscorr_tau_max", 10.0) or 10.0),
                    misfit_mode=misfit_mode,
                )
            except ValueError as exc:
                skipped.append(f"{key}: {exc}")
                continue
            time_shift_s = result.time_shift_s
            amplitude_factor = result.amplitude_factor
            cross_corr_max = result.cross_corr_max
            misfit_mode = result.misfit_mode
            misfit_value = result.misfit_value
            misfit_cc = result.misfit_cc
            shape_misfit = result.shape_misfit
            variance_reduction = result.variance_reduction
            sample_rate_used_hz = result.sample_rate_used_hz
            window_npts = result.window_npts
        observed_y, synthetic_y = normalize_pair(observed_y, synthetic_y, args.normalize)
        azimuth_deg = get_sac_value(observed_sac, "az")
        pairs.append(
            WaveformPair(
                station_key=key,
                distance_deg=float(distance_deg),
                azimuth_deg=float(azimuth_deg) if azimuth_deg is not None else None,
                align_time_s=float(observed_align_time_s),
                observed_path=observed_path,
                synthetic_path=synthetic_path,
                observed_t=observed_t,
                observed_y=observed_y,
                synthetic_t=synthetic_t,
                synthetic_y=synthetic_y,
                synthetic_align_time_s=float(synthetic_align_time_s),
                target_phase=target_phase,
                observed_target_delta_s=observed_target_delta_s,
                synthetic_target_delta_s=synthetic_target_delta_s,
                target_delta_residual_s=target_delta_residual_s,
                time_shift_s=time_shift_s,
                amplitude_factor=amplitude_factor,
                cross_corr_max=cross_corr_max,
                misfit_mode=misfit_mode,
                misfit_value=misfit_value,
                misfit_cc=misfit_cc,
                shape_misfit=shape_misfit,
                variance_reduction=variance_reduction,
                sample_rate_used_hz=sample_rate_used_hz,
                window_npts=window_npts,
            )
        )

    sort_key = {
        "distance": lambda pair: (pair.distance_deg, pair.station_key),
        "azimuth": lambda pair: (pair.azimuth_deg if pair.azimuth_deg is not None else float("inf"), pair.station_key),
        "station": lambda pair: pair.station_key,
    }[args.sort_by]
    pairs.sort(key=sort_key, reverse=args.reverse_order)

    if args.max_traces is not None:
        pairs = pairs[: args.max_traces]

    return pairs, skipped


def write_pair_csv(path: Path, pairs: list[WaveformPair]) -> None:
    """可选：输出参与绘图的配对清单 CSV（仅元数据，不含波形样本）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "station_key",
                "distance_deg",
                "azimuth_deg",
                "align_time_s",
                "synthetic_align_time_s",
                "target_phase",
                "observed_target_delta_s",
                "synthetic_target_delta_s",
                "target_delta_residual_s",
                "time_shift_s",
                "amplitude_factor",
                "cross_corr_max",
                "misfit_mode",
                "misfit_value",
                "misfit_cc",
                "shape_misfit",
                "variance_reduction",
                "sample_rate_used_hz",
                "window_npts",
                "observed_path",
                "synthetic_path",
            ],
        )
        writer.writeheader()

        def fmt_optional(value: float) -> str:
            return "" if not np.isfinite(value) else f"{value:.5f}"

        for pair in pairs:
            writer.writerow(
                {
                    "station_key": pair.station_key,
                    "distance_deg": f"{pair.distance_deg:.5f}",
                    "azimuth_deg": "" if pair.azimuth_deg is None else f"{pair.azimuth_deg:.5f}",
                    "align_time_s": f"{pair.align_time_s:.5f}",
                    "synthetic_align_time_s": f"{pair.synthetic_align_time_s:.5f}",
                    "target_phase": pair.target_phase,
                    "observed_target_delta_s": fmt_optional(pair.observed_target_delta_s),
                    "synthetic_target_delta_s": fmt_optional(pair.synthetic_target_delta_s),
                    "target_delta_residual_s": fmt_optional(pair.target_delta_residual_s),
                    "time_shift_s": f"{pair.time_shift_s:.5f}",
                    "amplitude_factor": f"{pair.amplitude_factor:.5f}",
                    "cross_corr_max": f"{pair.cross_corr_max:.5f}",
                    "misfit_mode": pair.misfit_mode,
                    "misfit_value": f"{pair.misfit_value:.5f}",
                    "misfit_cc": f"{pair.misfit_cc:.5f}",
                    "shape_misfit": f"{pair.shape_misfit:.5f}",
                    "variance_reduction": f"{pair.variance_reduction:.5f}",
                    "sample_rate_used_hz": f"{pair.sample_rate_used_hz:.5f}",
                    "window_npts": str(pair.window_npts),
                    "observed_path": str(pair.observed_path),
                    "synthetic_path": str(pair.synthetic_path),
                }
            )
