#!/usr/bin/env python3
"""Stack 厚度审阅模块的纯数据层。

从 ``ThicknessPoint`` + sidecar + members.txt + 穿透点缓存组装三类数据：
  1. group 成员信息（``GroupMember``）—— 用于列表/成员穿透点下垫
  2. 成员穿透点（``load_member_pierce_points``）—— 地图下垫散点
  3. 预览渲染数据（``PreviewTrace``）—— 左侧 stack 预览（TopDist / Overlay）

预览对齐切片复用 ``WaveFigure.EvtData`` (WaveFigure.py:627) 的逻辑：
``t1_index = (reference_t - b) / dt``，``start = t1_index + x1/dt``。
关键简化：本项目 stack.sac 与成员 SAC 均为 ``b=0``、t 头段为相对秒，故每道的
``reference_t`` 统一取其 SAC ``t{align}`` 头段（stack 的 t{align}=-x1，成员为其拾取时刻），
modern / legacy 两帧同公式适用。

本模块零 GUI 依赖、零 WaveFigure 依赖，可单测。
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import obspy

from pierce_point_cache import (
    DEFAULT_PIERCE_DEPTH_KM,
    load_pierce_points,
    pierce_file_path,
)
from stack_group_thickness_index import (
    DEFAULT_PIERCE_MODEL,
    ThicknessPoint,
)

SAC_KM_PER_DEG = 111.19492


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class GroupMember:
    wave_name: str
    gcarc: float
    longitude: float | None
    latitude: float | None
    sac_path: Path | None


@dataclass
class PreviewTrace:
    wave_name: str
    is_stack: bool
    t_array: np.ndarray   # 时间轴（相对对齐头段的秒），长度 = wavelength
    y_array: np.ndarray   # 振幅（Overlay 已归一化；TopDist 为原始振幅，按 gcarc 偏移由调用方处理）
    gcarc: float
    reference_t: float    # 对齐头段时刻（与 b 同帧）


@dataclass
class PreviewBundle:
    point: ThicknessPoint
    traces: list[PreviewTrace] = field(default_factory=list)
    align_marker: str = ""
    window: tuple[float, float] = (0.0, 0.0)
    display_mode: str = "overlay"
    stack_available: bool = True
    note: str = ""


# ---------------------------------------------------------------------------
# SAC 小工具
# ---------------------------------------------------------------------------

def _sac_float(trace, key: str, default: float = float("nan")) -> float:
    sac = getattr(trace.stats, "sac", None)
    if sac is None:
        return default
    try:
        value = sac[key]
    except (KeyError, AttributeError, TypeError):
        if str(key).lower() != "gcarc":
            return default
        dist_km = _sac_float(trace, "dist", default)
        if math.isfinite(dist_km):
            return dist_km / SAC_KM_PER_DEG
        evla = _sac_float(trace, "evla", default)
        evlo = _sac_float(trace, "evlo", default)
        stla = _sac_float(trace, "stla", default)
        stlo = _sac_float(trace, "stlo", default)
        if all(math.isfinite(v) for v in (evla, evlo, stla, stlo)):
            try:
                from obspy.geodetics import locations2degrees
                return float(locations2degrees(evla, evlo, stla, stlo))
            except Exception:
                return default
        return default
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    if value in (-12345.0, 12345.0):
        return default
    return value


def _read_trace(path: Path) -> obspy.Trace | None:
    try:
        st = obspy.read(str(path), headonly=False)
    except Exception:
        return None
    if not st:
        return None
    return st[0]


def _member_sac_path(point: ThicknessPoint, wave_name: str) -> Path | None:
    candidate = Path(point.source_event_dir) / wave_name
    if candidate.exists():
        return candidate
    # 回退：仅用 basename 在源目录里找
    base = Path(wave_name).name
    candidate2 = Path(point.source_event_dir) / base
    if candidate2.exists():
        return candidate2
    return None


def _used_member_wave_names(members_file: Path) -> list[str]:
    names: list[str] = []
    try:
        with members_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if str(row.get("status") or "").strip().lower() != "used":
                    continue
                wave_name = str(row.get("wave_name") or "").strip()
                if wave_name:
                    names.append(wave_name)
    except OSError:
        return []
    return names


def _members_file_for_point(point: ThicknessPoint) -> Path | None:
    if not point.result_package_dir:
        return None
    p = Path(point.result_package_dir).expanduser().resolve() / "members.txt"
    return p if p.exists() else None


def _stack_sac_path(point: ThicknessPoint) -> Path | None:
    if not point.result_package_dir:
        return None
    p = Path(point.result_package_dir).expanduser().resolve() / "stack.sac"
    return p if p.exists() else None


def _sidecar_window(point: ThicknessPoint) -> tuple[float, float] | None:
    try:
        import json
        payload = json.loads(Path(point.sidecar_path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    window = payload.get("window")
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        return None
    try:
        x1, x2 = float(window[0]), float(window[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x1) and math.isfinite(x2)) or x2 <= x1:
        return None
    return (x1, x2)


# ---------------------------------------------------------------------------
# 成员 / 穿透点
# ---------------------------------------------------------------------------

def _pierce_records_for_point(point: ThicknessPoint) -> dict:
    try:
        path = pierce_file_path(
            point.source_event_dir, point.phase_kind, DEFAULT_PIERCE_MODEL,
            pierce_depth_km=DEFAULT_PIERCE_DEPTH_KM,
        )
        if not path.exists():
            return {}
        return load_pierce_points(path)
    except Exception:
        return {}


def load_group_members(point: ThicknessPoint) -> list[GroupMember]:
    """读 members.txt 的 used 成员，附 gcarc 与穿透点经纬度。"""
    members_file = _members_file_for_point(point)
    if members_file is None:
        return []
    wave_names = _used_member_wave_names(members_file)
    records = _pierce_records_for_point(point)
    out: list[GroupMember] = []
    for wave_name in wave_names:
        sac_path = _member_sac_path(point, wave_name)
        gcarc = float("nan")
        if sac_path is not None:
            trace = _read_trace(sac_path)
            if trace is not None:
                gcarc = _sac_float(trace, "gcarc", float("nan"))
        rec = records.get(wave_name)
        lon = float(rec.longitude) if rec else None
        lat = float(rec.latitude) if rec else None
        out.append(GroupMember(
            wave_name=wave_name, gcarc=gcarc, longitude=lon, latitude=lat,
            sac_path=sac_path,
        ))
    return out


def load_member_pierce_points(point: ThicknessPoint) -> list[tuple[float, float]]:
    """该 group used 成员的穿透点 (lon, lat) 列表，供地图下垫散点。"""
    members_file = _members_file_for_point(point)
    if members_file is None:
        return []
    wave_names = _used_member_wave_names(members_file)
    records = _pierce_records_for_point(point)
    pts: list[tuple[float, float]] = []
    for wave_name in wave_names:
        rec = records.get(wave_name)
        if rec is None:
            continue
        pts.append((float(rec.longitude), float(rec.latitude)))
    return pts


# ---------------------------------------------------------------------------
# 预览渲染
# ---------------------------------------------------------------------------

def _extract_window(data: np.ndarray, start_index: int, wavelength: int) -> np.ndarray:
    """零填充截窗，与 EvtData._extract_window_with_padding 一致。"""
    window = np.zeros(wavelength, dtype=float)
    n = len(data)
    clipped_start = max(0, start_index)
    clipped_end = min(n, start_index + wavelength)
    if clipped_end > clipped_start:
        window[: clipped_end - clipped_start] = data[clipped_start:clipped_end]
    return window


def _reference_time(trace: obspy.Trace, align_key: str, x1: float) -> float:
    """取对齐头段时刻。stack 道若缺该头段，回退 -x1（modern 帧 b=0）。"""
    ref = _sac_float(trace, align_key, float("nan"))
    if math.isfinite(ref):
        return ref
    return -x1


def build_preview_traces(
    point: ThicknessPoint,
    *,
    display_mode: str = "overlay",
    align_marker: str | None = None,
) -> PreviewBundle:
    """组装一个 group 的预览波形数据。

    display_mode: ``"overlay"``（归一化叠绘）或 ``"top"``（按 gcarc 剖面，stack 抬顶）。
    align_marker: 默认用 sidecar 的 align_marker。若强制用其它头段，仅成员可重对齐；
                  stack 道只保留了原 align 头段，无法对齐到强制头段时会省略 stack。
    """
    window = _sidecar_window(point)
    if window is None:
        return PreviewBundle(point=point, note="missing sidecar window")
    x1, x2 = window
    sidecar_align = point.align_marker or ""
    align = (align_marker or sidecar_align).strip()
    if not align:
        return PreviewBundle(point=point, note="missing align marker")
    align_key = align if align.startswith("t") else f"t{align}"

    members_file = _members_file_for_point(point)
    if members_file is None:
        return PreviewBundle(point=point, note="missing members.txt")
    member_wave_names = _used_member_wave_names(members_file)

    traces: list[PreviewTrace] = []
    member_gcarcs: list[float] = []

    # 成员道
    for wave_name in member_wave_names:
        sac_path = _member_sac_path(point, wave_name)
        if sac_path is None:
            continue
        trace = _read_trace(sac_path)
        if trace is None:
            continue
        dt = float(trace.stats.delta)
        if dt <= 0:
            continue
        b = _sac_float(trace, "b", 0.0)
        ref = _sac_float(trace, align_key, float("nan"))
        if not math.isfinite(ref):
            continue
        wavelength = int(round((x2 - x1) / dt))
        if wavelength <= 0:
            continue
        t1_index = int(round((ref - b) / dt))
        start_index = t1_index + int(round(x1 / dt))
        y = _extract_window(np.asarray(trace.data, dtype=float), start_index, wavelength)
        gcarc = _sac_float(trace, "gcarc", float("nan"))
        if math.isfinite(gcarc):
            member_gcarcs.append(gcarc)
        # 两种模式都做单道峰值归一化（与 WaveFigure.plot_waves 一致），
        # 避免某条原始振幅过大的道撑爆 TopDist 剖面的 y 轴。
        y = _normalize(y)
        t_axis = x1 + np.arange(wavelength, dtype=float) * dt
        traces.append(PreviewTrace(
            wave_name=wave_name, is_stack=False,
            t_array=t_axis, y_array=y, gcarc=gcarc, reference_t=ref,
        ))

    # stack 道
    stack_available = True
    stack_path = _stack_sac_path(point)
    stack_trace = _read_trace(stack_path) if stack_path is not None else None
    if stack_trace is not None:
        dt = float(stack_trace.stats.delta)
        if dt > 0:
            b = _sac_float(stack_trace, "b", 0.0)
            # stack 只保留了原 align 头段；若强制头段 != 原 align，stack 无法对齐 → 省略
            if align_key == (sidecar_align if sidecar_align.startswith("t") else f"t{sidecar_align}"):
                ref = _reference_time(stack_trace, align_key, x1)
            elif math.isfinite(_sac_float(stack_trace, align_key, float("nan"))):
                ref = _sac_float(stack_trace, align_key, float("nan"))
            else:
                ref = float("nan")
            if math.isfinite(ref):
                wavelength = int(round((x2 - x1) / dt))
                if wavelength > 0:
                    t1_index = int(round((ref - b) / dt))
                    start_index = t1_index + int(round(x1 / dt))
                    y = _extract_window(np.asarray(stack_trace.data, dtype=float), start_index, wavelength)
                    gcarc = _sac_float(stack_trace, "gcarc", float("nan"))
                    if display_mode == "top" and member_gcarcs:
                        max_g = max(g for g in member_gcarcs if math.isfinite(g))
                        padding = max(1.0, abs(max_g) * 0.03)
                        gcarc = max_g + padding
                    y = _normalize(y)
                    t_axis = x1 + np.arange(wavelength, dtype=float) * dt
                    traces.insert(0, PreviewTrace(
                        wave_name=point.group_name + " [stack]",
                        is_stack=True, t_array=t_axis, y_array=y,
                        gcarc=gcarc, reference_t=ref,
                    ))
            else:
                stack_available = False
        else:
            stack_available = False
    else:
        stack_available = False

    return PreviewBundle(
        point=point, traces=traces, align_marker=align,
        window=(x1, x2), display_mode=display_mode,
        stack_available=stack_available,
        note="" if stack_available else "stack trace unavailable",
    )


def _normalize(y: np.ndarray) -> np.ndarray:
    """峰值归一化到 [-1, 1]，零道保护。"""
    peak = float(np.nanmax(np.abs(y)))
    if peak <= 1e-12:
        return y
    return y / peak


# ---------------------------------------------------------------------------
# 异常评分
# ---------------------------------------------------------------------------

def compute_outlier_score(
    points: list[ThicknessPoint], *, k: int = 5,
) -> dict[str, float]:
    """每个点相对其 k 近邻厚度中位数的稳健 z-score（MAD）。

    返回 {group_key: score}。score 越大越离群。邻域按经纬度欧氏距离取 k 近邻
    （含自身），厚度偏离邻域中位数的绝对值除以邻域 MAD。
    """
    n = len(points)
    if n == 0:
        return {}
    lons = np.array([p.longitude for p in points], dtype=float)
    lats = np.array([p.latitude for p in points], dtype=float)
    thick = np.array([p.thickness_km for p in points], dtype=float)
    scores: dict[str, float] = {}
    for i, point in enumerate(points):
        d = np.sqrt((lons - lons[i]) ** 2 + (lats - lats[i]) ** 2)
        order = np.argsort(d)[: max(k, 1)]
        local = thick[order]
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med)))
        if mad <= 1e-9:
            # 邻域厚度一致：偏差完全由该点贡献时给绝对偏差，否则 0
            scores[point.group_key] = 0.0 if abs(thick[i] - med) <= 1e-9 else abs(thick[i] - med)
            continue
        scores[point.group_key] = abs(thick[i] - med) / (1.4826 * mad)
    return scores


__all__ = [
    "GroupMember",
    "PreviewTrace",
    "PreviewBundle",
    "load_group_members",
    "load_member_pierce_points",
    "build_preview_traces",
    "compute_outlier_score",
]
