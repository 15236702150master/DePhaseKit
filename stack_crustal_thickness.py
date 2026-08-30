from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from pierce_point_cache import PROJECT_ROOT

# 与 calucate_xmP.py 一致：单条波形厚度计算用写死的 Vp/Vs。
SINGLE_TRACE_VP = 5.8
SINGLE_TRACE_VS = 3.2
# obspy 路径首点 ray param 单位为 s/rad，转换到 s/km 用 /57.29578/111.19492。
_RAD2DEG = 57.29578
_DEG2KM = 111.19492


DEFAULT_TAUP_BIN = PROJECT_ROOT / "opt" / "TauP-3.1.0" / "bin" / "taup"


def fetch_taup_ray_parameter(
    taup_bin: str | Path,
    *,
    evdp_km: float,
    gcarc_deg: float,
    phase: str,
    model: str = "prem",
):
    # Use --text (full table) rather than --onlytime: --onlytime prints only the
    # travel time, so the ray parameter cannot be parsed and every call would
    # raise, silently suppressing the crustal-thickness display. --text emits a
    # table whose 5th column is the ray parameter p (s/deg).
    cmd = [
        str(Path(taup_bin).expanduser().resolve()),
        "time",
        "--mod",
        str(model),
        "-h",
        f"{float(evdp_km):g}",
        "-p",
        str(phase),
        "--deg",
        f"{float(gcarc_deg):g}",
        "--text",
    ]
    completed = subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    match = re.search(r"rayParam\s+([0-9.]+)\s+s/deg", completed.stdout)
    if match:
        return float(match.group(1))
    for raw_line in completed.stdout.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line.startswith("Model:")
            or line.startswith("Distance")
            or line.startswith("(deg)")
            or set(line) <= {"-", " "}
        ):
            continue
        parts = line.split()
        # Data rows: Distance Depth Phase TravelTime RayParam ...
        if len(parts) >= 5 and parts[2] == str(phase):
            try:
                return float(parts[4])
            except ValueError:
                continue
    raise ValueError("Unable to parse TauP ray parameter")


def calculate_pp_pmp_thickness(time_diff, vp_crust, ray_param):
    # 与 calucate_xmP.py 一致：Vp/Vs 写死，射线参数单位转换用 /57.29578/111.19492。
    # vp_crust / vs_crust 参数仅为兼容旧调用，实际不使用。
    vp = 5.8
    p_s_per_km = float(ray_param) / 57.29578 / 111.19492
    term = (1.0 / vp ** 2) - p_s_per_km**2
    if term <= 0.0:
        return math.nan
    return float(time_diff) / (2.0 * math.sqrt(term))


def calculate_sp_smp_thickness(time_diff, vp_crust, vs_crust, ray_param):
    # 与 calucate_xmP.py 一致：Vp/Vs 写死，射线参数单位转换用 /57.29578/111.19492。
    # vp_crust / vs_crust 参数仅为兼容旧调用，实际不使用。
    vp = 5.8
    vs = 3.2
    p_s_per_km = float(ray_param) / 57.29578 / 111.19492
    slowness_s = (1.0 / (vs * vs)) - p_s_per_km**2
    slowness_p = (1.0 / (vp * vp)) - p_s_per_km**2
    if slowness_s <= 0.0 or slowness_p <= 0.0:
        return math.nan
    denominator = slowness_s**0.5 + slowness_p**0.5
    if denominator <= 0.0:
        return math.nan
    return float(time_diff) / denominator


# 单条波形场景：完全照搬 calucate_xmP.py 的方案。
# 用 obspy get_ray_paths_geo 取路径首点 ray param（s/rad），转 s/km；Vp/Vs 写死。
_TAUP_MODEL_CACHE: dict[str, object] = {}


def _single_trace_taup_model(model: str = "prem"):
    if model not in _TAUP_MODEL_CACHE:
        from obspy.taup import TauPyModel
        _TAUP_MODEL_CACHE[model] = TauPyModel(model=model)
    return _TAUP_MODEL_CACHE[model]


def fetch_obspy_ray_parameter(
    evdp_km: float,
    evla: float,
    evlo: float,
    stla: float,
    stlo: float,
    phase: str,
    model: str = "prem",
):
    """与 calucate_xmP.ray_paths_geo 一致：取射线路径首点 ray param (s/rad)，转 s/km。"""
    arrival = _single_trace_taup_model(model).get_ray_paths_geo(
        evdp_km, evla, evlo, stla, stlo, phase_list=[phase]
    )[0]
    return arrival.path[0][0] / _RAD2DEG / _DEG2KM


def reverse_station_coord(evla, evlo, az_deg, gcarc_deg):
    """从震源沿方位角 az 走 gcarc 度，反推台站平均经纬度（球面公式）。

    stack 场景无单台坐标，用 event 坐标 + gcarc_mean + az_mean 反推等效台站平均点。
    """
    lat1 = math.radians(evla)
    lon1 = math.radians(evlo)
    az = math.radians(az_deg)
    d = math.radians(gcarc_deg)
    lat2 = math.asin(
        math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(az)
    )
    lon2 = lon1 + math.atan2(
        math.sin(az) * math.sin(d) * math.cos(lat1),
        math.cos(d) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), math.degrees(lon2)


def calculate_single_trace_thickness(time_diff, ray_param, label):
    """与 calucate_xmP.calucate_h 一致：label='s' 为 sP-smP，'p' 为 pP-pmP。"""
    vp = SINGLE_TRACE_VP
    vs = SINGLE_TRACE_VS
    p_s_per_km = float(ray_param)  # fetch_obspy_ray_parameter 已转成 s/km
    slowness_s = (1.0 / (vs * vs)) - p_s_per_km**2
    slowness_p = (1.0 / (vp * vp)) - p_s_per_km**2
    if label == "s":
        if slowness_s <= 0.0 or slowness_p <= 0.0:
            return math.nan
        denominator = slowness_s**0.5 + slowness_p**0.5
        if denominator <= 0.0:
            return math.nan
        return float(time_diff) / denominator
    elif label == "p":
        if slowness_p <= 0.0:
            return math.nan
        return float(time_diff) / (2.0 * slowness_p**0.5)
    return math.nan
