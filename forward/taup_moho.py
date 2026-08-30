#!/usr/bin/env python3
"""TauP 变莫霍走时正演引擎。

He et al. (2017) 用 TauP 射线参数 + 解析公式定厚度；Jia & Sun (2021) 用传播矩阵，
两者均未使用完整理论地震图（后者明确指出理论地震图"对计算量的要求较高，不适合
在波形反演中运用"）。本模块沿这条路线提供两条相互独立的正演路径并交叉验证：

  路径 A（解析）  H = Δt / (2*sqrt(1/Vp^2 - p^2))            平层近似
  路径 B（TauP）  自建变莫霍 .nd 模型，精确射线追踪求 pP-pmP  球对称分层

在 Vp = 6.0 km/s、震源深度 92 km、震中距 55° 下实测：TauP 斜率 0.30327 s/km，
解析斜率 0.30675 s/km，相对差异 1.13%。该差异远小于观测总不确定度（±2.2 km），
构成论文的"方法自洽性检验"——它验证的是平层近似的适用性，而不是厚度结果本身。

复算命令：

    python -c "from forward.taup_moho import cross_validate; print(cross_validate())"
"""
from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass, field

_DEG2KM = 111.19492664455873
_RAD2DEG = 180.0 / math.pi

# 速度与不确定度的默认值集中在 forward.constants，依据见该文件。
# 作为参照：He et al. (2017) 在安第斯用 6.21±0.22 km/s。
from .constants import (  # noqa: E402
    DEFAULT_CRUST_RHO,
    DEFAULT_CRUST_VP,
    DEFAULT_CRUST_VS,
    DEFAULT_H_MAX,
    DEFAULT_H_MIN,
    DEFAULT_H_STEP,
    DEFAULT_PICK_UNCERTAINTY,
    DEFAULT_VP_UNCERTAINTY,
)

_MODEL_CACHE: dict[tuple, object] = {}
_BUILD_DIR = os.path.join(tempfile.gettempdir(), "taup_moho_models")


def _nd_text(moho_km: float, vp: float, vs: float, rho: float) -> str:
    """生成 PREM 骨架 + 可变莫霍深度的 .nd 速度模型文本。

    地壳设为单层均匀（与 Jia & Sun 的 one-layer crust 反演一致）。
    莫霍以下沿用 PREM，保证远场射线参数正确。
    """
    return f"""0.0 {vp} {vs} {rho}
{moho_km} {vp} {vs} {rho}
mantle
{moho_km} 8.11 4.49 3.38
80.0 8.10 4.48 3.37
220.0 8.56 4.64 3.44
400.0 8.91 4.77 3.54
670.0 10.20 5.61 3.99
2891.0 13.72 7.27 5.57
outer-core
2891.0 8.06 0.0 9.90
5150.0 10.36 0.0 12.17
inner-core
5150.0 11.03 3.50 12.76
6371.0 11.26 3.67 13.09
"""


def get_moho_model(moho_km: float, vp: float = DEFAULT_CRUST_VP,
                   vs: float = DEFAULT_CRUST_VS, rho: float = DEFAULT_CRUST_RHO):
    """编译并缓存一个指定莫霍深度的 TauP 模型。实测约 0.4 s/模型。"""
    key = (round(moho_km, 3), round(vp, 3), round(vs, 3), round(rho, 3))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from obspy.taup import TauPyModel
    from obspy.taup.taup_create import build_taup_model

    os.makedirs(_BUILD_DIR, exist_ok=True)
    tag = "moho_%.3f_%.3f_%.3f" % (moho_km, vp, vs)
    tag = tag.replace(".", "p").replace("-", "m")
    nd_path = os.path.join(_BUILD_DIR, tag + ".nd")
    npz_path = os.path.join(_BUILD_DIR, tag + ".npz")

    if not os.path.exists(npz_path):
        with open(nd_path, "w") as fh:
            fh.write(_nd_text(moho_km, vp, vs, rho))
        build_taup_model(nd_path, output_folder=_BUILD_DIR, verbose=False)

    model = TauPyModel(model=npz_path)
    _MODEL_CACHE[key] = model
    return model


def _first_arrivals(arrivals) -> dict[str, float]:
    """每个震相名只取最早到时。重复分支若不去重会产生虚假离群值。"""
    out: dict[str, float] = {}
    for a in arrivals:
        if a.name not in out:
            out[a.name] = a.time
    return out


def taup_differential(moho_km: float, evdp_km: float, gcarc_deg: float,
                      label: str = "p", vp: float = DEFAULT_CRUST_VP,
                      vs: float = DEFAULT_CRUST_VS) -> float:
    """路径 B：用变莫霍模型精确射线追踪，返回 pP-pmP（或 sP-smP）走时差。"""
    parent, precursor = ("pP", "p^mP") if label == "p" else ("sP", "s^mP")
    model = get_moho_model(moho_km, vp, vs)
    arr = _first_arrivals(model.get_travel_times(
        source_depth_in_km=evdp_km, distance_in_degree=gcarc_deg,
        phase_list=[parent, precursor]))
    if parent not in arr or precursor not in arr:
        return math.nan
    return arr[parent] - arr[precursor]


def reference_ray_parameter(evdp_km: float, gcarc_deg: float, label: str = "p",
                           model_name: str = "prem") -> float:
    """取 pP/sP 的射线参数并转成 s/km（与 dephasekit 既有实现一致）。

    He et al. (2017)：假定 pmP 与 pP 射线参数相同（同 Zandt et al. 1994;
    McGlashan et al. 2008）。
    """
    from obspy.taup import TauPyModel
    key = ("ref", model_name)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = TauPyModel(model=model_name)
    phase = "pP" if label == "p" else "sP"
    arrivals = _MODEL_CACHE[key].get_travel_times(
        source_depth_in_km=evdp_km, distance_in_degree=gcarc_deg,
        phase_list=[phase])
    if not arrivals:
        return math.nan
    # ray_param 单位 s/rad
    return arrivals[0].ray_param / _RAD2DEG / _DEG2KM


def analytic_thickness(dt: float, p_s_per_km: float, label: str = "p",
                       vp: float = DEFAULT_CRUST_VP,
                       vs: float = DEFAULT_CRUST_VS) -> float:
    """路径 A：He et al. (2017) 式 —— H = Δt / (2*sqrt(1/Vp^2 - p^2))。"""
    sp = (1.0 / (vp * vp)) - p_s_per_km ** 2
    if label == "p":
        if sp <= 0.0:
            return math.nan
        return dt / (2.0 * math.sqrt(sp))
    ss = (1.0 / (vs * vs)) - p_s_per_km ** 2
    if ss <= 0.0 or sp <= 0.0:
        return math.nan
    return dt / (math.sqrt(ss) + math.sqrt(sp))


def analytic_sensitivity(p_s_per_km: float, label: str = "p",
                         vp: float = DEFAULT_CRUST_VP,
                         vs: float = DEFAULT_CRUST_VS) -> float:
    """dΔt/dH，单位 s/km。厚度误差 = 时间误差 / 该值。"""
    sp = (1.0 / (vp * vp)) - p_s_per_km ** 2
    if label == "p":
        return 2.0 * math.sqrt(sp) if sp > 0 else math.nan
    ss = (1.0 / (vs * vs)) - p_s_per_km ** 2
    if ss <= 0 or sp <= 0:
        return math.nan
    return math.sqrt(ss) + math.sqrt(sp)


def thickness_uncertainty(dt: float, p_s_per_km: float, label: str = "p",
                          vp: float = DEFAULT_CRUST_VP,
                          vs: float = DEFAULT_CRUST_VS,
                          d_pick: float = DEFAULT_PICK_UNCERTAINTY,
                          d_vp: float = DEFAULT_VP_UNCERTAINTY) -> dict:
    """误差传播：拾取误差 + 速度不确定度，独立平方和。

    He et al. (2017) 的主要误差来源同样是地壳速度偏差（6.21±0.22 km/s）。
    """
    sens = analytic_sensitivity(p_s_per_km, label, vp, vs)
    h0 = analytic_thickness(dt, p_s_per_km, label, vp, vs)
    from_pick = abs(d_pick / sens) if sens and sens == sens else math.nan
    h_lo = analytic_thickness(dt, p_s_per_km, label, vp - d_vp, vs)
    h_hi = analytic_thickness(dt, p_s_per_km, label, vp + d_vp, vs)
    from_vp = abs(h_hi - h_lo) / 2.0
    total = math.sqrt(from_pick ** 2 + from_vp ** 2)
    return {"H": h0, "sigma_pick": from_pick, "sigma_vp": from_vp,
            "sigma_total": total, "sensitivity_s_per_km": sens}


@dataclass
class GridResult:
    """一个 group 的走时正演网格搜索结果。"""
    best_H: float = math.nan
    best_residual: float = math.nan
    H_lo: float = math.nan          # 置信下界
    H_hi: float = math.nan          # 置信上界
    at_edge: bool = False           # 最优解顶在网格边界（结果不可信的信号）
    analytic_H: float = math.nan    # 路径 A 独立估计
    discrepancy: float = math.nan   # 两条路径之差
    grid: list = field(default_factory=list)


def grid_search_thickness(dt_observed: float, evdp_km: float, gcarc_deg: float,
                          label: str = "p", h_min: float = DEFAULT_H_MIN,
                          h_max: float = DEFAULT_H_MAX, h_step: float = DEFAULT_H_STEP,
                          vp: float = DEFAULT_CRUST_VP,
                          vs: float = DEFAULT_CRUST_VS,
                          tolerance: float | None = None) -> GridResult:
    """用 TauP 变莫霍模型扫 H，找使理论 Δt 最接近观测 Δt 的厚度。

    网格范围默认 4-30 km / 1 km 步长：总不确定度约 ±2.2 km，
    1 km 已细于观测分辨能力，0.5 km 步长是假精确。

    at_edge 为真表示最优解顶在搜索边界，该组结果不可信。
    """
    res = GridResult()
    if not (dt_observed == dt_observed):
        return res

    p = reference_ray_parameter(evdp_km, gcarc_deg, label)
    res.analytic_H = analytic_thickness(dt_observed, p, label, vp, vs)
    if tolerance is None:
        tolerance = DEFAULT_PICK_UNCERTAINTY

    n = int(round((h_max - h_min) / h_step)) + 1
    grid = []
    for i in range(n):
        h = h_min + i * h_step
        dt_syn = taup_differential(h, evdp_km, gcarc_deg, label, vp, vs)
        if dt_syn != dt_syn:
            continue
        grid.append((h, dt_syn, dt_syn - dt_observed))
    res.grid = grid
    if not grid:
        return res

    best = min(grid, key=lambda r: abs(r[2]))
    res.best_H, res.best_residual = best[0], best[2]
    res.at_edge = best[0] <= h_min + 1e-9 or best[0] >= h_max - 1e-9

    acceptable = [r[0] for r in grid if abs(r[2]) <= tolerance]
    if acceptable:
        res.H_lo, res.H_hi = min(acceptable), max(acceptable)
    if res.analytic_H == res.analytic_H:
        res.discrepancy = res.best_H - res.analytic_H
    return res


def cross_validate(evdp_km: float = 92.0, gcarc_deg: float = 55.0,
                   label: str = "p", h_values=(8.0, 12.0, 16.0, 20.0),
                   vp: float = DEFAULT_CRUST_VP,
                   vs: float = DEFAULT_CRUST_VS) -> dict:
    """方法自洽性检验：TauP 精确射线追踪 vs 平层解析公式的 dΔt/dH 斜率。

    默认参数下实测相对差异 1.13%，说明平层近似引入的误差远小于速度不确定度。
    """
    rows = []
    for h in h_values:
        rows.append((h, taup_differential(h, evdp_km, gcarc_deg, label, vp, vs)))
    valid = [r for r in rows if r[1] == r[1]]
    taup_slope = math.nan
    if len(valid) >= 2:
        taup_slope = (valid[-1][1] - valid[0][1]) / (valid[-1][0] - valid[0][0])
    p = reference_ray_parameter(evdp_km, gcarc_deg, label)
    exact_slope = analytic_sensitivity(p, label, vp, vs)
    rel = abs(taup_slope - exact_slope) / exact_slope * 100.0 if exact_slope else math.nan
    return {"rows": rows, "ray_parameter_s_per_km": p,
            "taup_slope_s_per_km": taup_slope,
            "analytic_slope_s_per_km": exact_slope,
            "relative_difference_percent": rel}
