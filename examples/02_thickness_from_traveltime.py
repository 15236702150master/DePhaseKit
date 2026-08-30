#!/usr/bin/env python3
"""示例 2：从 pP-pmP 走时差反演地壳厚度（无需波形数据）。

演示两条正演路径给出的厚度、误差预算，以及网格搜索的边界判据。

运行：
    python examples/02_thickness_from_traveltime.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forward import (
    DEFAULT_CRUST_VP,
    analytic_sensitivity,
    analytic_thickness,
    grid_search_thickness,
    thickness_uncertainty,
)

# 一组典型观测：2011-09-03 事件（震源深度 96.3 km）某叠加分组。
EVDP_KM = 96.3
GCARC_DEG = 55.0
DT_OBSERVED = 4.20  # pP - pmP 走时差，秒


def main():
    print(f"观测：Δt(pP-pmP) = {DT_OBSERVED} s，震源深度 {EVDP_KM} km，震中距 {GCARC_DEG}°")
    print(f"地壳平均 P 波速度 Vp = {DEFAULT_CRUST_VP} km/s\n")

    # ---- 路径 A：平层解析公式 ----
    from forward import reference_ray_parameter
    p = reference_ray_parameter(EVDP_KM, GCARC_DEG)
    h_analytic = analytic_thickness(DT_OBSERVED, p)
    sens = analytic_sensitivity(p)
    print("路径 A（平层解析公式）")
    print(f"  射线参数 p     = {p:.5f} s/km")
    print(f"  敏感度 dΔt/dH  = {sens:.4f} s/km  （厚度每变 1 km，走时差变这么多）")
    print(f"  厚度 H         = {h_analytic:.2f} km\n")

    # ---- 路径 B：TauP 变莫霍精确射线追踪 + 网格搜索 ----
    print("路径 B（TauP 变莫霍精确射线追踪，网格搜索 4-30 km / 1 km）")
    res = grid_search_thickness(DT_OBSERVED, EVDP_KM, GCARC_DEG)
    print(f"  最优厚度 H     = {res.best_H:.1f} km")
    print(f"  网格残差       = {res.best_residual:+.3f} s")
    print(f"  置信区间       = {res.H_lo:.1f} - {res.H_hi:.1f} km")
    print(f"  顶在网格边界   = {'是（结果不可信！）' if res.at_edge else '否'}")
    print(f"  两法之差       = {res.discrepancy:+.2f} km\n")

    # ---- 误差预算 ----
    unc = thickness_uncertainty(DT_OBSERVED, p)
    print("误差预算（拾取误差 ±0.5 s，速度不确定度 ±0.5 km/s）")
    print(f"  拾取误差贡献   = ±{unc['sigma_pick']:.2f} km")
    print(f"  速度误差贡献   = ±{unc['sigma_vp']:.2f} km")
    print(f"  合成总误差     = ±{unc['sigma_total']:.2f} km")
    print()
    print("两项贡献量级相当，说明单纯提高拾取精度而不改善速度约束，无法显著降低总误差。")


if __name__ == "__main__":
    main()
