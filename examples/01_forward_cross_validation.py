#!/usr/bin/env python3
"""示例 1：双路径正演一致性检验（无需波形数据）。

复算论文方法学自洽性检验的核心数字：平层解析公式与 TauP 变莫霍精确射线追踪
两条独立路径的走时敏感度之差。

运行：
    python examples/01_forward_cross_validation.py

预期输出（Vp = 6.0 km/s，震源深度 92 km，震中距 55°）：
    TauP 精确射线追踪斜率   0.30327 s/km
    平层解析公式斜率        0.30675 s/km
    相对差异                1.13 %

首次运行需编译 4 个变莫霍速度模型（约 0.4 s/个），之后走缓存。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forward import DEFAULT_CRUST_VP, cross_validate


def main():
    print(f"地壳平均 P 波速度 Vp = {DEFAULT_CRUST_VP} km/s（Allen 1966 斯科舍海折射实测）")
    print("正在编译变莫霍速度模型并计算理论走时差 ...\n")

    out = cross_validate()

    print("莫霍深度 H (km)    TauP 理论 pP-pmP (s)")
    for h, dt in out["rows"]:
        print(f"{h:>10.1f}    {dt:>20.3f}")

    print()
    print(f"射线参数 p            = {out['ray_parameter_s_per_km']:.5f} s/km")
    print(f"TauP 斜率 dΔt/dH      = {out['taup_slope_s_per_km']:.5f} s/km")
    print(f"解析斜率 dΔt/dH       = {out['analytic_slope_s_per_km']:.5f} s/km")
    print(f"相对差异              = {out['relative_difference_percent']:.2f} %")
    print()
    print("该差异远小于观测总不确定度（±2.2 km），说明平层近似在本文震中距范围内成立。")
    print("注意：这检验的是平层近似的适用性，不是厚度结果本身的正确性。")


if __name__ == "__main__":
    main()
