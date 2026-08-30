#!/usr/bin/env python3
"""示例 3：读取随包附带的 20 条远震波形，并计算各道的理论 pP / pmP 到时。

examples/data/ 中是 2011-09-03 南桑威奇事件（震源深度 96.3 km）的 20 条
BHZ 记录，取自完整数据集的 266 条，用于让使用者不必下载全量数据即可跑通流程。

运行：
    python examples/03_inspect_waveforms.py
"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obspy

from forward import DEFAULT_CRUST_VP, analytic_sensitivity, reference_ray_parameter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.sac")))
    if not paths:
        print(f"未找到波形，请确认 {DATA_DIR} 存在")
        return 1

    rows = []
    for path in paths:
        trace = obspy.read(path)[0]
        sac = trace.stats.sac
        rows.append((
            trace.stats.station,
            float(sac.gcarc),
            float(sac.az),
            float(sac.evdp),
            trace.stats.npts,
            float(trace.stats.delta),
        ))
    rows.sort(key=lambda r: r[1])

    evdp = rows[0][3]
    print(f"事件：2011-09-03 南桑威奇，震源深度 {evdp:.1f} km")
    print(f"波形：{len(rows)} 条 BHZ，采样间隔 {rows[0][5]:g} s")
    print(f"震中距范围：{rows[0][1]:.2f}° - {rows[-1][1]:.2f}°\n")

    print(f"{'台站':<8}{'震中距°':>9}{'方位角°':>9}{'射线参数':>11}{'敏感度':>10}")
    print("-" * 48)
    for station, gcarc, az, _evdp, _npts, _delta in rows:
        # 震中距超出 pP 可用范围时 TauP 取不到射线，跳过。
        p = reference_ray_parameter(evdp, gcarc)
        if p != p:
            print(f"{station:<8}{gcarc:>9.2f}{az:>9.1f}{'—':>11}{'—':>10}")
            continue
        print(f"{station:<8}{gcarc:>9.2f}{az:>9.1f}{p:>11.5f}{analytic_sensitivity(p):>10.4f}")

    print()
    print(f"敏感度单位 s/km（Vp = {DEFAULT_CRUST_VP} km/s）：厚度每变化 1 km，")
    print("pP-pmP 走时差变化这么多秒。它同时决定了拾取误差折算成厚度误差的比例。")
    print()
    print("下一步：用 GUI 对这些记录分组叠加并拾取 pmP —— 见 README 的叠加子系统一节。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
