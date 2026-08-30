"""stack_thickness_review_core 的单测。

构造假 stack.sac + 成员 SAC（b=0、t 头段相对秒）+ members.txt + sidecar + pierce 文件，
验证 build_preview_traces 的对齐切片（x=0 落在 align 头段）、TopDist 抬顶、Overlay 归一化、
以及 compute_outlier_score 的基本行为。
"""

from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import obspy
from obspy.core.util.attribdict import AttribDict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stack_thickness_review_core as core  # noqa: E402
import stack_group_thickness_index as idx  # noqa: E402


def _make_trace(npts: int, dt: float, sac: dict, pulse_at_sample: int | None) -> obspy.Trace:
    data = np.zeros(npts, dtype="float32")
    if pulse_at_sample is not None and 0 <= pulse_at_sample < npts:
        # 高斯脉冲
        x = np.arange(-50, 51)
        data[pulse_at_sample - 50: pulse_at_sample + 51] = np.exp(-(x ** 2) / (2 * 8 ** 2))
    tr = obspy.Trace(data=data)
    tr.stats.delta = dt
    tr.stats.starttime = obspy.UTCDateTime(2020, 1, 1)
    tr.stats.sac = AttribDict(sac)
    return tr


class PreviewTracesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.source_dir = self.tmp / "pick_jandy" / "EVT"
        self.source_dir.mkdir(parents=True)
        self.pkg_dir = self.tmp / "pkg" / "stack_group1_20240101_000000_000000"
        self.pkg_dir.mkdir(parents=True)
        self.x1, self.x2 = -55.0, 25.0
        self.dt = 0.01

        # stack.sac：t6 = -x1 = 55，脉冲放在 sample 5500（=55s）→ 对齐后 x=0
        stack_pulse_sample = int(round(55.0 / self.dt))
        stack_trace = _make_trace(
            8000, self.dt,
            {"b": 0.0, "t6": 55.0, "gcarc": 90.0, "evdp": 100.0},
            stack_pulse_sample,
        )
        stack_trace.write(str(self.pkg_dir / "stack.sac"), format="SAC")

        # 两个成员：t6=682 / 700，脉冲各自在 t6 处 → 对齐后 x=0
        self.member_names = ["PF.A.2020.001.HHZ.sac", "PF.B.2020.001.HHZ.sac"]
        for i, name in enumerate(self.member_names):
            t6 = 682.0 + i * 18.0
            pulse = int(round(t6 / self.dt))
            m = _make_trace(
                80000, self.dt,
                {"b": 0.0, "t6": t6, "gcarc": 80.0 + i * 4.0, "evdp": 100.0},
                pulse,
            )
            m.write(str(self.source_dir / name), format="SAC")

        # members.txt
        lines = ["wave_name\tstatus\tdetail"]
        for name in self.member_names:
            lines.append(f"{name}\tused\tok")
        (self.pkg_dir / "members.txt").write_text("\n".join(lines), encoding="utf-8")

        # pierce 文件 + sidecar
        self.pierce_file = self.tmp / "pierce" / "pierce_points_pP_24.4km_prem.txt"
        self.pierce_file.parent.mkdir(parents=True)
        self.pierce_file.write_text(
            f"{self.member_names[0]} -60.0 -58.0\n{self.member_names[1]} -60.2 -58.2\n",
            encoding="utf-8",
        )
        self.sidecar = self.pkg_dir.parent / "stack_group1.sac.stack.json"
        self.sidecar.write_text(json.dumps({
            "stack_wave_name": "stack_group1.sac",
            "source_event_dir": str(self.source_dir),
            "result_package_dir": str(self.pkg_dir),
            "group_name": "group1",
            "align_marker": "t6",
            "window": [self.x1, self.x2],
            "markers": {"t6": 55.0, "t8": -5.0},
            "geometry": {"gcarc_mean": 82.0},
            "event": {"evlo": -25.0, "evla": -60.0, "evdp": 100.0},
        }), encoding="utf-8")

        self.point = idx.ThicknessPoint(
            dataset="pick_jandy", event="EVT", event_label="EVT",
            group_name="group1", pair_kind="t6+t8", phase_kind="pP",
            longitude=-60.1, latitude=-58.1, thickness_km=12.0,
            event_lon=-25.0, event_lat=-60.0, align_marker="t6",
            member_count_used=2, gcarc=82.0, evdp=100.0,
            result_package_dir=str(self.pkg_dir), sidecar_path=str(self.sidecar),
            source_event_dir=str(self.source_dir),
        )
        self._patches = [
            patch.object(core, "pierce_file_path", lambda *a, **k: self.pierce_file),
        ]
        for p in self._patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in self._patches])

    def test_overlay_align_peak_at_zero(self):
        bundle = core.build_preview_traces(self.point, display_mode="overlay")
        self.assertTrue(bundle.stack_available)
        self.assertEqual(len(bundle.traces), 3)  # 1 stack + 2 members
        self.assertEqual(bundle.window, (self.x1, self.x2))
        # 每条道峰值应在 t≈0（对齐头段处）
        for tr in bundle.traces:
            idx_peak = int(np.argmax(np.abs(tr.y_array)))
            t_peak = tr.t_array[idx_peak]
            self.assertAlmostEqual(t_peak, 0.0, places=1, msg=f"{tr.wave_name} peak at {t_peak}")
        # Overlay 归一化：stack 峰值≈1
        stack_tr = [t for t in bundle.traces if t.is_stack][0]
        self.assertAlmostEqual(abs(stack_tr.y_array).max(), 1.0, places=3)

    def test_top_stack_raised_above_members(self):
        bundle = core.build_preview_traces(self.point, display_mode="top")
        stack_tr = [t for t in bundle.traces if t.is_stack][0]
        member_gcarcs = [t.gcarc for t in bundle.traces if not t.is_stack]
        self.assertTrue(math.isfinite(stack_tr.gcarc))
        self.assertGreater(stack_tr.gcarc, max(member_gcarcs))

    def test_missing_stack_reported(self):
        (self.pkg_dir / "stack.sac").unlink()
        bundle = core.build_preview_traces(self.point, display_mode="overlay")
        self.assertFalse(bundle.stack_available)
        self.assertEqual(len(bundle.traces), 2)  # 仅成员

    def test_force_align_unavailable_stack_dropped(self):
        # 强制 t7（stack 没有该头段且 != 原 align）→ stack 被省略
        bundle = core.build_preview_traces(self.point, display_mode="overlay", align_marker="t7")
        self.assertFalse(bundle.stack_available)
        # 成员也没有 t7 → 成员也被跳过
        self.assertEqual(len(bundle.traces), 0)

    def test_member_pierce_points(self):
        pts = core.load_member_pierce_points(self.point)
        self.assertEqual(len(pts), 2)
        self.assertAlmostEqual(pts[0][0], -60.0, places=4)


class OutlierScoreTests(unittest.TestCase):
    def _point(self, name, lon, lat, thick):
        return idx.ThicknessPoint(
            dataset="d", event=name, event_label=name, group_name="g",
            pair_kind="t6+t8", phase_kind="pP", longitude=lon, latitude=lat,
            thickness_km=thick, event_lon=0, event_lat=0, align_marker="t6",
            member_count_used=1, gcarc=50, evdp=100, result_package_dir="",
            sidecar_path="", source_event_dir="",
        )

    def test_uniform_thickness_zero_score(self):
        pts = [self._point(f"e{i}", float(i), 0.0, 10.0) for i in range(8)]
        scores = core.compute_outlier_score(pts, k=5)
        self.assertTrue(all(abs(s) < 1e-9 for s in scores.values()))

    def test_outlier_high_score(self):
        pts = [self._point(f"e{i}", float(i), 0.0, 10.0) for i in range(8)]
        pts.append(self._point("outlier", 4.0, 0.0, 40.0))  # 在群中央但厚度离群
        scores = core.compute_outlier_score(pts, k=5)
        self.assertGreater(scores["d/outlier|g|t6+t8"], 3.0)
        # 邻域一致的点得分低
        self.assertLess(scores["d/e0|g|t6+t8"], scores["d/outlier|g|t6+t8"])


if __name__ == "__main__":
    unittest.main()
