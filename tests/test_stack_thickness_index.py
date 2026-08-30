"""stack_group_thickness_index 的单测。

构造最小假数据集（sidecar + members.txt + pierce 文件），patch 掉外部依赖
（TauP subprocess、pierce 缓存路径），验证 collect_points 与
审阅标记读写的正确性。load_pierce_points 用真实解析，覆盖文件格式链路。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stack_group_thickness_index as idx  # noqa: E402
from pierce_point_cache import PiercePointRecord  # noqa: E402


def _write_sidecar(scan_root: Path, dataset: str, event: str, payload: dict) -> Path:
    event_dir = scan_root / dataset / event
    event_dir.mkdir(parents=True, exist_ok=True)
    sidecar = event_dir / "stack_group1.sac.stack.json"
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    return sidecar


def _write_members(package_dir: Path, rows: list[tuple[str, str, str]]) -> None:
    package_dir.mkdir(parents=True, exist_ok=True)
    lines = ["wave_name\tstatus\tdetail"]
    for wave_name, status, detail in rows:
        lines.append(f"{wave_name}\t{status}\t{detail}")
    (package_dir / "members.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_pierce_file(path: Path, records: dict[str, tuple[float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# wave_name longitude latitude"]
    for wave_name, (lon, lat) in records.items():
        lines.append(f"{wave_name} {lon:.6f} {lat:.6f}")
    path.write_text("\n".join(lines), encoding="utf-8")


class CollectPointsTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.scan_root = self.tmp / "analysis"
        self.pierce_file = self.tmp / "pierce" / "pierce_points_pP_24.4km_prem.txt"
        self.pkg_dir = self.tmp / "pkg" / "stack_group1_20240101_000000_000000"

        _write_pierce_file(self.pierce_file, {
            "wave1.SAC": (-60.0, -58.0),
            "wave2.SAC": (-60.2, -58.2),
            "wave3.SAC": (-59.8, -57.8),  # 非 used，应被忽略
        })
        _write_members(self.pkg_dir, [
            ("wave1.SAC", "used", "ok"),
            ("wave2.SAC", "used", "ok"),
            ("wave3.SAC", "skipped_normalization", "x"),
        ])
        payload = {
            "stack_wave_name": "stack_group1.sac",
            "source_event_dir": str(self.tmp / "pick_jandy" / "EVT"),
            "result_package_dir": str(self.pkg_dir),
            "group_name": "group1",
            "align_marker": "t6",
            "markers": {"t6": 10.0, "t8": -2.0},  # t6+t8 → pP，时差 12s
            "geometry": {"gcarc_mean": 55.0},
            "event": {"evlo": -25.0, "evla": -60.0, "evdp": 100.0},
        }
        self.sidecar = _write_sidecar(self.scan_root, "pick_jandy", "EVT", payload)

    def _patches(self):
        return [
            patch.object(idx, "fetch_taup_ray_parameter", return_value=8.0),
            patch.object(idx, "ensure_pierce_file", lambda *a, **k: self.pierce_file),
            patch.object(idx, "pierce_file_path", lambda *a, **k: self.pierce_file),
        ]

    def test_collect_one_pP_point(self):
        with self._patches_context():
            points, colors = idx.collect_points(self.scan_root)
        self.assertEqual(len(points), 1)
        p = points[0]
        self.assertEqual(p.dataset, "pick_jandy")
        self.assertEqual(p.event, "EVT")
        self.assertEqual(p.group_name, "group1")
        self.assertEqual(p.pair_kind, "t6+t8")
        self.assertEqual(p.phase_kind, "pP")
        self.assertEqual(p.align_marker, "t6")
        self.assertEqual(p.member_count_used, 2)
        # 均值穿透点：wave1(-60,-58) 与 wave2(-60.2,-58.2) 的平均
        self.assertAlmostEqual(p.longitude, -60.1, places=4)
        self.assertAlmostEqual(p.latitude, -58.1, places=4)
        self.assertEqual(p.event_lon, -25.0)
        self.assertEqual(p.event_lat, -60.0)
        self.assertEqual(p.gcarc, 55.0)
        self.assertEqual(p.evdp, 100.0)
        # 厚度为正且有限（真实公式计算）
        self.assertTrue(p.thickness_km > 0.0 and p.thickness_km == p.thickness_km)
        # event_key / group_key
        self.assertEqual(p.event_key, "pick_jandy/EVT")
        self.assertEqual(p.group_key, "pick_jandy/EVT|group1|t6+t8")
        # 事件星色
        self.assertIn("pick_jandy/EVT", colors)

    def test_missing_pair_skipped(self):
        # 改 sidecar 为缺配对（只有 t5，无 t9）
        payload = json.loads(self.sidecar.read_text(encoding="utf-8"))
        payload["markers"] = {"t5": 10.0}
        self.sidecar.write_text(json.dumps(payload), encoding="utf-8")
        with self._patches_context():
            points, _ = idx.collect_points(self.scan_root)
        self.assertEqual(points, [])

    def test_sP_pair_branch(self):
        payload = json.loads(self.sidecar.read_text(encoding="utf-8"))
        payload["markers"] = {"t5": 12.0, "t9": 0.0}  # sP，时差 12s
        self.sidecar.write_text(json.dumps(payload), encoding="utf-8")
        with self._patches_context():
            points, _ = idx.collect_points(self.scan_root)
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].pair_kind, "t5+t9")
        self.assertEqual(points[0].phase_kind, "sP")

    def test_cache_roundtrip_invalidates_on_mtime(self):
        with self._patches_context():
            idx.build_thickness_index(self.scan_root, use_cache=True)
            p1, _ = idx.collect_points(self.scan_root)
            # 第二次应命中缓存
            idx.build_thickness_index(self.scan_root, use_cache=True)
            # 改 sidecar mtime/内容 → 缓存失效，重建
            payload = json.loads(self.sidecar.read_text(encoding="utf-8"))
            payload["group_name"] = "group2"
            self.sidecar.write_text(json.dumps(payload), encoding="utf-8")
            index2 = idx.build_thickness_index(self.scan_root, use_cache=True)
            self.assertEqual(index2.points[0].group_name, "group2")
        idx.invalidate_cache(self.scan_root)

    # 便捷：把一组 patch 对象合成 context manager
    def _patches_context(self):
        return _PatchesCtx(self._patches())


class _PatchesCtx:
    def __init__(self, patches):
        self._patches = patches

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        return False


class ReviewMarksTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.event_dir = Path(self._tmp.name) / "pick_jandy" / "EVT"
        self.meta_dir = Path(self._tmp.name) / "meta"
        # 把 stack_metadata_dir_for_event 重定向到临时 meta_dir
        self._patcher = patch.object(
            idx, "stack_metadata_dir_for_event", lambda ed: self.meta_dir
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_load_empty_when_absent(self):
        self.assertEqual(idx.load_review_marks(self.event_dir), {})

    def test_save_load_roundtrip(self):
        idx.save_review_marks(self.event_dir, {
            "group1": {"status": "suspect", "note": "thickness 35 too high"},
            "group2": {"status": "fixed"},
        })
        marks = idx.load_review_marks(self.event_dir)
        self.assertEqual(marks["group1"]["status"], "suspect")
        self.assertEqual(marks["group1"]["note"], "thickness 35 too high")
        self.assertEqual(marks["group2"]["status"], "fixed")
        self.assertEqual(marks["group2"]["note"], "")

    def test_invalid_status_normalized(self):
        idx.save_review_marks(self.event_dir, {"group1": {"status": "weird"}})
        marks = idx.load_review_marks(self.event_dir)
        self.assertEqual(marks["group1"]["status"], "pending")

    def test_set_review_mark_merges(self):
        idx.save_review_marks(self.event_dir, {"group1": {"status": "pending"}})
        idx.set_review_mark(self.event_dir, "group2", status="ignore")
        marks = idx.load_review_marks(self.event_dir)
        self.assertIn("group1", marks)
        self.assertEqual(marks["group2"]["status"], "ignore")


if __name__ == "__main__":
    unittest.main()
