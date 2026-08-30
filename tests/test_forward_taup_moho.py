"""forward 子包的单元测试。

覆盖两条正演路径的解析侧、误差传播与网格搜索。TauP 精确射线追踪需要编译速度
模型（约 0.4 s/模型），故除一条标注 slow 的一致性检验外，其余用桩替换。
"""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forward import (  # noqa: E402
    DEFAULT_CRUST_VP,
    DEFAULT_CRUST_VS,
    analytic_sensitivity,
    analytic_thickness,
    grid_search_thickness,
    thickness_uncertainty,
)


class ConstantsTests(unittest.TestCase):
    def test_defaults_match_the_paper(self):
        # Vp 依据 Allen (1966) 斯科舍海折射实测；Vs 按 Vp/Vs=1.73 导出。
        # 这两个数直接决定论文报告的厚度，改动必须同步论文。
        self.assertAlmostEqual(DEFAULT_CRUST_VP, 6.0)
        self.assertAlmostEqual(DEFAULT_CRUST_VS, 3.46)


class AnalyticPathTests(unittest.TestCase):
    def test_sensitivity_matches_paper_worked_example(self):
        # 论文方法章的算例：V=6.0、pP-pmP 组中位射线参数 p=0.057 -> 0.313 s/km。
        self.assertAlmostEqual(analytic_sensitivity(0.057), 0.3132, places=4)

    def test_thickness_is_the_exact_inverse_of_sensitivity(self):
        # H = Δt / S，两个函数必须严格互逆，否则误差传播的分母就是错的。
        p = 0.057
        for dt in (2.10, 4.00, 6.28):
            with self.subTest(dt=dt):
                expected = dt / analytic_sensitivity(p)
                self.assertAlmostEqual(analytic_thickness(dt, p), expected, places=9)

    def test_observed_traveltime_range_maps_into_reported_thickness_range(self):
        # 论文实测走时差 2.10-6.28 s，在 Vp=6.0 下应落在 7-21 km 量级。
        self.assertAlmostEqual(analytic_thickness(2.10, 0.057), 6.70, places=1)
        self.assertAlmostEqual(analytic_thickness(6.28, 0.057), 20.05, places=1)

    def test_thicker_crust_needs_proportionally_larger_delay(self):
        p = 0.057
        h1 = analytic_thickness(3.0, p)
        h2 = analytic_thickness(6.0, p)
        self.assertAlmostEqual(h2 / h1, 2.0, places=9)

    def test_sp_smp_branch_uses_both_slownesses(self):
        # sP-smP：分母是 sqrt(1/Vs^2-p^2) + sqrt(1/Vp^2-p^2)，敏感度高于 pP-pmP，
        # 故同一走时差对应更薄的地壳。
        p = 0.057
        self.assertLess(
            analytic_thickness(4.0, p, label='s'),
            analytic_thickness(4.0, p, label='p'),
        )
        expected = 4.0 / (
            math.sqrt(1.0 / DEFAULT_CRUST_VS ** 2 - p ** 2)
            + math.sqrt(1.0 / DEFAULT_CRUST_VP ** 2 - p ** 2)
        )
        self.assertAlmostEqual(analytic_thickness(4.0, p, label='s'), expected, places=9)

    def test_returns_nan_when_ray_parameter_exceeds_crustal_slowness(self):
        # p > 1/Vp 时射线在地壳内不传播，公式无解，必须返回 nan 而不是抛异常。
        self.assertTrue(math.isnan(analytic_thickness(4.0, 1.0)))
        self.assertTrue(math.isnan(analytic_sensitivity(1.0)))

    def test_higher_velocity_gives_thicker_crust(self):
        # 速度-厚度耦合：同一走时差下 Vp 越大厚度越大，约 3 km / (km/s)。
        p = 0.057
        h_low = analytic_thickness(4.0, p, vp=6.0)
        h_high = analytic_thickness(4.0, p, vp=7.0)
        self.assertGreater(h_high, h_low)
        self.assertAlmostEqual(h_high - h_low, 3.0, delta=0.6)


class UncertaintyTests(unittest.TestCase):
    def test_budget_combines_two_terms_in_quadrature(self):
        out = thickness_uncertainty(4.0, 0.057)
        self.assertAlmostEqual(
            out['sigma_total'],
            math.hypot(out['sigma_pick'], out['sigma_vp']),
            places=9,
        )

    def test_pick_term_is_pick_error_over_sensitivity(self):
        out = thickness_uncertainty(4.0, 0.057, d_pick=0.5)
        self.assertAlmostEqual(out['sigma_pick'], 0.5 / analytic_sensitivity(0.057), places=9)

    def test_two_terms_are_comparable_at_paper_settings(self):
        # 论文结论之一：拾取误差与速度不确定度贡献量级相当（1.60 / 1.43 km），
        # 故单独提高拾取精度收效有限。
        out = thickness_uncertainty(4.0, 0.057, d_pick=0.5, d_vp=0.5)
        self.assertAlmostEqual(out['sigma_pick'], 1.60, delta=0.05)
        self.assertLess(abs(out['sigma_pick'] - out['sigma_vp']), 0.5)


class GridSearchTests(unittest.TestCase):
    """网格搜索用线性正演桩替代 TauP，避免编译速度模型。"""

    @staticmethod
    def _linear_forward(slope=0.313):
        def fake(moho_km, evdp_km, gcarc_deg, label='p', vp=None, vs=None):
            return slope * moho_km
        return fake

    def _search(self, dt_observed, **kwargs):
        with patch('forward.taup_moho.taup_differential', self._linear_forward()), \
                patch('forward.taup_moho.reference_ray_parameter', return_value=0.057):
            return grid_search_thickness(dt_observed, 92.0, 55.0, **kwargs)

    def test_finds_interior_minimum(self):
        res = self._search(0.313 * 14.0)
        self.assertAlmostEqual(res.best_H, 14.0)
        self.assertFalse(res.at_edge)
        self.assertLess(abs(res.best_residual), 1e-9)

    def test_flags_solution_pinned_to_grid_edge(self):
        # 顶在边界说明搜索区间没有覆盖真解，该组结果不可信——论文用
        # "0/63 落于边界" 作为反演有效性的判据，这个标志必须可靠。
        res = self._search(0.313 * 100.0)
        self.assertTrue(res.at_edge)

    def test_confidence_interval_spans_models_within_pick_tolerance(self):
        res = self._search(0.313 * 14.0, tolerance=0.5)
        self.assertLessEqual(res.H_lo, 14.0)
        self.assertGreaterEqual(res.H_hi, 14.0)
        # 容差 0.5 s / 斜率 0.313 -> 约 ±1.6 km
        self.assertAlmostEqual(res.H_hi - res.H_lo, 3.0, delta=1.0)

    def test_reports_discrepancy_against_analytic_path(self):
        # 双路径互校：桩的斜率与解析敏感度几乎相同，两法差应远小于不确定度。
        res = self._search(0.313 * 14.0)
        self.assertFalse(math.isnan(res.analytic_H))
        self.assertLess(abs(res.discrepancy), 2.2)

    def test_nan_observation_returns_empty_result(self):
        res = self._search(math.nan)
        self.assertTrue(math.isnan(res.best_H))
        self.assertEqual(res.grid, [])


class CrossValidationTests(unittest.TestCase):
    """真跑 TauP 的双路径一致性检验——论文的方法学卖点。

    需要编译 4 个变莫霍速度模型（约 0.4 s/个，之后有缓存）。
    """

    def test_two_forward_paths_agree_within_paper_tolerance(self):
        from forward import cross_validate
        try:
            out = cross_validate()
        except Exception as exc:  # obspy taup 构建失败时跳过而非误报
            self.skipTest(f'TauP 模型构建不可用: {exc}')

        # 论文表 2 与 4.1 节：Vp=6.0、evdp=92 km、gcarc=55° 下
        # TauP 斜率 0.30327 s/km、解析斜率 0.30675 s/km、相对差异 1.13%。
        self.assertAlmostEqual(out['taup_slope_s_per_km'], 0.30327, places=4)
        self.assertAlmostEqual(out['analytic_slope_s_per_km'], 0.30675, places=4)
        self.assertAlmostEqual(out['relative_difference_percent'], 1.13, places=1)

        # 该差异必须远小于总不确定度对应的量级，否则平层近似不成立。
        self.assertLess(out['relative_difference_percent'], 5.0)

    def test_traveltime_difference_is_linear_in_moho_depth(self):
        # Δt/H 在 8-20 km 内应为常数——证明一致的是整条函数关系，
        # 而不只是某一个厚度点上的巧合。
        from forward import cross_validate
        try:
            out = cross_validate()
        except Exception as exc:
            self.skipTest(f'TauP 模型构建不可用: {exc}')

        ratios = [dt / h for h, dt in out['rows'] if dt == dt]
        self.assertEqual(len(ratios), 4)
        self.assertAlmostEqual(max(ratios) - min(ratios), 0.0, places=3)


if __name__ == '__main__':
    unittest.main()
