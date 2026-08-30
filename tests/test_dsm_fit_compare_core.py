import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from obspy import Trace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dsm_fit_compare_core import (  # noqa: E402
    build_pairs,
    compute_cross_correlation_result,
    get_actual_phase_time,
    get_observed_manual_phase_time,
    misfit_values_from_cc,
    semantic_phase_for_alignment,
)


class DSMFitCompareCoreTests(unittest.TestCase):
    def test_manual_observed_phase_mapping_uses_picked_slots(self):
        sac = SimpleNamespace(
            t0=763.07,
            t2=787.49,
            t3=797.61,
            t5=805.145,
            t6=794.431,
            t7=769.056,
        )

        self.assertEqual(semantic_phase_for_alignment("t7"), "P")
        self.assertEqual(semantic_phase_for_alignment("t6"), "pP")
        self.assertEqual(semantic_phase_for_alignment("t5"), "sP")
        self.assertEqual(semantic_phase_for_alignment("t0"), "P")
        self.assertEqual(semantic_phase_for_alignment("t2"), "pP")
        self.assertEqual(semantic_phase_for_alignment("t3"), "sP")
        self.assertAlmostEqual(get_observed_manual_phase_time(sac, "P"), 769.056)
        self.assertAlmostEqual(get_observed_manual_phase_time(sac, "pP"), 794.431)
        self.assertAlmostEqual(get_observed_manual_phase_time(sac, "sP"), 805.145)
        self.assertAlmostEqual(get_actual_phase_time(sac, "t7"), 769.056)
        self.assertAlmostEqual(get_actual_phase_time(sac, "t0"), 769.056)

    def test_missing_actual_pick_is_not_replaced_by_theoretical_time(self):
        sac = SimpleNamespace(t0=763.07, t2=787.49, t3=797.61, t7=769.056)

        self.assertAlmostEqual(get_actual_phase_time(sac, "P"), 769.056)
        self.assertAlmostEqual(get_actual_phase_time(sac, "t0"), 769.056)
        self.assertIsNone(get_actual_phase_time(sac, "pP"))
        self.assertIsNone(get_actual_phase_time(sac, "sP"))

    def test_cross_correlation_recovers_known_shift_and_amplitude(self):
        dt = 0.05
        t = np.arange(-5.0, 5.0 + dt / 2, dt)
        observed_y = np.exp(-0.5 * (t / 0.35) ** 2)
        # Synthetic is the same pulse 0.35 s too early and half the amplitude.
        synthetic_y = 0.5 * np.exp(-0.5 * ((t + 0.35) / 0.35) ** 2)

        result = compute_cross_correlation_result(
            t,
            observed_y,
            t,
            synthetic_y,
            -4.0,
            4.0,
            tau_max_s=1.0,
        )

        self.assertAlmostEqual(result.time_shift_s, 0.35, places=6)
        self.assertGreater(result.cross_corr_max, 0.99)
        self.assertLess(result.misfit_cc, 0.01)
        self.assertAlmostEqual(result.shape_misfit, 1.0 - result.cross_corr_max ** 2)
        self.assertAlmostEqual(result.misfit_value, result.shape_misfit)
        self.assertEqual(result.misfit_mode, "shape")
        self.assertAlmostEqual(result.amplitude_factor, 2.0, delta=0.05)
        self.assertGreater(result.variance_reduction, 0.98)
        self.assertEqual(result.window_npts, 161)
        self.assertAlmostEqual(result.sample_rate_used_hz, 20.0)

    def test_misfit_modes_keep_linear_and_paper_shape_definitions(self):
        selected, linear, shape, mode = misfit_values_from_cc(0.8, "shape")

        self.assertEqual(mode, "shape")
        self.assertAlmostEqual(linear, 0.2)
        self.assertAlmostEqual(shape, 0.36)
        self.assertAlmostEqual(selected, shape)

        selected, linear, shape, mode = misfit_values_from_cc(0.8, "linear")
        self.assertEqual(mode, "linear")
        self.assertAlmostEqual(selected, linear)

    def test_cross_correlation_rejects_different_native_sampling(self):
        obs_t = np.arange(-5.0, 5.0 + 0.025 / 2, 0.025)
        syn_t = np.arange(-5.0, 5.0 + 0.05 / 2, 0.05)
        obs_y = np.sin(obs_t)
        syn_y = np.sin(syn_t)

        with self.assertRaisesRegex(ValueError, "same sample count|sampling rates differ"):
            compute_cross_correlation_result(
                obs_t,
                obs_y,
                syn_t,
                syn_y,
                -4.0,
                4.0,
                tau_max_s=1.0,
            )

    def test_build_pairs_snaps_offgrid_actual_picks_to_native_samples_for_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            obs_dir = tmp / "obs"
            syn_dir = tmp / "syn"
            obs_dir.mkdir()
            syn_dir.mkdir()
            dt = 0.25
            b = -10.0
            t = b + np.arange(121) * dt
            obs_y = np.exp(-0.5 * (t / 0.45) ** 2)
            syn_y = 0.75 * np.exp(-0.5 * ((t - 0.25) / 0.45) ** 2)

            self._write_sac(
                obs_dir / "XX.AAA.test.BHZ.sac",
                obs_y,
                dt,
                b,
                t0=0.0,
                t7=0.037,
            )
            self._write_sac(
                syn_dir / "XX.AAA.test.bhz",
                syn_y,
                dt,
                b,
                t0=0.0,
                t7=0.169,
            )

            args = self._build_args(obs_dir, syn_dir)
            pairs, skipped = build_pairs(args)

        self.assertEqual(skipped, [])
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertAlmostEqual(pair.sample_rate_used_hz, 4.0)
        self.assertEqual(pair.window_npts, 33)
        self.assertAlmostEqual(pair.time_shift_s, 0.0)
        self.assertGreater(pair.cross_corr_max, 0.99)
        self.assertAlmostEqual(pair.misfit_cc, 1.0 - pair.cross_corr_max)
        self.assertAlmostEqual(pair.shape_misfit, 1.0 - pair.cross_corr_max ** 2)
        self.assertAlmostEqual(pair.misfit_value, pair.shape_misfit)
        self.assertAlmostEqual(pair.amplitude_factor, 1.0 / 0.75, delta=0.05)
        self.assertGreater(pair.variance_reduction, 0.98)

    def test_build_pairs_skips_metrics_when_synthetic_actual_pick_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            obs_dir = tmp / "obs"
            syn_dir = tmp / "syn"
            obs_dir.mkdir()
            syn_dir.mkdir()
            dt = 0.25
            b = -10.0
            t = b + np.arange(121) * dt
            y = np.exp(-0.5 * (t / 0.45) ** 2)

            self._write_sac(
                obs_dir / "XX.AAA.test.BHZ.sac",
                y,
                dt,
                b,
                t0=0.0,
                t7=0.0,
            )
            self._write_sac(
                syn_dir / "XX.AAA.test.bhz",
                y,
                dt,
                b,
                t0=0.0,
            )

            args = self._build_args(obs_dir, syn_dir)
            pairs, skipped = build_pairs(args)

        self.assertEqual(pairs, [])
        self.assertEqual(skipped, ["XX.AAA: missing actual pick for t0 metrics"])

    def test_build_pairs_reports_target_phase_interval_residual(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            obs_dir = tmp / "obs"
            syn_dir = tmp / "syn"
            obs_dir.mkdir()
            syn_dir.mkdir()
            dt = 0.25
            b = -5.0
            t = b + np.arange(81) * dt
            y = np.exp(-0.5 * (t / 0.45) ** 2)

            self._write_sac(
                obs_dir / "XX.AAA.test.BHZ.sac",
                y,
                dt,
                b,
                t7=0.0,
                t5=32.0,
            )
            self._write_sac(
                syn_dir / "XX.AAA.test.bhz",
                y,
                dt,
                b,
                t7=0.0,
                t5=29.0,
            )

            args = self._build_args(obs_dir, syn_dir)
            args.align_phase = "t7"
            args.target_phase = "t5"
            args.use_crosscorr_align = False
            pairs, skipped = build_pairs(args)

        self.assertEqual(skipped, [])
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0].target_phase, "t5")
        self.assertAlmostEqual(pairs[0].observed_target_delta_s, 32.0)
        self.assertAlmostEqual(pairs[0].synthetic_target_delta_s, 29.0)
        self.assertAlmostEqual(pairs[0].target_delta_residual_s, 3.0)

    def _build_args(self, observed_dir: Path, synthetic_dir: Path) -> SimpleNamespace:
        return SimpleNamespace(
            observed_dir=observed_dir,
            synthetic_dir=synthetic_dir,
            align_phase="t0",
            taup_model="iasp91",
            align_source="header",
            time_min=-4.0,
            time_max=4.0,
            distance_min=None,
            distance_max=None,
            max_traces=None,
            amplitude_scale=1.0,
            observed_station_keys=None,
            normalize="separate",
            use_observed_manual_picks=True,
            target_phase=None,
            use_crosscorr_align=True,
            crosscorr_tau_max=1.0,
            misfit_mode="shape",
            sort_by="station",
            reverse_order=False,
            bandpass_freqmin=None,
            bandpass_freqmax=None,
        )

    def _write_sac(
        self,
        path: Path,
        data: np.ndarray,
        delta: float,
        begin: float,
        **headers: float,
    ) -> None:
        trace = Trace(data=np.asarray(data, dtype=np.float32))
        trace.stats.delta = float(delta)
        trace.stats.sac = {
            "b": float(begin),
            "gcarc": 80.0,
            "evdp": 100.0,
            **{key: float(value) for key, value in headers.items()},
        }
        trace.write(str(path), format="SAC")


if __name__ == "__main__":
    unittest.main()
