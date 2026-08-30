import math
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import obspy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from obspy import Trace
from PySide6.QtCore import Qt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pierce_point_cache import DEFAULT_OUTPUT_ROOT, PROJECT_ROOT  # noqa: E402

# Common prefix for fixture paths, derived from the project layout so the
# tests don't hardcode an absolute filesystem location.
_PROJ = str(PROJECT_ROOT)

from WaveFigure import (  # noqa: E402
    EvtData,
    MEMBER_TRACE_COLOR,
    MEMBER_TRACE_LINEWIDTH,
    STACK_TRACE_COLOR,
    STACK_TRACE_LINEWIDTH,
    WaveFigure,
    _event_name_from_dsm_path,
    _sac_float,
    _stack_member_visible_mask,
    plot_waves_with_masked_azimuth,
)
from ppk import MatplotlibWidget  # noqa: E402
from forward.constants import DEFAULT_CRUST_VP  # noqa: E402
from stack_crustal_thickness import calculate_pp_pmp_thickness  # noqa: E402
from stack_thickness_review_dialog import (  # noqa: E402
    OUTLIER_THRESHOLD,
    ThicknessReviewWindow,
    ThicknessTableModel,
)
from stack_system import (  # noqa: E402
    build_stack_workspace_index,
    build_stack_workspace_manifest,
    delete_stack_config,
    ensure_stack_workspace_dir,
    inspect_stack_event_health,
    is_stack_event_dir,
    iter_stack_sac_paths,
    load_stack_event_marker,
    load_stack_sidecar_map,
    quarantine_invalid_stack_files,
    repair_stack_event_metadata,
    resolve_stack_workspace_dir,
    stack_event_dir_for_source,
    stack_event_marker_path,
    stack_index_path,
    stack_metadata_dir_for_event,
    stack_sac_time_window,
    stack_output_dir_for_runtime,
    stack_wave_name_from_path,
    stack_wave_summary_from_sidecar,
    write_stack_workspace_index,
)
from ppk_stack import main as ppk_stack_main, resolve_launcher_config, stack_workspace_maintenance_report, stack_workspace_open_report  # noqa: E402
from window_geometry import detect_windows_workarea, parse_windows_workarea_output  # noqa: E402


class DummyLine:
    def __init__(self, xdata, ydata):
        self._xdata = np.asarray(xdata, dtype=float)
        self._ydata = np.asarray(ydata, dtype=float)

    def get_xdata(self):
        return self._xdata

    def get_ydata(self):
        return self._ydata


class DummyTrace:
    def __init__(self, data, delta=0.05, b=0.0):
        self.data = np.asarray(data, dtype=float)
        self.stats = SimpleNamespace(
            network='NET',
            station='STA',
            delta=delta,
            sac=SimpleNamespace(
                b=b,
                gcarc=10.0,
                az=20.0,
                baz=30.0,
            ),
        )

    def times(self):
        return np.arange(len(self.data), dtype=float) * float(self.stats.delta)


class DummyRect:
    def __init__(self, x, y, width, height):
        self._x = x
        self._y = y
        self._width = width
        self._height = height

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._width

    def height(self):
        return self._height

    def isValid(self):
        return True


class PreviewCurveReferenceTests(unittest.TestCase):
    def test_parse_windows_workarea_output_reads_monitor_bounds(self):
        self.assertEqual(
            parse_windows_workarea_output('WA L=1920 T=0 R=3840 B=1040\n'),
            (1920, 0, 1920, 1040),
        )

    def test_detect_windows_workarea_uses_forced_taskbar_margin(self):
        class DummyScreen:
            class DummyGeometry:
                def x(self):
                    return 1920

                def y(self):
                    return 0

                def width(self):
                    return 2560

                def height(self):
                    return 1440

                def isValid(self):
                    return True

            def geometry(self):
                return self.DummyGeometry()

        with patch.dict(os.environ, {'DPK_TASKBAR_MARGIN': '48'}, clear=False):
            self.assertEqual(
                detect_windows_workarea(screen=DummyScreen()),
                (1920, 0, 2560, 1392),
            )

    def _run_set_geom_center(self, geo_size, workarea):
        """跑一次 _set_geom_center，返回记录到的调用。

        geo_size: showMaximized() 之后窗口实际变成的 (宽, 高)。
        workarea: 屏幕工作区 (x, y, 宽, 高)。
        """
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        recorded = {}
        widget.showMaximized = lambda: recorded.__setitem__('maximized_called', True)
        widget.show = lambda: recorded.__setitem__('show_called', True)
        widget.geometry = lambda: SimpleNamespace(
            width=lambda: geo_size[0], height=lambda: geo_size[1]
        )
        wx, wy, ww, wh = workarea
        rect = SimpleNamespace(
            isValid=lambda: True, x=lambda: wx, y=lambda: wy,
            width=lambda: ww, height=lambda: wh,
        )
        with patch('ppk.screen_workarea_rect', lambda widget=None, **kw: rect), \
                patch('ppk.center_widget_on_workarea',
                      lambda w, frac=0.96: recorded.__setitem__('centered_frac', frac)):
            widget._set_geom_center()
        return recorded

    def test_main_window_startup_maximizes_without_fallback_when_wm_honours_it(self):
        # 窗管理器正常放大到工作区尺寸时，不应再走居中回退。
        recorded = self._run_set_geom_center(geo_size=(1920, 1040), workarea=(0, 0, 1920, 1040))

        self.assertTrue(recorded.get('maximized_called'))
        self.assertNotIn('centered_frac', recorded)

    def test_main_window_startup_falls_back_to_centered_geometry_when_maximize_ignored(self):
        # WSLg/XWayland 上 showMaximized 可能被忽略（窗口停在左上角默认尺寸），
        # 此时必须回退到近满屏居中几何，否则主拾取窗开出来是小窗。
        recorded = self._run_set_geom_center(geo_size=(800, 600), workarea=(0, 0, 1920, 1040))

        self.assertTrue(recorded.get('maximized_called'))
        self.assertEqual(recorded.get('centered_frac'), 0.98)
        self.assertTrue(recorded.get('show_called'))

    def test_evtdata_can_override_event_name(self):
        trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        trace.stats.network = 'DPK'
        trace.stats.station = 'STACK'
        trace.stats.delta = 0.05
        trace.stats.sac = obspy.core.AttribDict(
            b=0.0,
            gcarc=10.0,
            az=20.0,
            baz=30.0,
            nzyear=2011,
            nzjday=65,
            nzhour=14,
            nzmin=32,
            nzsec=20,
            evla=-56.0,
            evlo=-27.0,
            evdp=92.0,
        )

        evtdata = EvtData(
            [trace],
            np.asarray([0.0], dtype=float),
            x1=-1.0,
            x2=1.0,
            dt=0.05,
            event_name_override='2011_03_06_14_32_36',
        )

        self.assertEqual(evtdata.evtname, '2011_03_06_14_32_36')

    def test_dsm_group_path_uses_event_directory_name(self):
        dsm_group_path = Path(
            _PROJ,
            'data',
            'dsm',
            '2011_03_06_14_32_36',
            'group1_10',
        )

        self.assertEqual(
            _event_name_from_dsm_path(dsm_group_path),
            '2011_03_06_14_32_36',
        )
        self.assertEqual(
            _event_name_from_dsm_path(Path(_PROJ, 'data', 'pick_jandy', '2011_03_06_14_32_36')),
            '',
        )

    def test_non_stack_dsm_window_prefers_semantic_event_name(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = str(Path(_PROJ, 'data', 'dsm', '2011_03_06_14_32_36', 'group1_10'))
        figure.runtime_event_dir = figure.wavepath
        figure.stack_event_marker = {}

        self.assertEqual(figure._semantic_event_name(), '2011_03_06_14_32_36')

    def test_stack_sac_time_window_uses_sac_b_and_e(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            trace_a = Trace(data=np.zeros(3000, dtype=np.float32))
            trace_a.stats.delta = 0.02
            trace_a.stats.sac = obspy.core.AttribDict(b=0.0, e=59.98)
            trace_a.write(str(Path(temp_dir, 'stack_a.sac')), format='SAC')
            trace_b = Trace(data=np.zeros(2500, dtype=np.float32))
            trace_b.stats.delta = 0.02
            trace_b.stats.sac = obspy.core.AttribDict(b=-30.0, e=19.98)
            trace_b.write(str(Path(temp_dir, 'stack_b.sac')), format='SAC')

            self.assertEqual(stack_sac_time_window(temp_dir), [-30, 60])

    def test_preview_view_mode_label_defaults_to_wide(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_view_mode = 'wide'

        self.assertEqual(figure._preview_view_mode_label(), 'Wide')

    def test_preview_state_shortcut_keys_do_not_trigger_matplotlib_scale_keymaps(self):
        self.assertNotIn('l', plt.rcParams['keymap.yscale'])
        self.assertNotIn('L', plt.rcParams['keymap.xscale'])

    def test_preview_window_size_hint_uses_tall_geometry(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_view_mode = 'tall'

        self.assertEqual(figure._preview_window_size_hint(), (860, 1380))

    def test_preview_group_directory_uses_process_group_output(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'

        group_directory = figure._preview_group_directory()

        self.assertEqual(
            group_directory,
            _PROJ + '/data/output/process/group/pick_jandy/2011_03_06_14_32_36',
        )

    def test_preview_group_directory_uses_runtime_event_dir_in_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/stack/pick_jandy/2011_03_06_14_32_36'
        figure.runtime_event_dir = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'

        group_directory = figure._preview_group_directory()

        self.assertEqual(
            group_directory,
            _PROJ + '/data/output/process/group/pick_jandy/2011_03_06_14_32_36',
        )

    def test_pierce_output_directory_uses_runtime_event_dir_in_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/stack/pick_jandy/2011_03_06_14_32_36'
        figure.runtime_event_dir = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'
        figure.preview_pierce_phase = 'pP'
        figure.preview_pierce_model = 'iasp91'
        figure.preview_pierce_depth_km = 35.0
        figure.preview_pierce_output_root = _PROJ + '/data/output/process/pierce'

        pierce_directory = figure._pierce_output_directory()

        self.assertEqual(
            pierce_directory,
            _PROJ + '/data/output/process/pierce/pick_jandy/2011_03_06_14_32_36',
        )

    def test_list_preview_groups_reads_txt_stems(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure._preview_group_directory = lambda: temp_dir
            Path(temp_dir, 'group1.txt').write_text('waveA\n', encoding='utf-8')
            Path(temp_dir, 'group1.png').write_text('not a group list\n', encoding='utf-8')
            Path(temp_dir, 'group2.txt').write_text('waveB\n', encoding='utf-8')

            self.assertEqual(figure._list_preview_groups(), ['group1', 'group2'])

    def test_save_preview_group_excludes_selected_user1_waveforms(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure.wavepath = '/tmp/pick_jandy/event1'
            figure.preview_modes = [['7', -10.0, 10.0]]
            figure.user_markers = {
                'user1': {'waveA': math.nan, 'waveB': 1.0},
            }
            figure._preview_group_directory = lambda: temp_dir
            figure._refresh_preview_group_combo = lambda fig, selected_group=None: None
            fig = SimpleNamespace(
                _preview_state={
                    'metadata': [
                        {'wave_name': 'waveA', 'name': 'NET.A'},
                        {'wave_name': 'waveB', 'name': 'NET.B', 'is_user1_marked': True},
                    ],
                    'selected_indices': {0, 1},
                },
                savefig=lambda *args, **kwargs: None,
            )

            success, message = figure._save_preview_group(fig, 0, '1')

            saved_text = Path(temp_dir, 'group1.txt').read_text(encoding='utf-8')
            self.assertTrue(success)
            self.assertIn('Saved group1 (1 waveforms; skipped 1 user1)', message)
            self.assertIn('waveA\tNET.A', saved_text)
            self.assertNotIn('waveB', saved_text)

    def test_delete_preview_group_removes_txt_and_png(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_path = Path(temp_dir, 'group1.txt')
            png_path = Path(temp_dir, 'group1.png')
            txt_path.write_text('waveA\n', encoding='utf-8')
            png_path.write_text('image placeholder\n', encoding='utf-8')
            figure._preview_group_directory = lambda: temp_dir
            refreshed = []
            figure._refresh_preview_group_combo = lambda fig, selected_group=None: refreshed.append(selected_group)

            success, message = figure._delete_preview_group(SimpleNamespace(), 'group1')

            self.assertTrue(success)
            self.assertEqual(message, 'Deleted group1')
            self.assertFalse(txt_path.exists())
            self.assertFalse(png_path.exists())
            self.assertEqual(refreshed, [None])
            self.assertEqual(figure._list_preview_groups(), [])

    def test_preview_stack_output_directory_uses_stack_analysis_output(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'

        stack_directory = figure._preview_stack_output_directory()

        self.assertEqual(
            stack_directory,
            _PROJ + '/data/output/stack/analysis/pick_jandy/2011_03_06_14_32_36',
        )

    def test_sanitize_preview_stack_label_keeps_filename_safe_text(self):
        figure = WaveFigure.__new__(WaveFigure)

        label = figure._sanitize_preview_stack_label(' pP precursor / Test 1! ')

        self.assertEqual(label, 'pp_precursor_test_1')

    def test_preview_stack_scope_uses_visible_or_selected_waveforms(self):
        figure = WaveFigure.__new__(WaveFigure)
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA'},
                    {'wave_name': 'waveB'},
                    {'wave_name': 'waveC'},
                ],
                'selected_indices': {0, 2},
            }
        )

        visible_names, visible_error = figure._preview_stack_scope_wave_names(fig, 'visible')
        selected_names, selected_error = figure._preview_stack_scope_wave_names(fig, 'selected')

        self.assertIsNone(visible_error)
        self.assertEqual(visible_names, ['waveA', 'waveB', 'waveC'])
        self.assertIsNone(selected_error)
        self.assertEqual(selected_names, ['waveA', 'waveC'])

    def test_preview_stack_rms_normalization_and_linear_mean(self):
        figure = WaveFigure.__new__(WaveFigure)
        evtdata = SimpleNamespace(
            data=np.asarray([
                [3.0, 4.0, 0.0],
                [0.0, 6.0, 8.0],
            ], dtype=float)
        )

        stack_data, normalized_rows, valid_mask, skipped_reasons = figure._compute_preview_linear_stack(evtdata, 'rms')

        expected_first = np.asarray([3.0, 4.0, 0.0]) / math.sqrt((9.0 + 16.0) / 3.0)
        expected_second = np.asarray([0.0, 6.0, 8.0]) / math.sqrt((36.0 + 64.0) / 3.0)
        self.assertTrue(np.all(valid_mask))
        self.assertEqual(skipped_reasons, [])
        self.assertTrue(np.allclose(normalized_rows[0], expected_first))
        self.assertTrue(np.allclose(normalized_rows[1], expected_second))
        self.assertTrue(np.allclose(stack_data, np.mean([expected_first, expected_second], axis=0)))

    def test_preview_stack_peak_normalization_skips_zero_rows(self):
        figure = WaveFigure.__new__(WaveFigure)
        rows = np.asarray([
            [0.0, 0.0, 0.0],
            [2.0, -4.0, 1.0],
        ], dtype=float)

        normalized_rows, valid_mask, skipped_reasons = figure._preview_stack_normalize_rows(rows, 'peak')

        self.assertEqual(valid_mask.tolist(), [False, True])
        self.assertEqual(skipped_reasons, [(0, 'zero peak scale')])
        self.assertTrue(np.allclose(normalized_rows, [[0.5, -1.0, 0.25]]))

    def test_preview_stack_group_scope_uses_saved_group_members(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_group_paths = lambda group_name: ('group1', '/tmp/group1.txt', None)
        figure._read_preview_group_wave_names = lambda txt_path: ['waveB', 'waveC', 'waveX']
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA'},
                    {'wave_name': 'waveB'},
                    {'wave_name': 'waveC'},
                ],
                'selected_indices': set(),
            }
        )

        wave_names, error_message = figure._preview_stack_scope_wave_names(fig, 'group:group1')

        self.assertIsNone(error_message)
        self.assertEqual(wave_names, ['waveB', 'waveC'])

    def test_preview_stack_group_scope_reports_missing_alignment_members(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_group_paths = lambda group_name: ('group1', '/tmp/group1.txt', None)
        figure._read_preview_group_wave_names = lambda txt_path: ['waveX', 'waveY']
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        fig = SimpleNamespace(
            _preview_state={
                'tmarker': '5',
                'metadata': [
                    {'wave_name': 'waveA'},
                    {'wave_name': 'waveB'},
                ],
                'selected_indices': set(),
            }
        )

        wave_names, error_message = figure._preview_stack_scope_wave_names(fig, 'group:group1')

        self.assertEqual(wave_names, [])
        self.assertIn('No visible waveforms matched group1', error_message)
        self.assertIn('group1: 0/2 visible; missing 2 for t5 [waveX, waveY]', error_message)

    def test_restore_preview_group_reports_partial_visibility_summary(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_group_paths = lambda group_name: ('group1', '/tmp/group1.txt', None)
        figure._read_preview_group_wave_names = lambda txt_path: ['waveA', 'waveB', 'waveC']
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        figure._set_preview_group_input_value = WaveFigure._set_preview_group_input_value.__get__(figure, WaveFigure)
        selected_wave_names = []
        applied = []
        figure._update_preview_selection_by_wave_names = lambda fig, wave_names: selected_wave_names.extend(wave_names)
        figure._apply_preview_selection = lambda fig: applied.append(True)
        group_input = SimpleNamespace(setText=lambda value: setattr(fig, '_group_input_value', value))
        fig = SimpleNamespace(
            _preview_state={
                'tmarker': '6',
                'metadata': [
                    {'wave_name': 'waveA'},
                    {'wave_name': 'waveC'},
                ],
                'selected_indices': set(),
            }
        )
        fig._preview_controls = {'group_save_widget': group_input}

        success, message = figure._restore_preview_group(fig, 'group1')

        self.assertTrue(success)
        self.assertEqual(selected_wave_names, ['waveA', 'waveC'])
        self.assertEqual(applied, [True])
        self.assertEqual(fig._group_input_value, 'group1')
        self.assertIn('Restored group1 (2 waveforms)', message)
        self.assertIn('group1: 2/3 visible; missing 1 for t6 [waveB]', message)

    def test_preview_stack_default_scope_prefers_current_group_input(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        figure._preview_exact_group_match = WaveFigure._preview_exact_group_match.__get__(figure, WaveFigure)
        figure._list_preview_group_names_for_stack = lambda: ['group1', 'group2']
        figure._preview_stack_group_wave_names = lambda group_name: (
            group_name,
            ['waveX', 'waveY'] if group_name == 'group1' else ['waveA'],
            None,
        )
        figure.plotfig = SimpleNamespace(
            _preview_controls={
                'group_save_widget': SimpleNamespace(text=lambda: 'group1'),
                'group_combo_widget': SimpleNamespace(
                    currentData=lambda: '',
                    currentText=lambda: '',
                ),
            },
            _preview_state={
                'selected_indices': set(),
                'metadata': [{'wave_name': 'waveZ'}],
            },
        )

        scope = WaveFigure._preview_stack_default_scope(figure, 0)

        self.assertEqual(scope, 'group:group1')

    def test_preview_stack_default_scope_prefers_exact_visible_group_match(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        figure._preview_exact_group_match = WaveFigure._preview_exact_group_match.__get__(figure, WaveFigure)
        figure._list_preview_group_names_for_stack = lambda: ['group1', 'group2']
        figure._preview_stack_group_wave_names = lambda group_name: (
            group_name,
            ['waveA', 'waveB'] if group_name == 'group1' else ['waveC'],
            None,
        )
        figure.plotfig = SimpleNamespace(
            _preview_controls={
                'group_save_widget': SimpleNamespace(text=lambda: 'group9'),
                'group_combo_widget': SimpleNamespace(
                    currentData=lambda: '',
                    currentText=lambda: '',
                ),
            },
            _preview_state={
                'selected_indices': set(),
                'metadata': [{'wave_name': 'waveA'}, {'wave_name': 'waveB'}],
            },
        )

        scope = WaveFigure._preview_stack_default_scope(figure, 0)

        self.assertEqual(scope, 'group:group1')

    def test_marker_pick_is_lightweight_and_draws_without_page_refresh(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.wave = [DummyTrace(np.ones(20), delta=0.05, b=0.0)]
        figure.ipage = 0
        figure.maxidx = 5
        figure.ori_sacnames = ['wave_a.sac']
        figure.current_pick_wave_name = 'wave_a.sac'
        figure.current_pick_station_name = 'AA.BB'
        figure.preview_hidden_wave_names = set()
        figure.markers = {str(idx): {'wave_a.sac': math.nan} for idx in range(10)}
        figure.marker_styles = {str(idx): (f't{idx}', '#800080') for idx in range(10)}
        figure.key = '6'
        figure.tmarker = 't7'
        figure.tmarker_t = np.asarray([math.nan], dtype=float)
        figure.wave_raw = []
        figure.pick_mode_armed = True
        figure.stack_mode = False
        figure._event_x_to_absolute = lambda click_time, wave_index: click_time
        figure._stack_marker_display_x_value = lambda marker_time, wave_index: marker_time
        figure._set_wave_marker_time = WaveFigure._set_wave_marker_time.__get__(figure, WaveFigure)
        figure._wave_index_by_name = WaveFigure._wave_index_by_name.__get__(figure, WaveFigure)
        figure._mark_wave_markers_dirty = WaveFigure._mark_wave_markers_dirty.__get__(figure, WaveFigure)
        figure._marker_artist_gid = WaveFigure._marker_artist_gid.__get__(figure, WaveFigure)
        figure._draw_marker_artists = WaveFigure._draw_marker_artists.__get__(figure, WaveFigure)
        figure._remove_marker_artists = WaveFigure._remove_marker_artists.__get__(figure, WaveFigure)
        event = SimpleNamespace(inaxes=figure.ax1, xdata=12.345, ydata=0.2)

        changed = figure.onclick(event)

        self.assertTrue(changed)
        self.assertFalse(figure._last_click_refresh_needed)
        self.assertEqual(figure.markers['6']['wave_a.sac'], 12.345)
        marker_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [12.345, 12.345])
        ]
        self.assertEqual(len(marker_lines), 1)

    def test_crustal_text_refreshes_after_lightweight_thickness_marker_pick(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.wave = [DummyTrace(np.ones(20), delta=0.05, b=0.0)]
        figure.wave_raw = []
        figure.ipage = 0
        figure.maxidx = 5
        figure.ori_sacnames = ['wave_a.sac']
        figure.current_pick_wave_name = 'wave_a.sac'
        figure.current_pick_station_name = 'AA.BB'
        figure.preview_hidden_wave_names = set()
        figure.markers = {str(idx): {'wave_a.sac': math.nan} for idx in range(10)}
        figure.markers['6']['wave_a.sac'] = 10.0
        figure.marker_styles = {str(idx): (f't{idx}', '#800080') for idx in range(10)}
        figure.key = '6'
        figure.tmarker = 't7'
        figure.tmarker_t = np.asarray([math.nan], dtype=float)
        figure.pick_mode_armed = True
        figure.stack_mode = False
        figure._event_x_to_absolute = lambda click_time, wave_index: click_time
        figure._stack_marker_display_x_value = lambda marker_time, wave_index: marker_time
        figure._set_wave_marker_time = WaveFigure._set_wave_marker_time.__get__(figure, WaveFigure)
        figure._wave_index_by_name = WaveFigure._wave_index_by_name.__get__(figure, WaveFigure)
        figure._mark_wave_markers_dirty = WaveFigure._mark_wave_markers_dirty.__get__(figure, WaveFigure)
        figure._marker_artist_gid = WaveFigure._marker_artist_gid.__get__(figure, WaveFigure)
        figure._crustal_text_artist_gid = WaveFigure._crustal_text_artist_gid.__get__(figure, WaveFigure)
        figure._marker_affects_crustal_text = WaveFigure._marker_affects_crustal_text.__get__(figure, WaveFigure)
        figure._draw_marker_artists = WaveFigure._draw_marker_artists.__get__(figure, WaveFigure)
        figure._remove_marker_artists = WaveFigure._remove_marker_artists.__get__(figure, WaveFigure)
        figure._draw_crustal_text_artist = WaveFigure._draw_crustal_text_artist.__get__(figure, WaveFigure)
        figure._remove_crustal_text_artist = WaveFigure._remove_crustal_text_artist.__get__(figure, WaveFigure)
        figure._refresh_crustal_text_artist = WaveFigure._refresh_crustal_text_artist.__get__(figure, WaveFigure)
        figure._current_crustal_text = lambda wave_name: f"Thickness:{figure.markers['6'][wave_name]:.3f}"
        old_text = figure.ax1.text(0.01, 0.52, 'Thickness:10.000')
        old_text.set_gid(figure._crustal_text_artist_gid('wave_a.sac'))
        event = SimpleNamespace(inaxes=figure.ax1, xdata=12.345, ydata=0.2)

        changed = figure.onclick(event)

        self.assertTrue(changed)
        self.assertFalse(figure._last_click_refresh_needed)
        crustal_texts = [
            text for text in figure.ax1.texts
            if text.get_gid() == figure._crustal_text_artist_gid('wave_a.sac')
        ]
        self.assertEqual([text.get_text() for text in crustal_texts], ['Thickness:12.345'])

    def test_sac_gcarc_falls_back_to_dist_when_obspy_field_is_missing(self):
        trace = SimpleNamespace(
            stats=SimpleNamespace(
                sac=obspy.core.AttribDict({
                    'dist': 5152.4805,
                    'evla': -56.42,
                    'evlo': -27.06,
                    'stla': -75.7161,
                    'stlo': 120.226,
                })
            )
        )

        self.assertAlmostEqual(_sac_float(trace, 'gcarc'), 46.33737, places=4)

    def test_current_alignment_marker_pick_refreshes_reference_immediately(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.wave = [DummyTrace(np.ones(20), delta=0.05, b=0.0)]
        figure.wave_raw = []
        figure.ipage = 0
        figure.maxidx = 5
        figure.ori_sacnames = ['wave_a.sac']
        figure.current_pick_wave_name = 'wave_a.sac'
        figure.current_pick_station_name = 'AA.BB'
        figure.preview_hidden_wave_names = set()
        figure.markers = {str(idx): {'wave_a.sac': math.nan} for idx in range(10)}
        figure.markers['6']['wave_a.sac'] = 100.0
        figure.marker_styles = {str(idx): (f't{idx}', '#800080') for idx in range(10)}
        figure.key = '6'
        figure.tmarker = 't6'
        figure.tmarker_t = np.asarray([100.0], dtype=float)
        figure.t6 = np.asarray([100.0], dtype=float)
        figure.pick_mode_armed = True
        figure.stack_mode = False
        figure._event_x_to_absolute = lambda click_time, wave_index: click_time + 100.0
        figure._stack_marker_display_x_value = lambda marker_time, wave_index: marker_time - 100.0
        figure._is_stack_trace_align_marker = lambda wave_name, marker_key: False
        event = SimpleNamespace(inaxes=figure.ax1, xdata=12.5, ydata=0.2)

        changed = figure.onclick(event)

        self.assertTrue(changed)
        self.assertTrue(figure._last_click_refresh_needed)
        self.assertEqual(figure.tmarker_t[0], 112.5)
        self.assertEqual(figure.t6[0], 112.5)

    def test_repeated_marker_pick_replaces_existing_artist(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.wave = [DummyTrace(np.ones(20), delta=0.05, b=0.0)]
        figure.ipage = 0
        figure.maxidx = 5
        figure.ori_sacnames = ['wave_a.sac']
        figure.current_pick_wave_name = 'wave_a.sac'
        figure.current_pick_station_name = 'AA.BB'
        figure.preview_hidden_wave_names = set()
        figure.markers = {str(idx): {'wave_a.sac': math.nan} for idx in range(10)}
        figure.marker_styles = {str(idx): (f't{idx}', '#800080') for idx in range(10)}
        figure.key = '6'
        # 本用例只关心标记图元是否被替换，与对齐参考无关；取 __init__ 的默认对齐头段 t0，
        # 使 _set_wave_marker_time 里的 updates_alignment_reference 判定为假。
        figure.tmarker = 't0'
        figure.pick_mode_armed = True
        figure.stack_mode = False
        figure._event_x_to_absolute = lambda click_time, wave_index: click_time
        figure._stack_marker_display_x_value = lambda marker_time, wave_index: marker_time
        figure._set_wave_marker_time = WaveFigure._set_wave_marker_time.__get__(figure, WaveFigure)
        figure._wave_index_by_name = WaveFigure._wave_index_by_name.__get__(figure, WaveFigure)
        figure._mark_wave_markers_dirty = WaveFigure._mark_wave_markers_dirty.__get__(figure, WaveFigure)
        figure._marker_artist_gid = WaveFigure._marker_artist_gid.__get__(figure, WaveFigure)
        figure._draw_marker_artists = WaveFigure._draw_marker_artists.__get__(figure, WaveFigure)
        figure._remove_marker_artists = WaveFigure._remove_marker_artists.__get__(figure, WaveFigure)

        figure.onclick(SimpleNamespace(inaxes=figure.ax1, xdata=12.0, ydata=0.2))
        figure.key = '6'
        figure.pick_mode_armed = True
        figure.onclick(SimpleNamespace(inaxes=figure.ax1, xdata=15.0, ydata=0.2))

        old_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [12.0, 12.0])
        ]
        new_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [15.0, 15.0])
        ]
        self.assertFalse(old_lines)
        self.assertEqual(len(new_lines), 1)

    def test_preview_stack_pws_matches_linear_for_identical_rows(self):
        figure = WaveFigure.__new__(WaveFigure)
        evtdata = SimpleNamespace(
            data=np.asarray([
                [1.0, 2.0, 1.0, 0.0],
                [1.0, 2.0, 1.0, 0.0],
            ], dtype=float)
        )

        stack_data, linear_stack, phase_weights, valid_mask, skipped_reasons = figure._compute_preview_pws_stack(evtdata, 'off')

        self.assertTrue(np.all(valid_mask))
        self.assertEqual(skipped_reasons, [])
        self.assertTrue(np.allclose(linear_stack, [1.0, 2.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(phase_weights, np.ones_like(phase_weights)))
        self.assertTrue(np.allclose(stack_data, linear_stack))

    def test_preview_stack_smatstack_aligns_shifted_rows(self):
        figure = WaveFigure.__new__(WaveFigure)
        base = np.asarray([0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0])
        early = np.asarray([0.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        late = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, 0.0, 0.0])
        evtdata = SimpleNamespace(
            data=np.asarray([base, early, late], dtype=float),
            dt=1.0,
        )

        stack_data, shifted_rows, linear_stack, valid_mask, skipped_reasons, info = (
            figure._compute_preview_smatstack_stack(evtdata, 'off', max_shift_seconds=3.0)
        )

        self.assertTrue(np.all(valid_mask))
        self.assertEqual(skipped_reasons, [])
        self.assertEqual(info['max_shift_samples'], 3)
        self.assertEqual(info['shift_samples_by_input_row'], [0, 2, -1])
        self.assertTrue(np.allclose(shifted_rows, np.vstack([base, base, base])))
        self.assertTrue(np.allclose(stack_data, base))
        self.assertFalse(np.allclose(linear_stack, base))

    def test_preview_stack_moveout_seconds_uses_theory_delta_offset(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.theory_time_model = 'iasp91'
        figure._current_wave_theory_delta = lambda wave_name=None, model=None: {'pP-P': 12.0, 'sP-P': 20.0}
        figure._event_theory_delta_summary = lambda model=None: {'pP-P_mean': 10.5, 'sP-P_mean': 18.0}

        moveout_seconds, error_message = figure._preview_stack_moveout_seconds('waveA', 'phase', '7', '2')
        moveout_seconds_s, error_message_s = figure._preview_stack_moveout_seconds('waveA', 'phase', '7', '3')

        self.assertIsNone(error_message)
        self.assertAlmostEqual(moveout_seconds, 1.5)
        self.assertIsNone(error_message_s)
        self.assertAlmostEqual(moveout_seconds_s, 2.0)

    def test_init_variables_reserves_plot_slots_for_single_trace_directory(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.sta_num = 1

        figure.init_variables()

        self.assertEqual(len(figure.A1lines), 5)
        self.assertEqual(figure.A1lines, [[], [], [], [], []])

    def test_preview_stack_output_paths_use_per_run_package_directory(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_stack_output_directory = lambda: '/tmp/stack_root'
        stack_inputs = {
            'align_marker': '6',
            'x1': -40.0,
            'x2': 20.0,
            'scope': 'group:group2',
            'polarity': 'apply_user4',
            'stack_type': 'linear',
            'normalize': 'rms',
            'moveout_mode': 'off',
            'moveout_phase': '',
            'label': '',
        }

        output_paths = figure._preview_stack_output_paths(stack_inputs, '20260610_230000')

        self.assertEqual(
            output_paths['package_dir'],
            '/tmp/stack_root/stack_group2_t6_xm40_p20_20260610_230000',
        )
        self.assertEqual(output_paths['png'], output_paths['package_dir'] + '/preview.png')
        self.assertEqual(output_paths['txt'], output_paths['package_dir'] + '/stack.txt')
        self.assertEqual(output_paths['sac'], output_paths['package_dir'] + '/stack.sac')
        self.assertEqual(output_paths['json'], output_paths['package_dir'] + '/meta.json')
        self.assertEqual(output_paths['members'], output_paths['package_dir'] + '/members.txt')

    def test_stack_wave_filename_reuses_same_processing_config(self):
        figure = WaveFigure.__new__(WaveFigure)
        base_inputs = {
            'scope': 'group:group2',
            'align_marker': '6',
            'x1': -40.0,
            'x2': 20.0,
            'polarity': 'apply_user4',
            'normalize': 'rms',
            'stack_type': 'linear',
            'moveout_mode': 'off',
            'moveout_phase': '',
            'label': 'first_label',
        }

        figure._last_stack_inputs_for_filename = dict(base_inputs)
        base_name = figure._stack_wave_filename('ignored_timestamped_package_name')
        figure._last_stack_inputs_for_filename = dict(base_inputs, label='second_label')
        same_config_name = figure._stack_wave_filename('ignored_timestamped_package_name')
        figure._last_stack_inputs_for_filename = dict(base_inputs, stack_type='pws')
        pws_name = figure._stack_wave_filename('ignored_timestamped_package_name')
        figure._last_stack_inputs_for_filename = dict(base_inputs, stack_type='smatstack', smatstack_max_shift_s=5.0)
        smatstack_name = figure._stack_wave_filename('ignored_timestamped_package_name')
        figure._last_stack_inputs_for_filename = dict(base_inputs, x1=-30.0)
        different_window_name = figure._stack_wave_filename('ignored_timestamped_package_name')

        self.assertEqual(base_name, 'stack_group2_t6_xm40_p20.sac')
        self.assertEqual(same_config_name, base_name)
        self.assertEqual(pws_name, 'stack_group2_t6_xm40_p20_pws.sac')
        self.assertEqual(smatstack_name, 'stack_group2_t6_xm40_p20_smatstack_smaxp5.sac')
        self.assertEqual(different_window_name, 'stack_group2_t6_xm30_p20.sac')

    def test_run_preview_stack_success_message_includes_saved_stack_filename(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_stack_inputs = lambda fig, preview_index, options: ({
            'evtdata': SimpleNamespace(data=np.asarray([[1.0, 2.0, 3.0]], dtype=float)),
            'stack_type': 'linear',
            'normalize': 'rms',
        }, None)
        figure._compute_preview_linear_stack = lambda evtdata, normalize: (
            np.asarray([0.5, 1.0, 0.5], dtype=float),
            np.asarray([[0.5, 1.0, 0.5]], dtype=float),
            np.asarray([True], dtype=bool),
            [],
        )
        figure._save_preview_stack_outputs = lambda *args, **kwargs: {
            'wave_count_used': 27,
            'normalize': 'rms',
            'stack_type': 'linear',
            'stack_data_sac': '/tmp/stack_group3_t5_xm50_p20.sac',
        }
        figure.stack_mode = False
        figure._stack_data_event_directory = lambda: '/tmp/stack_evt'
        figure.stack_sidecars = {}

        success, message, metadata = figure._run_preview_stack(SimpleNamespace(), 0, {})

        self.assertTrue(success)
        self.assertIn('stack_group3_t5_xm50_p20.sac', message)
        self.assertEqual(metadata['wave_count_used'], 27)

    def test_run_preview_stack_smatstack_passes_shift_metadata(self):
        figure = WaveFigure.__new__(WaveFigure)
        evtdata = SimpleNamespace(
            data=np.asarray([
                [0.0, 0.0, 1.0, 2.0, 1.0, 0.0],
                [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
            ], dtype=float),
            dt=1.0,
        )
        figure._preview_stack_inputs = lambda fig, preview_index, options: ({
            'evtdata': evtdata,
            'stack_type': 'smatstack',
            'normalize': 'off',
            'smatstack_max_shift_s': 2.0,
        }, None)
        captured = {}

        def fake_save(*args, **kwargs):
            captured['stack_extra'] = kwargs.get('stack_extra')
            return {
                'wave_count_used': 2,
                'normalize': 'off',
                'stack_type': 'smatstack',
                'stack_data_sac': '/tmp/stack_group3_t5_xm50_p20_smatstack_smaxp2.sac',
            }

        figure._save_preview_stack_outputs = fake_save
        figure.stack_mode = False
        figure._stack_data_event_directory = lambda: '/tmp/stack_evt'
        figure.stack_sidecars = {}

        success, message, metadata = figure._run_preview_stack(SimpleNamespace(), 0, {})

        self.assertTrue(success)
        self.assertIn('smatstack', message)
        self.assertEqual(metadata['stack_type'], 'smatstack')
        self.assertEqual(
            captured['stack_extra']['smatstack']['shift_samples_by_input_row'],
            [0, 2],
        )

    def test_save_preview_stack_outputs_writes_members_manifest_and_package_metadata(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure.wavepath = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'
            # __init__ 里是 source_event_dir_for_runtime(wavepath)；对非 stack 的源事件目录
            # 该函数原样返回 wavepath，故此处直接取同值。
            figure.runtime_event_dir = figure.wavepath
            figure.dt = 0.05
            figure.preview_pierce_phase = 'PKIKP'
            figure.preview_pierce_model = 'iasp91'
            # 同为 __init__ 默认值；本用例 patch 掉了穿透点均值，二者只需存在即可。
            figure.preview_pierce_output_root = str(DEFAULT_OUTPUT_ROOT)
            figure.preview_pierce_cache = {}
            figure._current_bandpass_profile = lambda: {'low': 1.0, 'high': 2.0}
            figure._preview_stack_output_directory = lambda: temp_dir
            figure._stack_preview_pierce_mean = lambda wave_names: (math.nan, math.nan)
            figure._write_preview_stack_sac = (
                lambda output_path, evtdata, stack_data, x1, x2=None, align_marker='0', align_time=None:
                Path(output_path).write_bytes(b'SAC')
            )
            figure._write_stack_data_directory = (
                lambda output_paths, stack_inputs, metadata:
                str(Path(metadata['result_package_dir']) / 'stack_data.sac')
            )

            trace_a = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            trace_a.stats.dpk_wave_name = 'waveA.sac'
            trace_a.stats.sac = obspy.core.AttribDict(t7=8.0)
            trace_b = Trace(data=np.asarray([0.0, 0.0, 0.0], dtype=np.float32))
            trace_b.stats.dpk_wave_name = 'waveB.sac'
            trace_b.stats.sac = obspy.core.AttribDict(t7=math.nan)
            evtdata = SimpleNamespace(
                data=np.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=float),
                time_axis=np.asarray([-1.0, 0.0, 1.0], dtype=float),
                wave_ori=[trace_a, trace_b],
                sta_num=2,
                gcarc=np.asarray([87.0, 88.0], dtype=float),
                az=np.asarray([65.0, 67.0], dtype=float),
                baz=np.asarray([210.0, 212.0], dtype=float),
            )
            stack_inputs = {
                'evtdata': evtdata,
                'align_marker': '6',
                'x1': -1.0,
                'x2': 1.0,
                'scope': 'group:group2',
                'requested_wave_names': ['waveA.sac', 'waveB.sac', 'waveC.sac'],
                'active_wave_names': ['waveA.sac', 'waveB.sac'],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'linear',
                'label': '',
                'skipped_missing': ['waveC.sac'],
                'apply_user4_flips': True,
                'moveout_mode': 'off',
                'moveout_phase': '2',
                'moveout_applied': [],
                'moveout_skipped': [],
            }

            metadata = figure._save_preview_stack_outputs(
                stack_inputs=stack_inputs,
                stack_data=np.asarray([0.5, 1.0, 0.5], dtype=float),
                normalized_rows=np.asarray([[0.5, 1.0, 0.5]], dtype=float),
                valid_mask=np.asarray([True, False], dtype=bool),
                skipped_reasons=[(1, 'zero rms scale')],
            )

            package_dir = Path(metadata['result_package_dir'])
            self.assertTrue(package_dir.is_dir())
            self.assertEqual(package_dir.parent, Path(temp_dir))
            self.assertTrue((package_dir / 'preview.png').exists())
            self.assertTrue((package_dir / 'stack.txt').exists())
            self.assertTrue((package_dir / 'stack.sac').exists())
            self.assertTrue((package_dir / 'meta.json').exists())
            self.assertTrue((package_dir / 'members.txt').exists())

            members_text = (package_dir / 'members.txt').read_text(encoding='utf-8')
            self.assertIn('waveA.sac\tused\t', members_text)
            self.assertIn('waveB.sac\tskipped_normalization\tzero rms scale', members_text)
            self.assertIn('waveC.sac\tskipped_missing_reference\tt6', members_text)

            saved_metadata = json.loads((package_dir / 'meta.json').read_text(encoding='utf-8'))
            self.assertEqual(saved_metadata['wave_count_requested'], 3)
            self.assertEqual(saved_metadata['wave_count_input'], 2)
            self.assertEqual(saved_metadata['wave_count_used'], 1)
            self.assertEqual(saved_metadata['wave_names_requested'], ['waveA.sac', 'waveB.sac', 'waveC.sac'])
            self.assertEqual(saved_metadata['wave_names_aligned'], ['waveA.sac', 'waveB.sac'])
            self.assertEqual(saved_metadata['wave_names_used'], ['waveA.sac'])
            self.assertEqual(saved_metadata['stack_markers']['t7'], 8.0)
            self.assertEqual(saved_metadata['outputs']['members'], str(package_dir / 'members.txt'))

    def test_write_preview_stack_sac_keeps_align_marker_at_stack_relative_position(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.dt = 0.02
        trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        trace.stats.network = 'XX'
        trace.stats.station = 'AAA'
        trace.stats.delta = 0.02
        trace.stats.sampling_rate = 50.0
        trace.stats.sac = obspy.core.AttribDict(
            b=100.0,
            e=100.04,
            t0=757.1,
            t6=785.4,
            t7=763.1,
            user0=0.0,
            kstnm='AAA',
            knetwk='XX',
        )
        evtdata = SimpleNamespace(
            wave_ori=[trace],
            sta_num=27,
            reference_t=np.asarray([785.4, 786.6], dtype=float),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir, 'stack.sac')

            figure._write_preview_stack_sac(
                str(output_path),
                evtdata,
                np.asarray([0.5, -0.25, 0.75], dtype=float),
                x1=-40.0,
                align_marker='6',
            )

            saved_trace = obspy.read(str(output_path))[0]
            sac = saved_trace.stats.sac
            # Window-relative frame: b=0 at window start, align marker t6 at -x1 (=40).
            self.assertAlmostEqual(float(sac.b), 0.0, delta=1e-5)
            self.assertTrue(math.isnan(float(sac.t0)))
            self.assertAlmostEqual(float(sac.t6), 40.0, delta=1e-5)
            self.assertTrue(math.isnan(float(sac.t7)))

    def test_stack_preview_window_fallback_uses_relative_sac_window(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {'stack_a.sac': {'align_marker': 't6'}}
        trace = Trace(data=np.ones(10, dtype=np.float32))
        trace.stats.delta = 0.05
        trace.stats.sac = obspy.core.AttribDict(b=756.0, e=806.0, t6=786.0)
        figure._trace_from_runtime_dir = lambda wave_name: trace

        self.assertEqual(figure._stack_preview_window_for_wave('stack_a.sac'), (-30.0, 20.0))

    def test_preview_relative_phase_time_can_read_transient_trace_header(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {'6': {}, '7': {}}
        trace = Trace(data=np.ones(10, dtype=np.float32))
        trace.stats.sac = obspy.core.AttribDict(t6=785.0, t7=790.5)

        relative_time = figure._preview_relative_phase_time(
            '6',
            '7',
            'member_a.sac',
            reference_times={'member_a.sac': 785.0},
            trace=trace,
        )

        self.assertAlmostEqual(relative_time, 5.5)

    def test_legacy_relative_stack_preview_reanchors_to_member_reference_center(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, 'source_evt')
            stack_dir = Path(temp_dir, 'stack_evt')
            source_dir.mkdir()
            stack_dir.mkdir()

            def write_sac(path, network, station, data, gcarc, az, t6, b=0.0):
                trace = Trace(data=np.asarray(data, dtype=np.float32))
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(
                    b=b,
                    e=b + (len(data) - 1) * 0.05,
                    gcarc=gcarc,
                    az=az,
                    baz=az + 180.0,
                    t6=t6,
                )
                trace.write(str(path), format='SAC')

            write_sac(source_dir / 'member_a.sac', 'AA', 'A', np.ones(1000), 20.0, 10.0, 12.5)
            write_sac(source_dir / 'member_b.sac', 'BB', 'B', np.ones(1000), 30.0, 20.0, 15.0)
            write_sac(stack_dir / 'stack_a.sac', 'DPK', 'STACK', np.ones(1000), 25.0, 15.0, 0.0, b=0.0)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(source_dir)
            figure.stack_mode = True
            figure.stack_sidecars = {
                'stack_a.sac': {
                    'align_marker': 't6',
                    'window': [-30.0, 20.0],
                    'wave_names_used': ['member_a.sac', 'member_b.sac'],
                    'markers': {'t6': 0.0},
                }
            }
            figure.dt = 0.05
            figure.bandpass_settings = {'freqmin': None, 'freqmax': None, 'corners': 2, 'passes': 2}
            figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
            figure.preview_hidden_wave_names = set()

            waves, reference_times, _active_reference_times = figure._collect_stack_preview_stream('6', 'stack_a.sac')

            stack_trace = waves[0]
            self.assertAlmostEqual(reference_times[0], 13.75)
            self.assertAlmostEqual(float(stack_trace.stats.sac.t6), 13.75)
            self.assertAlmostEqual(float(stack_trace.stats.sac.b), -16.25)

    def test_stack_preview_anchors_stack_axis_at_window_relative_point_despite_drifted_header(self):
        # 现代帧约定：stack 数据在构造时就对齐在 -x1，所以显示锚点恒取 -x1，
        # 不采信 SAC 头段里可能已漂移的对齐值（这里头段 t6=791 就是漂移值）。
        # 这条约定用来切断标记漂移的自传播，见 _collect_stack_preview_stream 内注释。
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, 'source_evt')
            stack_dir = Path(temp_dir, 'stack_evt')
            source_dir.mkdir()
            stack_dir.mkdir()

            def write_sac(path, network, station, data, gcarc, az, t6, b=0.0):
                trace = Trace(data=np.asarray(data, dtype=np.float32))
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(
                    b=b,
                    e=b + (len(data) - 1) * 0.05,
                    gcarc=gcarc,
                    az=az,
                    baz=az + 180.0,
                    t6=t6,
                )
                trace.write(str(path), format='SAC')

            stack_t6 = 791.0
            write_sac(source_dir / 'member_a.sac', 'AA', 'A', np.ones(1000), 20.0, 10.0, 790.0)
            write_sac(source_dir / 'member_b.sac', 'BB', 'B', np.ones(1000), 30.0, 20.0, 792.0)
            write_sac(stack_dir / 'stack_a.sac', 'DPK', 'STACK', np.ones(1000), 25.0, 15.0, stack_t6, b=0.0)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(source_dir)
            figure.stack_mode = True
            figure.stack_sidecars = {
                'stack_a.sac': {
                    'align_marker': 't6',
                    'window': [-30.0, 20.0],
                    'wave_names_used': ['member_a.sac', 'member_b.sac'],
                    'markers': {'t6': stack_t6},
                }
            }
            figure.dt = 0.05
            figure.bandpass_settings = {'freqmin': None, 'freqmax': None, 'corners': 2, 'passes': 2}
            figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
            figure.preview_hidden_wave_names = set()

            waves, reference_times, _active_reference_times = figure._collect_stack_preview_stream('6', 'stack_a.sac')

            stack_trace = waves[0]
            window_start = -30.0
            self.assertAlmostEqual(reference_times[0], -window_start)
            # 成员道仍保持各自的绝对到时，不被 stack 的相对帧带偏。
            self.assertAlmostEqual(reference_times[1], 790.0)
            self.assertAlmostEqual(reference_times[2], 792.0)
            # 锚点已落在道内，无需重锚，b/e 保持原值；头段 t6 也不被改写。
            self.assertAlmostEqual(float(stack_trace.stats.sac.b), 0.0)
            self.assertAlmostEqual(float(stack_trace.stats.sac.e), 999 * 0.05, places=4)
            self.assertAlmostEqual(float(stack_trace.stats.sac.t6), stack_t6)

    def test_stack_preview_reanchors_legacy_window_relative_stack_to_member_center(self):
        # 旧格式 stack SAC：对齐头段写成 0、b/e 就是窗口本身([-30,20])。
        # 这类道要重锚到成员对齐点的中心，否则会和成员道差一整个绝对到时。
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, 'source_evt')
            stack_dir = Path(temp_dir, 'stack_evt')
            source_dir.mkdir()
            stack_dir.mkdir()

            def write_sac(path, network, station, data, gcarc, az, t6, b=0.0):
                trace = Trace(data=np.asarray(data, dtype=np.float32))
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(
                    b=b,
                    e=b + (len(data) - 1) * 0.05,
                    gcarc=gcarc,
                    az=az,
                    baz=az + 180.0,
                    t6=t6,
                )
                trace.write(str(path), format='SAC')

            write_sac(source_dir / 'member_a.sac', 'AA', 'A', np.ones(1000), 20.0, 10.0, 790.0)
            write_sac(source_dir / 'member_b.sac', 'BB', 'B', np.ones(1000), 30.0, 20.0, 792.0)
            # 旧格式：t6=0、b=-30（窗口起点）
            write_sac(stack_dir / 'stack_a.sac', 'DPK', 'STACK', np.ones(1001), 25.0, 15.0, 0.0, b=-30.0)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(source_dir)
            figure.stack_mode = True
            figure.stack_sidecars = {
                'stack_a.sac': {
                    'align_marker': 't6',
                    'window': [-30.0, 20.0],
                    'wave_names_used': ['member_a.sac', 'member_b.sac'],
                    'markers': {'t6': 0.0},
                }
            }
            figure.dt = 0.05
            figure.bandpass_settings = {'freqmin': None, 'freqmax': None, 'corners': 2, 'passes': 2}
            figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
            figure.preview_hidden_wave_names = set()

            _waves, reference_times, _active = figure._collect_stack_preview_stream('6', 'stack_a.sac')

            # 成员对齐点中心 = (790 + 792) / 2
            self.assertAlmostEqual(reference_times[0], 791.0)

    def _build_member_marker_fixture(self, temp_dir):
        source_dir = Path(temp_dir, 'source_evt')
        stack_dir = Path(temp_dir, 'stack_evt')
        source_dir.mkdir()
        stack_dir.mkdir()

        def write_sac(path, network, station, data, gcarc, az, t6, b=0.0):
            trace = Trace(data=np.asarray(data, dtype=np.float32))
            trace.stats.network = network
            trace.stats.station = station
            trace.stats.delta = 0.05
            trace.stats.sac = obspy.core.AttribDict(
                b=b,
                e=b + (len(data) - 1) * 0.05,
                gcarc=gcarc,
                az=az,
                baz=az + 180.0,
                t6=t6,
            )
            trace.write(str(path), format='SAC')

        write_sac(source_dir / 'member_a.sac', 'AA', 'A', np.ones(1000), 20.0, 10.0, 790.0)
        write_sac(source_dir / 'member_b.sac', 'BB', 'B', np.ones(1000), 30.0, 20.0, 792.0)
        stack_t6 = 791.0
        write_sac(stack_dir / 'stack_a.sac', 'DPK', 'STACK', np.ones(1000), 25.0, 15.0, stack_t6, b=0.0)

        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = str(stack_dir)
        figure.runtime_event_dir = str(source_dir)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack_a.sac': {
                'align_marker': 't6',
                'window': [-30.0, 20.0],
                'wave_names_used': ['member_a.sac', 'member_b.sac'],
                'markers': {'t6': stack_t6},
            }
        }
        figure.dt = 0.05
        figure.bandpass_settings = {'freqmin': None, 'freqmax': None, 'corners': 2, 'passes': 2}
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure.preview_hidden_wave_names = set()
        figure.marker_styles = {str(i): None for i in range(10)}
        figure.markers = {str(i): {} for i in range(10)}
        figure.stack_manual_marker_keys = set()
        figure.ori_sacnames = ['stack_a.sac']
        return figure, stack_t6

    def test_stack_preview_member_marker_edit_reflects_without_restart(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figure, _stack_t6 = self._build_member_marker_fixture(temp_dir)

            _, _, active = figure._collect_stack_preview_stream('6', 'stack_a.sac')
            # First load seeds member t6 from the source SAC header.
            self.assertAlmostEqual(active['member_a.sac'], 790.0)

            # Simulate an in-session member marker edit (no disk write, no restart).
            figure._set_wave_marker_time('member_a.sac', '6', 845.0)

            _, _, active_after = figure._collect_stack_preview_stream('6', 'stack_a.sac')
            self.assertAlmostEqual(active_after['member_a.sac'], 845.0)
            # member_b is untouched and still reads its source value.
            self.assertAlmostEqual(active_after['member_b.sac'], 792.0)

    def test_stack_preview_member_marker_write_is_deferred_until_flush(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figure, _stack_t6 = self._build_member_marker_fixture(temp_dir)
            source_path = Path(figure.runtime_event_dir) / 'member_a.sac'

            self.assertAlmostEqual(obspy.read(str(source_path))[0].stats.sac.t6, 790.0)

            figure._set_wave_marker_time('member_a.sac', '6', 845.0)

            self.assertAlmostEqual(obspy.read(str(source_path))[0].stats.sac.t6, 790.0)
            self.assertEqual(getattr(figure, '_pending_source_marker_writes'), {'member_a.sac': {'t6'}})

            written = figure._flush_pending_source_marker_writes(notify_review=False)

            self.assertEqual(written, ['member_a.sac'])
            self.assertAlmostEqual(obspy.read(str(source_path))[0].stats.sac.t6, 845.0)

    def test_stack_trace_align_marker_is_read_only(self):
        # 对齐头段在 stack 道上是结构性的（数据已按它叠好，值恒为 -x1），
        # 允许改会让显示与数据脱节，要重新对齐只能重新叠加。
        with tempfile.TemporaryDirectory() as temp_dir:
            figure, _stack_t6 = self._build_member_marker_fixture(temp_dir)
            window_align_point = 30.0  # = -window[0]

            _, reference_times, _ = figure._collect_stack_preview_stream('6', 'stack_a.sac')
            self.assertAlmostEqual(reference_times[0], window_align_point)
            self.assertTrue(figure._is_stack_trace_align_marker('stack_a.sac', '6'))

            accepted = figure._set_wave_marker_time('stack_a.sac', '6', 820.0)

            self.assertFalse(accepted)
            _, reference_times_after, _ = figure._collect_stack_preview_stream('6', 'stack_a.sac')
            self.assertAlmostEqual(reference_times_after[0], window_align_point)

    def test_stack_trace_non_align_marker_stays_editable(self):
        # 只读只针对对齐头段本身；其余震相标记仍应能在 stack 道上正常拾取。
        with tempfile.TemporaryDirectory() as temp_dir:
            figure, _stack_t6 = self._build_member_marker_fixture(temp_dir)

            self.assertFalse(figure._is_stack_trace_align_marker('stack_a.sac', '8'))
            accepted = figure._set_wave_marker_time('stack_a.sac', '8', 26.0)

            self.assertTrue(accepted)
            self.assertAlmostEqual(figure.markers['8']['stack_a.sac'], 26.0)

    def test_stack_preview_member_marker_seed_does_not_overwrite_manual_edit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            figure, _stack_t6 = self._build_member_marker_fixture(temp_dir)

            figure._set_wave_marker_time('member_a.sac', '7', 12.0)
            # Repeated collects / marker switches must not reseed over the manual value.
            figure._collect_stack_preview_stream('7', 'stack_a.sac')
            figure._collect_stack_preview_stream('6', 'stack_a.sac')
            figure._collect_stack_preview_stream('7', 'stack_a.sac')

            self.assertAlmostEqual(figure.markers['7']['member_a.sac'], 12.0)

    def test_set_wave_marker_time_accepts_member_name_in_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.markers = {str(i): {} for i in range(10)}
        figure.stack_manual_marker_keys = set()
        figure.marker_styles = {str(i): None for i in range(10)}
        figure.ori_sacnames = ['stack_a.sac']

        accepted = figure._set_wave_marker_time('member_a.sac', '6', 1.0)
        self.assertTrue(accepted)
        self.assertAlmostEqual(figure.markers['6']['member_a.sac'], 1.0)

    def test_write_stack_data_directory_copies_sac_and_sidecar(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_root = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            output_root = Path(temp_dir, 'data', 'output', 'process', 'stack', 'pick_jandy', 'evt1')
            package_dir = output_root / 'stack_group2_t6_xm40_p20_20260610_230000'
            package_dir.mkdir(parents=True, exist_ok=True)
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(package_dir / 'stack.sac'), format='SAC')

            figure.runtime_event_dir = _PROJ + '/data/pick_jandy/evt1'
            figure._stack_data_event_directory = lambda: str(stack_root)
            metadata = {
                'result_package_dir': str(package_dir),
                'stack_type': 'linear',
                'normalize': 'rms',
                'scope': 'group:group2',
                'align_marker': 't6',
                'x1': -40.0,
                'x2': 20.0,
                'wave_count_requested': 27,
                'wave_count_input': 3,
                'wave_count_used': 24,
                'wave_names_requested': ['a.sac', 'b.sac', 'c.sac'],
                'wave_names_aligned': ['a.sac', 'b.sac'],
                'wave_names_used': ['a.sac'],
                'skipped_missing_reference': ['c.sac'],
                'skipped_normalization': [{'wave_name': 'b.sac', 'reason': 'zero rms scale'}],
                'moveout_mode': 'simple_theory',
                'moveout_phase': 't2',
                'moveout_applied': [{'wave_name': 'a.sac', 'seconds': 0.4}],
                'moveout_skipped': [{'wave_name': 'd.sac', 'reason': 'missing delta'}],
                'gcarc_mean': 87.4,
                'az_mean': 65.2,
                'baz_mean': 211.0,
                'pierce_lon_mean': -27.0,
                'pierce_lat_mean': math.nan,
                'event_info': {'nzyear': 2011, 'nzjday': 65, 'nzhour': 14, 'nzmin': 32, 'nzsec': 36, 'evla': -56.39, 'evlo': -27.03, 'evdp': 92.0},
                'stack_markers': {'t0': math.nan, 't6': np.float32(13.75), 't7': 8.0},
                'outputs': {'json': str(package_dir / 'meta.json')},
            }
            output_paths = {
                'output_root': str(output_root),
                'basename_tag': 'stack_group2_t6_xm40_p20_20260610_230000',
                'sac': str(package_dir / 'stack.sac'),
            }
            stack_inputs = {
                'scope': 'group:group2',
                'align_marker': '6',
                'x1': -40.0,
                'x2': 20.0,
                'polarity': 'apply_user4',
                'stack_type': 'linear',
                'normalize': 'rms',
                'moveout_mode': 'off',
                'moveout_phase': '',
                'label': '',
            }
            figure.markers = {str(idx): {} for idx in range(10)}
            figure.markers['7']['a.sac'] = 8.0
            (package_dir / 'meta.json').write_text(
                json.dumps(
                    {
                        'wave_names_requested': ['a.sac', 'b.sac', 'c.sac'],
                        'wave_names_aligned': ['a.sac', 'b.sac'],
                        'wave_names_used': ['a.sac'],
                        'skipped_missing_reference': ['c.sac'],
                        'skipped_normalization': [{'wave_name': 'b.sac', 'reason': 'zero rms scale'}],
                        'moveout_applied': [{'wave_name': 'a.sac', 'seconds': 0.4}],
                        'moveout_skipped': [{'wave_name': 'd.sac', 'reason': 'missing delta'}],
                        'outputs': {'json': str(package_dir / 'meta.json')},
                    }
                ),
                encoding='utf-8',
            )

            saved_path = figure._write_stack_data_directory(output_paths, stack_inputs, metadata)

            saved_stack = stack_root / 'stack_group2_t6_xm40_p20.sac'
            self.assertEqual(saved_path, str(saved_stack))
            self.assertTrue(saved_stack.exists())
            saved_trace = obspy.read(str(saved_stack), headonly=True)[0]
            self.assertAlmostEqual(float(saved_trace.stats.sac.b), 0.0)
            self.assertAlmostEqual(float(saved_trace.stats.sac.e), 0.1)
            self.assertTrue(math.isnan(getattr(saved_trace.stats.sac, 't5', math.nan)))
            self.assertAlmostEqual(float(saved_trace.stats.sac.t6), 40.0)
            self.assertAlmostEqual(float(saved_trace.stats.sac.t7), 34.25)
            self.assertTrue(stack_event_marker_path(stack_root).exists())
            self.assertFalse((stack_root / '.stack_event.json').exists())
            sidecars = load_stack_sidecar_map(stack_root)
            saved_wave_name = stack_wave_name_from_path(stack_root, saved_stack)
            self.assertEqual(saved_wave_name, 'stack_group2_t6_xm40_p20.sac')
            self.assertIn(saved_wave_name, sidecars)
            self.assertEqual(sidecars[saved_wave_name]['geometry']['gcarc_mean'], 87.4)
            self.assertEqual(sidecars[saved_wave_name]['markers']['t6'], 40.0)
            self.assertEqual(sidecars[saved_wave_name]['markers']['t7'], 34.25)
            self.assertEqual(sidecars[saved_wave_name]['group_name'], 'group2')
            self.assertEqual(sidecars[saved_wave_name]['wave_names_requested'], ['a.sac', 'b.sac', 'c.sac'])
            self.assertEqual(sidecars[saved_wave_name]['wave_names_aligned'], ['a.sac', 'b.sac'])
            self.assertEqual(sidecars[saved_wave_name]['wave_names_used'], ['a.sac'])
            self.assertEqual(sidecars[saved_wave_name]['skipped_missing_reference'], ['c.sac'])
            self.assertEqual(sidecars[saved_wave_name]['skipped_normalization'][0]['wave_name'], 'b.sac')
            self.assertEqual(sidecars[saved_wave_name]['moveout_applied'][0]['seconds'], 0.4)
            self.assertEqual(sidecars[saved_wave_name]['moveout_skipped'][0]['reason'], 'missing delta')
            raw_sidecar = json.loads((stack_metadata_dir_for_event(stack_root) / 'stack_group2_t6_xm40_p20.stack.json').read_text(encoding='utf-8'))
            self.assertEqual(raw_sidecar['stack_wave_name'], saved_wave_name)
            self.assertEqual(raw_sidecar['group_name'], 'group2')
            self.assertIsNone(raw_sidecar['geometry']['pierce_lat_mean'])
            self.assertEqual(raw_sidecar['markers']['t7'], 34.25)
            self.assertEqual(set(raw_sidecar['markers'].keys()), {f't{idx}' for idx in range(10)})
            self.assertEqual(raw_sidecar['markers']['t6'], 40.0)
            self.assertTrue(all(raw_sidecar['markers'][f't{idx}'] is None for idx in range(10) if idx not in (6, 7)))
            self.assertEqual(set(raw_sidecar['user_markers'].keys()), {'user1', 'user2', 'user3', 'user4', 'user5'})
            self.assertIsNone(raw_sidecar['user_markers']['user1'])
            self.assertNotIn('wave_names_requested', raw_sidecar)
            self.assertNotIn('wave_names_aligned', raw_sidecar)
            self.assertNotIn('wave_names_used', raw_sidecar)
            self.assertNotIn('skipped_missing_reference', raw_sidecar)
            self.assertNotIn('skipped_normalization', raw_sidecar)
            self.assertEqual(raw_sidecar['moveout_mode'], 'simple_theory')
            self.assertNotIn('moveout_applied', raw_sidecar)
            self.assertNotIn('moveout_skipped', raw_sidecar)
            index = json.loads(stack_index_path(stack_root).read_text(encoding='utf-8'))
            self.assertEqual(index['stack_count'], 1)
            self.assertEqual(index['valid_stack_count'], 1)
            self.assertEqual(index['stacks'][0]['wave_name'], saved_wave_name)
            self.assertEqual(index['health']['sidecars_needing_repair'], [])

    def test_delete_stack_config_removes_sac_sidecar_and_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_root = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_root.mkdir(parents=True, exist_ok=True)
            (stack_root / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            package_dir = Path(temp_dir, 'analysis', 'stack_group1_old')
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / 'meta.json').write_text(json.dumps({'wave_names_used': ['old.sac']}), encoding='utf-8')
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_root / 'stack_group1.sac'), format='SAC')
            (stack_root / 'stack_group1.stack.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'stack_wave_name': 'stack_group1.sac',
                        'group_name': 'group1',
                        'scope': 'group:group1',
                        'result_package_dir': str(package_dir),
                    }
                ),
                encoding='utf-8',
            )

            report = delete_stack_config(stack_root, 'stack_group1.sac')

            self.assertTrue(report['removed'])
            self.assertFalse((stack_root / 'stack_group1.sac').exists())
            self.assertFalse((stack_root / 'stack_group1.stack.json').exists())
            self.assertFalse(package_dir.exists())
            index = json.loads(stack_index_path(stack_root).read_text(encoding='utf-8'))
            self.assertEqual(index['stack_count'], 0)

    def test_restack_group_replaces_existing_group_artifacts(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_root = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_root.mkdir(parents=True, exist_ok=True)
            (stack_root / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            old_package = Path(temp_dir, 'analysis', 'stack_old_group1_package')
            old_package.mkdir(parents=True, exist_ok=True)
            (old_package / 'meta.json').write_text(json.dumps({'wave_names_used': ['new_member.sac']}), encoding='utf-8')
            old_trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            old_trace.stats.delta = 0.05
            old_trace.write(str(stack_root / 'stack_group1_old.sac'), format='SAC')
            (stack_root / 'stack_group1_old.stack.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'stack_wave_name': 'stack_group1_old.sac',
                        'group_name': 'group1',
                        'scope': 'group:group1',
                        'result_package_dir': str(old_package),
                        'wave_names_used': ['new_member.sac'],
                    }
                ),
                encoding='utf-8',
            )

            new_package = Path(temp_dir, 'analysis', 'stack_group1_t6_xm40_p20_20260610_230000')
            new_package.mkdir(parents=True, exist_ok=True)
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(new_package / 'stack.sac'), format='SAC')
            (new_package / 'meta.json').write_text(json.dumps({'wave_names_used': ['new_member.sac']}), encoding='utf-8')

            figure.runtime_event_dir = _PROJ + '/data/pick_jandy/evt1'
            figure._stack_data_event_directory = lambda: str(stack_root)
            output_paths = {
                'output_root': str(new_package.parent),
                'basename_tag': new_package.name,
                'sac': str(new_package / 'stack.sac'),
            }
            stack_inputs = {
                'scope': 'group:group1',
                'align_marker': '6',
                'x1': -40.0,
                'x2': 20.0,
                'polarity': 'apply_user4',
                'stack_type': 'linear',
                'normalize': 'rms',
                'moveout_mode': 'off',
                'moveout_phase': '',
                'label': '',
            }
            metadata = {
                'result_package_dir': str(new_package),
                'stack_type': 'linear',
                'normalize': 'rms',
                'scope': 'group:group1',
                'align_marker': 't6',
                'x1': -40.0,
                'x2': 20.0,
                'wave_count_requested': 1,
                'wave_count_input': 1,
                'wave_count_used': 1,
                'wave_names_requested': ['new_member.sac'],
                'wave_names_aligned': ['new_member.sac'],
                'wave_names_used': ['new_member.sac'],
                'moveout_mode': 'off',
                'moveout_phase': None,
                'gcarc_mean': 1.0,
                'az_mean': 2.0,
                'baz_mean': 3.0,
                'pierce_lon_mean': 4.0,
                'pierce_lat_mean': 5.0,
                'event_info': {},
                'stack_markers': {'t6': 13.75},
            }

            saved_path = figure._write_stack_data_directory(output_paths, stack_inputs, metadata)

            self.assertFalse((stack_root / 'stack_group1_old.sac').exists())
            self.assertFalse((stack_root / 'stack_group1_old.stack.json').exists())
            self.assertFalse(old_package.exists())
            self.assertTrue(Path(saved_path).exists())
            index = json.loads(stack_index_path(stack_root).read_text(encoding='utf-8'))
            self.assertEqual(index['stack_count'], 1)
            self.assertEqual(index['stacks'][0]['wave_name'], Path(saved_path).name)

    def test_restack_group_uses_updated_configuration_for_new_filename(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_root = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_root.mkdir(parents=True, exist_ok=True)
            (stack_root / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            old_package = Path(temp_dir, 'analysis', 'stack_old_group1_package')
            old_package.mkdir(parents=True, exist_ok=True)
            (old_package / 'meta.json').write_text(json.dumps({'wave_names_used': ['old_member.sac']}), encoding='utf-8')
            old_trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            old_trace.stats.delta = 0.05
            old_trace.write(str(stack_root / 'stack_group1_old.sac'), format='SAC')
            (stack_root / 'stack_group1_old.stack.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'stack_wave_name': 'stack_group1_old.sac',
                        'group_name': 'group1',
                        'scope': 'group:group1',
                        'align_marker': 't6',
                        'window': [-40.0, 20.0],
                        'result_package_dir': str(old_package),
                    }
                ),
                encoding='utf-8',
            )

            new_package = Path(temp_dir, 'analysis', 'stack_group1_t5_xm50_p20_20260610_230000')
            new_package.mkdir(parents=True, exist_ok=True)
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(new_package / 'stack.sac'), format='SAC')
            (new_package / 'meta.json').write_text(json.dumps({'wave_names_used': ['new_member.sac']}), encoding='utf-8')

            figure.runtime_event_dir = _PROJ + '/data/pick_jandy/evt1'
            figure._stack_data_event_directory = lambda: str(stack_root)
            output_paths = {
                'output_root': str(new_package.parent),
                'basename_tag': new_package.name,
                'sac': str(new_package / 'stack.sac'),
            }
            stack_inputs = {
                'scope': 'group:group1',
                'align_marker': '5',
                'x1': -50.0,
                'x2': 20.0,
                'polarity': 'apply_user4',
                'stack_type': 'linear',
                'normalize': 'rms',
                'moveout_mode': 'off',
                'moveout_phase': '',
                'label': '',
            }
            metadata = {
                'result_package_dir': str(new_package),
                'stack_type': 'linear',
                'normalize': 'rms',
                'scope': 'group:group1',
                'align_marker': 't5',
                'x1': -50.0,
                'x2': 20.0,
                'wave_count_requested': 1,
                'wave_count_input': 1,
                'wave_count_used': 1,
                'wave_names_requested': ['new_member.sac'],
                'wave_names_aligned': ['new_member.sac'],
                'wave_names_used': ['new_member.sac'],
                'moveout_mode': 'off',
                'moveout_phase': None,
                'gcarc_mean': 1.0,
                'az_mean': 2.0,
                'baz_mean': 3.0,
                'pierce_lon_mean': 4.0,
                'pierce_lat_mean': 5.0,
                'event_info': {},
                'stack_markers': {'t5': 21.5},
            }

            saved_path = figure._write_stack_data_directory(output_paths, stack_inputs, metadata)

            self.assertTrue(saved_path.endswith('stack_group1_t5_xm50_p20.sac'))
            self.assertFalse((stack_root / 'stack_group1_old.sac').exists())
            self.assertFalse((stack_root / 'stack_group1_old.stack.json').exists())
            self.assertFalse(old_package.exists())
            self.assertTrue(Path(saved_path).exists())

    def test_write_stack_data_directory_rejects_invalid_package_sac(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_root = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            output_root = Path(temp_dir, 'data', 'output', 'process', 'stack', 'pick_jandy', 'evt1')
            package_dir = output_root / 'stack_bad'
            package_dir.mkdir(parents=True, exist_ok=True)
            (package_dir / 'stack.sac').write_bytes(b'SAC')

            figure.runtime_event_dir = _PROJ + '/data/pick_jandy/evt1'
            figure._stack_data_event_directory = lambda: str(stack_root)
            metadata = {
                'result_package_dir': str(package_dir),
                'stack_type': 'linear',
                'normalize': 'rms',
                'scope': 'visible',
                'align_marker': 't6',
                'x1': -40.0,
                'x2': 20.0,
                'wave_count_requested': 1,
                'wave_count_used': 1,
                'wave_names_requested': ['a.sac'],
                'wave_names_used': ['a.sac'],
                'gcarc_mean': 1.0,
                'az_mean': 2.0,
                'baz_mean': 3.0,
                'pierce_lon_mean': 4.0,
                'pierce_lat_mean': 5.0,
                'event_info': {},
                'stack_markers': {'t0': math.nan, 't6': 13.75},
                'outputs': {'json': str(package_dir / 'meta.json')},
            }
            output_paths = {
                'output_root': str(output_root),
                'basename_tag': 'stack_bad',
                'sac': str(package_dir / 'stack.sac'),
            }

            with self.assertRaises(RuntimeError):
                figure._write_stack_data_directory(output_paths, {}, metadata)

            self.assertFalse((stack_root / 'stack_bad.sac').exists())

    def test_save_preview_stack_outputs_cleans_package_dir_on_write_failure(self):
        figure = WaveFigure.__new__(WaveFigure)
        with tempfile.TemporaryDirectory() as temp_dir:
            figure.wavepath = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'
            figure.runtime_event_dir = figure.wavepath
            figure.dt = 0.05
            figure.preview_pierce_phase = 'PKIKP'
            figure.preview_pierce_model = 'iasp91'
            figure._current_bandpass_profile = lambda: {'low': 1.0, 'high': 2.0}
            figure._preview_stack_output_directory = lambda: temp_dir
            figure._stack_preview_pierce_mean = lambda wave_names: (math.nan, math.nan)
            figure._write_preview_stack_sac = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom'))

            trace_a = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            trace_a.stats.dpk_wave_name = 'waveA.sac'
            evtdata = SimpleNamespace(
                data=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
                time_axis=np.asarray([-1.0, 0.0, 1.0], dtype=float),
                wave_ori=[trace_a],
                sta_num=1,
                gcarc=np.asarray([87.0], dtype=float),
                az=np.asarray([65.0], dtype=float),
                baz=np.asarray([210.0], dtype=float),
            )
            stack_inputs = {
                'evtdata': evtdata,
                'align_marker': '6',
                'x1': -1.0,
                'x2': 1.0,
                'scope': 'visible',
                'requested_wave_names': ['waveA.sac'],
                'active_wave_names': ['waveA.sac'],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'linear',
                'label': '',
                'skipped_missing': [],
                'apply_user4_flips': True,
                'moveout_mode': 'off',
                'moveout_phase': '2',
                'moveout_applied': [],
                'moveout_skipped': [],
            }

            with self.assertRaises(RuntimeError):
                figure._save_preview_stack_outputs(
                    stack_inputs=stack_inputs,
                    stack_data=np.asarray([0.5, 1.0, 0.5], dtype=float),
                    normalized_rows=np.asarray([[0.5, 1.0, 0.5]], dtype=float),
                    valid_mask=np.asarray([True], dtype=bool),
                    skipped_reasons=[],
                )

            self.assertEqual(list(Path(temp_dir).iterdir()), [])

    def test_stack_output_dir_for_runtime_ignores_legacy_non_stack_output_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': '/tmp/legacy-test-output',
                    }
                ),
                encoding='utf-8',
            )

            output_dir = stack_output_dir_for_runtime(stack_event_dir)

            self.assertEqual(
                str(output_dir),
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1',
            )

    def test_load_stack_event_marker_normalizes_legacy_tmp_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': '/tmp/legacy-test-output',
                    }
                ),
                encoding='utf-8',
            )

            marker = load_stack_event_marker(stack_event_dir)

            self.assertEqual(marker['source_event_name'], 'evt1')
            self.assertEqual(
                marker['output_dir'],
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1',
            )

    def test_load_stack_sidecar_map_normalizes_legacy_tmp_output_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': '/tmp/legacy-test-output',
                        'source_event_name': 'evt1',
                    }
                ),
                encoding='utf-8',
            )
            (stack_event_dir / 'stack_a.stack.json').write_text(
                json.dumps(
                    {
                        'stack_wave_name': 'stack_a.sac',
                        'outputs': {
                            'json': '/tmp/legacy-test-output/stack_a/meta.json',
                            'members': '/tmp/legacy-test-output/stack_a/members.txt',
                        },
                    }
                ),
                encoding='utf-8',
            )

            sidecars = load_stack_sidecar_map(stack_event_dir)

            payload = sidecars['stack_a.sac']
            self.assertEqual(payload['source_event_name'], 'evt1')
            self.assertEqual(payload['source_event_dir'], _PROJ + '/data/pick_jandy/evt1')
            self.assertEqual(
                payload['outputs']['json'],
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1/stack_a/meta.json',
            )

    def test_stack_wave_summary_from_sidecar_formats_stable_summary(self):
        summary = stack_wave_summary_from_sidecar({
            'scope': 'group:group2',
            'stack_type': 'linear',
            'normalize': 'rms',
            'wave_count_used': 27,
        })

        self.assertEqual(summary, 'group:group2 | linear | rms | N=27')

    def test_build_stack_workspace_manifest_lists_valid_and_invalid_stacks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': _PROJ + '/data/output/process/stack/pick_jandy/evt1',
                    }
                ),
                encoding='utf-8',
            )
            valid_trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            valid_trace.stats.delta = 0.05
            valid_trace.write(str(stack_event_dir / 'stack_valid.sac'), format='SAC')
            (stack_event_dir / 'stack_valid.stack.json').write_text(
                json.dumps(
                    {
                        'stack_wave_name': 'stack_valid.sac',
                        'scope': 'group:group2',
                        'stack_type': 'linear',
                        'normalize': 'rms',
                        'wave_count_used': 27,
                        'wave_count_requested': 30,
                        'wave_names_requested': ['a.sac', 'b.sac', 'c.sac'],
                        'wave_names_aligned': ['a.sac', 'b.sac'],
                        'wave_names_used': ['a.sac'],
                        'skipped_missing_reference': ['c.sac'],
                        'skipped_normalization': [{'wave_name': 'b.sac', 'reason': 'zero rms scale'}],
                        'moveout_skipped': [{'wave_name': 'd.sac', 'reason': 'missing delta'}],
                        'geometry': {
                            'gcarc_mean': 88.1,
                            'pierce_lon_mean': -27.2,
                            'pierce_lat_mean': math.nan,
                        },
                        'result_package_dir': _PROJ + '/data/output/process/stack/pick_jandy/evt1/stack_valid',
                    }
                ),
                encoding='utf-8',
            )
            (stack_event_dir / 'stack_bad.sac').write_bytes(b'SAC')

            manifest = build_stack_workspace_manifest(stack_event_dir)

            self.assertEqual(manifest['stack_count'], 2)
            self.assertEqual(manifest['valid_stack_count'], 1)
            self.assertEqual(manifest['invalid_stack_count'], 1)
            valid_item = next(item for item in manifest['stacks'] if item['wave_name'] == 'stack_valid.sac')
            bad_item = next(item for item in manifest['stacks'] if item['wave_name'] == 'stack_bad.sac')
            self.assertTrue(valid_item['valid_sac'])
            self.assertEqual(valid_item['summary'], 'group:group2 | linear | rms | N=27')
            self.assertEqual(valid_item['wave_count_requested'], 30)
            self.assertEqual(valid_item['member_counts']['requested'], 3)
            self.assertEqual(valid_item['member_counts']['aligned'], 2)
            self.assertEqual(valid_item['member_counts']['used'], 1)
            self.assertEqual(valid_item['member_counts']['skipped_missing_reference'], 1)
            self.assertEqual(valid_item['member_counts']['skipped_normalization'], 1)
            self.assertEqual(valid_item['member_counts']['skipped_moveout'], 1)
            self.assertEqual(valid_item['members']['used'], ['a.sac'])
            self.assertEqual(valid_item['members']['skipped_normalization'][0]['reason'], 'zero rms scale')
            self.assertAlmostEqual(valid_item['pierce_lon_mean'], -27.2)
            self.assertIsNone(valid_item['pierce_lat_mean'])
            self.assertTrue(valid_item['has_sidecar'])
            self.assertTrue(valid_item['sidecar_valid'])
            self.assertTrue(valid_item['sidecar_needs_repair'])
            self.assertIn('markers', valid_item['sidecar_repair_fields'])
            self.assertIn('user_markers', valid_item['sidecar_repair_fields'])
            self.assertFalse(bad_item['valid_sac'])
            self.assertIn('Unknown format', bad_item['read_error'])
            self.assertFalse(bad_item['has_sidecar'])
            self.assertFalse(bad_item['sidecar_valid'])

    def test_build_stack_workspace_manifest_marks_invalid_sidecar_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            stack_event_marker_path(stack_event_dir).parent.mkdir(parents=True, exist_ok=True)
            stack_event_marker_path(stack_event_dir).write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_event_dir / 'stack_bad_meta.sac'), format='SAC')
            (stack_metadata_dir_for_event(stack_event_dir) / 'stack_bad_meta.stack.json').write_text('{bad json', encoding='utf-8')

            manifest = build_stack_workspace_manifest(stack_event_dir)

            item = manifest['stacks'][0]
            self.assertTrue(item['valid_sac'])
            self.assertTrue(item['has_sidecar'])
            self.assertFalse(item['sidecar_valid'])
            self.assertIn('JSONDecodeError', item['sidecar_error'])
            self.assertEqual(item['summary'], '')

    def test_write_stack_workspace_index_persists_manifest_and_health(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'source_event_name': 'evt1',
                        'output_dir': _PROJ + '/data/output/process/stack/pick_jandy/evt1',
                    }
                ),
                encoding='utf-8',
            )
            valid_trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            valid_trace.stats.delta = 0.05
            valid_trace.write(str(stack_event_dir / 'stack_valid.sac'), format='SAC')
            (stack_event_dir / 'stack_valid.stack.json').write_text(
                json.dumps(
                    {
                        'stack_wave_name': 'stack_valid.sac',
                        'scope': 'visible',
                        'stack_type': 'linear',
                        'normalize': 'rms',
                        'wave_count_used': 3,
                        'geometry': {
                            'gcarc_mean': 88.1,
                            'pierce_lon_mean': -27.2,
                            'pierce_lat_mean': -56.4,
                        },
                    }
                ),
                encoding='utf-8',
            )
            (stack_event_dir / 'stack_bad.sac').write_bytes(b'SAC')

            index = write_stack_workspace_index(stack_event_dir)
            saved_index = json.loads(stack_index_path(stack_event_dir).read_text(encoding='utf-8'))

            self.assertEqual(index['mode'], 'stack_index')
            self.assertEqual(index['stack_count'], 2)
            self.assertEqual(index['valid_stack_count'], 1)
            self.assertEqual(index['invalid_stack_count'], 1)
            self.assertEqual(index['health']['valid_sac_count'], 1)
            self.assertEqual(saved_index['source_event_name'], 'evt1')
            self.assertEqual(saved_index['stacks'][0]['wave_name'], 'stack_bad.sac')

    def test_stack_finish_synchronizes_sidecar_markers_and_keeps_stack_file_in_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'source_event_name': 'evt1',
                        'output_dir': _PROJ + '/data/output/process/stack/pick_jandy/evt1',
                    }
                ),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.stats.sac = obspy.core.AttribDict(b=0.0, t0=0.0, t6=0.0, user1=math.nan)
            trace.write(str(stack_event_dir / 'stack_a.sac'), format='SAC')
            (stack_event_dir / 'stack_a.stack.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'stack_wave_name': 'stack_a.sac',
                        'scope': 'visible',
                        'stack_type': 'linear',
                        'normalize': 'rms',
                        'wave_count_used': 3,
                        'markers': {'t0': 0.0, 't6': 0.0},
                    }
                ),
                encoding='utf-8',
            )

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_event_dir)
            figure.stack_mode = True
            figure.ori_sacnames = ['stack_a.sac']
            figure.markers = {str(idx): {'stack_a.sac': math.nan} for idx in range(10)}
            figure.markers['6']['stack_a.sac'] = 12.34
            figure.markers['7']['stack_a.sac'] = 23.45
            figure.user_markers = {
                key: {'stack_a.sac': math.nan}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }
            figure.user_markers['user1']['stack_a.sac'] = 1.0
            figure.user_markers['user4']['stack_a.sac'] = -1.0
            moved_files = []
            figure._move_file_to_bucket = lambda *args: moved_files.append(args)

            figure.finish()

            self.assertTrue((stack_event_dir / 'stack_a.sac').exists())
            self.assertEqual(moved_files, [])
            saved_trace = obspy.read(str(stack_event_dir / 'stack_a.sac'))[0]
            self.assertAlmostEqual(float(saved_trace.stats.sac.t6), 12.34, delta=1e-5)
            self.assertAlmostEqual(float(saved_trace.stats.sac.t7), 23.45, delta=1e-5)
            self.assertAlmostEqual(float(saved_trace.stats.sac.user1), 1.0, delta=1e-5)
            sidecar = json.loads((stack_event_dir / 'stack_a.stack.json').read_text(encoding='utf-8'))
            self.assertIsNone(sidecar['markers']['t0'])
            self.assertAlmostEqual(sidecar['markers']['t6'], 12.34)
            self.assertAlmostEqual(sidecar['markers']['t7'], 23.45)
            self.assertAlmostEqual(sidecar['user_markers']['user1'], 1.0)
            self.assertAlmostEqual(sidecar['user_markers']['user4'], -1.0)
            self.assertIn('updated_at', sidecar)
            saved_index = json.loads(stack_index_path(stack_event_dir).read_text(encoding='utf-8'))
            self.assertEqual(saved_index['valid_stack_count'], 1)
            self.assertEqual(saved_index['stacks'][0]['wave_name'], 'stack_a.sac')

    def test_stack_finish_updates_result_package_sac_headers_too(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'source_event_name': 'evt1',
                        'output_dir': _PROJ + '/data/output/stack/analysis/pick_jandy/evt1',
                    }
                ),
                encoding='utf-8',
            )
            package_dir = Path(temp_dir, 'analysis', 'stack_group3_t5_xm50_p20_20260616_145000')
            package_dir.mkdir(parents=True, exist_ok=True)
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.stats.sac = obspy.core.AttribDict(b=0.0, t5=10.0, t7=11.0)
            trace.write(str(package_dir / 'stack.sac'), format='SAC')
            trace.write(str(stack_event_dir / 'stack_group3_t5_xm50_p20.sac'), format='SAC')
            (stack_event_dir / 'stack_group3_t5_xm50_p20.stack.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'stack_wave_name': 'stack_group3_t5_xm50_p20.sac',
                        'group_name': 'group3',
                        'scope': 'group:group3',
                        'align_marker': 't5',
                        'window': [-50.0, 20.0],
                        'result_package_dir': str(package_dir),
                    }
                ),
                encoding='utf-8',
            )

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_event_dir)
            figure.stack_mode = True
            figure.ori_sacnames = ['stack_group3_t5_xm50_p20.sac']
            figure.markers = {str(idx): {'stack_group3_t5_xm50_p20.sac': math.nan} for idx in range(10)}
            figure.markers['5']['stack_group3_t5_xm50_p20.sac'] = 21.0
            figure.markers['7']['stack_group3_t5_xm50_p20.sac'] = 11.0
            figure.user_markers = {key: {'stack_group3_t5_xm50_p20.sac': math.nan} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}

            figure.finish()

            workspace_trace = obspy.read(str(stack_event_dir / 'stack_group3_t5_xm50_p20.sac'))[0]
            package_trace = obspy.read(str(package_dir / 'stack.sac'))[0]
            self.assertAlmostEqual(float(workspace_trace.stats.sac.b), 0.0)
            self.assertAlmostEqual(float(package_trace.stats.sac.b), 0.0)
            self.assertAlmostEqual(float(workspace_trace.stats.sac.t5), 50.0)
            self.assertAlmostEqual(float(workspace_trace.stats.sac.t7), 40.0)
            self.assertAlmostEqual(float(package_trace.stats.sac.t5), 50.0)
            self.assertAlmostEqual(float(package_trace.stats.sac.t7), 40.0)

    def test_stack_window_relative_markers_preserve_existing_relative_t8(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.wavepath = '/tmp/stack_evt'
        figure.stack_sidecars = {
            'stack_group1_t6_xm40_p30.sac': {
                'align_marker': 't6',
                'window': [-40.0, 30.0],
                'markers': {'t6': 502.63, 't7': 478.15, 't8': 33.907},
            }
        }
        figure.markers = {str(idx): {'stack_group1_t6_xm40_p30.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_group1_t6_xm40_p30.sac'] = 502.63
        figure.markers['7']['stack_group1_t6_xm40_p30.sac'] = 478.15
        figure.markers['8']['stack_group1_t6_xm40_p30.sac'] = 33.907

        relative_markers, window_length = figure._stack_window_relative_markers_for_wave('stack_group1_t6_xm40_p30.sac')

        self.assertAlmostEqual(window_length, 70.0)
        self.assertAlmostEqual(relative_markers['t6'], 40.0)
        self.assertAlmostEqual(relative_markers['t7'], 15.52, places=2)
        self.assertAlmostEqual(relative_markers['t8'], 33.907, places=3)

    def test_stack_window_relative_markers_keep_manual_stack_marker_positions(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.wavepath = '/tmp/stack_evt'
        figure.stack_sidecars = {
            'stack_group1_t6_xm40_p30.sac': {
                'align_marker': 't6',
                'window': [-40.0, 30.0],
                'markers': {'t6': 502.63, 't7': 478.15, 't8': 33.907},
            }
        }
        figure.markers = {str(idx): {'stack_group1_t6_xm40_p30.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_group1_t6_xm40_p30.sac'] = 40.0
        figure.markers['7']['stack_group1_t6_xm40_p30.sac'] = 15.522
        figure.markers['8']['stack_group1_t6_xm40_p30.sac'] = 13.0
        figure.stack_manual_marker_keys = {
            ('stack_group1_t6_xm40_p30.sac', '7'),
            ('stack_group1_t6_xm40_p30.sac', '8'),
        }

        relative_markers, window_length = figure._stack_window_relative_markers_for_wave('stack_group1_t6_xm40_p30.sac')

        self.assertAlmostEqual(window_length, 70.0)
        self.assertAlmostEqual(relative_markers['t6'], 40.0)
        self.assertAlmostEqual(relative_markers['t7'], 15.522, places=3)
        self.assertAlmostEqual(relative_markers['t8'], 13.0, places=3)

    def test_stack_marker_display_x_value_keeps_existing_window_relative_markers(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.dt = 0.02
        figure.ori_sacnames = ['stack_group2_t6_xm40_p30.sac']
        figure.wave = []
        figure._stack_sidecar_relative_window = lambda wave_name: (-40.0, 30.0)
        figure._trace_time_bounds = lambda trace: (math.nan, math.nan)

        display_time = figure._stack_marker_display_x_value(22.750819156044486, 0)

        self.assertAlmostEqual(display_time, 22.750819156044486, places=6)

    def test_stack_crustal_summary_text_skips_missing_marker_pairs(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_crustal_taup_bin = '/tmp/taup'
        figure._stack_crustal_summary_cache = {}
        figure.markers = {str(idx): {'stack_group2_t6_xm40_p30.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_group2_t6_xm40_p30.sac'] = 40.0
        figure.markers['8']['stack_group2_t6_xm40_p30.sac'] = 33.0
        figure.stack_sidecars = {
            'stack_group2_t6_xm40_p30.sac': {
                'geometry': {
                    'pierce_lon_mean': -26.3,
                    'pierce_lat_mean': -56.2,
                    'gcarc_mean': 12.4,
                },
                'event': {'evdp': 118.0},
                'markers': {'t6': 40.0, 't8': 33.0},
            }
        }

        with patch('WaveFigure.fetch_taup_ray_parameter', return_value=14.0):
            text = figure._stack_crustal_summary_text('stack_group2_t6_xm40_p30.sac')

        self.assertIn('pP-pmP:', text)
        self.assertNotIn('sP-smP:', text)

    def test_stack_crustal_summary_text_refreshes_after_marker_update(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_crustal_taup_bin = '/tmp/taup'
        figure._stack_crustal_summary_cache = {}
        figure.markers = {str(idx): {'stack_group2_t6_xm40_p30.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_group2_t6_xm40_p30.sac'] = 40.0
        figure.markers['8']['stack_group2_t6_xm40_p30.sac'] = 33.0
        figure.markers['5']['stack_group2_t6_xm40_p30.sac'] = 22.0
        figure.markers['9']['stack_group2_t6_xm40_p30.sac'] = 15.0
        figure.stack_sidecars = {
            'stack_group2_t6_xm40_p30.sac': {
                'geometry': {
                    'pierce_lon_mean': -26.3,
                    'pierce_lat_mean': -56.2,
                    'gcarc_mean': 12.4,
                },
                'event': {'evdp': 118.0},
                'markers': {'t5': 22.0, 't6': 40.0, 't8': 33.0, 't9': 15.0},
            }
        }

        def fake_taup(*args, **kwargs):
            # Realistic ray parameters (s/deg): the sP value must stay below
            # 1/Vp * 111.195 (~17.4) or the crustal term goes negative and the
            # thickness is suppressed, which would hide sP-smP entirely.
            return 14.0 if kwargs.get('phase') == 'pP' else 6.5

        with patch('WaveFigure.fetch_taup_ray_parameter', side_effect=fake_taup):
            first_text = figure._stack_crustal_summary_text('stack_group2_t6_xm40_p30.sac')
            figure.markers['9']['stack_group2_t6_xm40_p30.sac'] = 14.0
            second_text = figure._stack_crustal_summary_text('stack_group2_t6_xm40_p30.sac')

        self.assertIn('sP-smP:', first_text)
        self.assertIn('sP-smP:', second_text)
        self.assertNotEqual(first_text, second_text)

    def test_stack_crustal_summary_uses_same_formula_as_review_index(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_crustal_taup_bin = '/tmp/taup'
        figure._stack_crustal_summary_cache = {}
        wave_name = 'stack_group2_t6_xm40_p30.sac'
        figure.markers = {str(idx): {wave_name: math.nan} for idx in range(10)}
        figure.markers['6'][wave_name] = 40.0
        figure.markers['8'][wave_name] = 33.0
        figure.stack_sidecars = {
            wave_name: {
                'geometry': {
                    'gcarc_mean': 12.4,
                    'az_mean': 145.0,
                    'pierce_lon_mean': -26.3,
                    'pierce_lat_mean': -56.2,
                },
                'event': {
                    'evdp': 118.0,
                    'evla': -59.0,
                    'evlo': -27.0,
                },
                'markers': {'t6': 40.0, 't8': 33.0},
            }
        }

        with patch('WaveFigure.fetch_taup_ray_parameter', return_value=14.0):
            summary = figure._stack_crustal_summary(wave_name)

        expected = calculate_pp_pmp_thickness(7.0, DEFAULT_CRUST_VP, 14.0)
        self.assertAlmostEqual(summary['pp_pmp_km'], expected, places=6)

    def test_open_workspace_window_uses_sac_suffix_for_source_event_dirs(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        widget.x1 = -10
        widget.x2 = 10
        widget.pending_tmarker = 't7'
        widget.preview_phases = ['t7', 't6']
        widget.suffix = '.bhz'
        widget.show_status_message = lambda *args, **kwargs: None
        widget._child_windows = []
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                order='gcarc',
                _persist_markers_to_disk=lambda: None,
            )
        )
        created = {}

        class DummyWindow:
            def __init__(self, *args, **kwargs):
                created['wavepath'] = args[0]
                created['suffix'] = kwargs.get('suffix')
                created['member_filter'] = kwargs.get('member_filter')
                self.mpl = SimpleNamespace(setFocus=lambda: None)

            def show(self):
                return None

            def raise_(self):
                return None

            def activateWindow(self):
                return None

            def _set_geom_center(self):
                return None

        # 与本文件其余用例一致，由 PROJECT_ROOT 派生，不写死绝对路径。
        event_dir = _PROJ + '/data/pick_jandy/2011_09_03_04_48_58'
        with patch('ppk.exists', return_value=True), patch('ppk.is_stack_event_dir', return_value=False), patch(
            'ppk.MatplotlibWidget', DummyWindow
        ), patch('ppk.QTimer.singleShot', lambda *args, **kwargs: None):
            opened = MatplotlibWidget._open_workspace_window(widget, event_dir, member_filter={'waveA.sac'})

        self.assertIsNotNone(opened)
        self.assertEqual(created['wavepath'], event_dir)
        self.assertEqual(created['suffix'], '.sac')
        self.assertEqual(created['member_filter'], {'waveA.sac'})

    def test_main_window_schedules_preview_resource_warmup(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        scheduled = []
        widget.layout = SimpleNamespace(
            addWidget=lambda *args, **kwargs: None,
            setSizeConstraint=lambda value: None,
        )
        widget.pending_tmarker = 't7'
        widget.stack_mode_for_controls = False
        widget.pending_axis_mode = 'relative'
        widget.x1 = -10
        widget.x2 = 70
        widget.bp_freqmin = 0.05
        widget.bp_freqmax = 0.4
        widget.bp_corners = 2
        widget.bp_passes = 2
        widget.preview_phases = ['t7']
        widget.bp_presets = []
        widget.phase_presets = []
        widget.tmaker = 't7'
        widget.axis_mode = 'relative'
        widget.suffix = '.sac'
        widget.preview_shortcuts = []
        widget.station_search_visible = False
        widget.mpl = SimpleNamespace(
            setFocus=lambda: None,
            mpl_connect=lambda *args, **kwargs: None,
            wavefig=SimpleNamespace(
                stack_mode=False,
                jump_status_callback=None,
                status_callback=None,
                phase_tokens_change_callback=None,
                stack_review_refresh_callback=None,
            ),
        )
        widget.show_status_message = lambda *args, **kwargs: None
        widget._populate_event_combo = lambda: None
        widget._set_geom_center = lambda: None
        widget._define_global_shortcuts = lambda: None
        widget.setWindowTitle = lambda value: None
        widget.setWindowIcon = lambda value: None

        class DummyMenu:
            def addAction(self, *args, **kwargs):
                return None

            def addSeparator(self):
                return None

        widget.menuBar = lambda: SimpleNamespace(addMenu=lambda name: DummyMenu())
        widget.setCentralWidget = lambda value: None
        widget._force_visible_cursor = lambda *args, **kwargs: None
        widget._load_bp_presets = lambda: []
        widget._load_phase_presets = lambda: []
        widget.add_btn = lambda: None

        class DummyCanvas:
            def __init__(self, *args, **kwargs):
                self.wavefig = widget.mpl.wavefig

            def mpl_connect(self, *args, **kwargs):
                return None

            def setFocus(self):
                return None

        with patch('ppk.QVBoxLayout', lambda: widget.layout), patch('ppk.QWidget', lambda: SimpleNamespace(setLayout=lambda layout: None)), patch(
            'ppk.MyMplCanvas', DummyCanvas
        ), patch('ppk.QAction', lambda *args, **kwargs: SimpleNamespace(setShortcut=lambda value: None, setStatusTip=lambda value: None, triggered=SimpleNamespace(connect=lambda fn: None))), patch(
            'ppk.QTimer.singleShot', lambda delay, fn: scheduled.append((delay, fn))
        ), patch('ppk.QIcon', lambda *args, **kwargs: None):
            MatplotlibWidget.initUi(widget, '/tmp/event', [-10, 70], 'gcarc', 't7', '.sac', 't7', None)

        self.assertTrue(
            any(delay == 250 and fn == widget._warm_default_preview_resources for delay, fn in scheduled)
        )

    def test_persist_markers_to_disk_triggers_stack_review_refresh_callback(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = '/tmp/fake_event'
        figure.stack_mode = True
        figure.markers = {str(i): {} for i in range(10)}
        figure.ori_sacnames = np.array([], dtype=object)
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure._stack_window_relative_markers_for_wave = lambda sac_file: None
        figure._user_marker_value = lambda sac_file, key: math.nan
        figure._sync_stack_sidecar_from_markers = lambda sac_file: None
        figure._sync_stack_package_sac_from_markers = lambda sac_file: None
        called = {'count': 0}
        figure.stack_review_refresh_callback = lambda: called.__setitem__('count', called['count'] + 1)

        with patch('WaveFigure.write_stack_workspace_index', lambda wave_path: {}):
            persisted = figure._persist_markers_to_disk()

        self.assertEqual(persisted, [])
        self.assertEqual(called['count'], 1)

    def test_persist_markers_to_disk_writes_only_dirty_source_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            event_dir = Path(temp_dir)

            def write_sac(path, t7):
                trace = Trace(data=np.ones(100, dtype=np.float32))
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(b=0.0, e=4.95, t7=t7)
                trace.write(str(path), format='SAC')

            write_sac(event_dir / 'wave_a.sac', 10.0)
            write_sac(event_dir / 'wave_b.sac', 20.0)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(event_dir)
            figure.stack_mode = False
            figure.ori_sacnames = ['wave_a.sac', 'wave_b.sac']
            figure.markers = {str(i): {} for i in range(10)}
            figure.markers['7'] = {'wave_a.sac': 11.0, 'wave_b.sac': 21.0}
            figure.user_markers = {
                key: {'wave_a.sac': math.nan, 'wave_b.sac': math.nan}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }
            figure.dirty_marker_wave_names = {'wave_a.sac'}
            figure.stack_review_refresh_callback = None
            figure._flush_pending_source_marker_writes = lambda notify_review=False: []
            figure._stack_window_relative_markers_for_wave = lambda sac_file: None

            persisted = figure._persist_markers_to_disk()

            self.assertEqual(persisted, ['wave_a.sac'])
            self.assertAlmostEqual(obspy.read(str(event_dir / 'wave_a.sac'))[0].stats.sac.t7, 11.0)
            self.assertAlmostEqual(obspy.read(str(event_dir / 'wave_b.sac'))[0].stats.sac.t7, 20.0)
            self.assertEqual(figure.dirty_marker_wave_names, set())

    def test_request_external_refresh_schedules_rebuild_instead_of_running_inline(self):
        review = ThicknessReviewWindow.__new__(ThicknessReviewWindow)
        review._last_focus_refresh = 99
        review._refresh_pending = False
        review._rebuild_running = False
        scheduled = {}
        review._run_scheduled_rebuild = lambda: scheduled.__setitem__('fired', True)

        with patch('stack_thickness_review_dialog.QTimer.singleShot', lambda delay, fn: scheduled.update(delay=delay, fn=fn)):
            ThicknessReviewWindow.request_external_refresh(review)

        self.assertEqual(review._last_focus_refresh, 0)
        self.assertTrue(review._refresh_pending)
        self.assertEqual(scheduled['delay'], 80)
        self.assertIs(scheduled['fn'], review._run_scheduled_rebuild)

    def test_thickness_table_tooltip_marks_outlier_map_symbol(self):
        point = SimpleNamespace(
            event_key='pick_other/2019_04_22_14_49_05',
            group_name='group4',
            pair_kind='t6+t8',
            phase_kind='pP',
            align_marker='t6',
            member_count_used=13,
            gcarc=45.1,
            thickness_km=7.0,
            group_key='pick_other/2019_04_22_14_49_05|group4|t6+t8',
        )
        model = ThicknessTableModel()
        model.set_points([point], {point.group_key: OUTLIER_THRESHOLD + 0.5}, event_keys_order=[point.event_key])
        map_col = 2
        idx = model.index(0, map_col)

        tooltip = model.data(idx, Qt.ItemDataRole.ToolTipRole)

        self.assertIn('[outlier]', tooltip)

    def test_stack_finish_skips_invalid_stack_sac_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_event_dir / 'bad.sac').write_bytes(b'SAC')

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_event_dir)
            figure.stack_mode = True
            figure.ori_sacnames = ['bad.sac']
            figure.markers = {str(idx): {'bad.sac': math.nan} for idx in range(10)}
            figure.user_markers = {
                key: {'bad.sac': math.nan}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }

            figure.finish()

            saved_index = json.loads(stack_index_path(stack_event_dir).read_text(encoding='utf-8'))
            self.assertEqual(saved_index['invalid_stack_count'], 1)

    def test_stack_sidecar_marker_sync_serializes_numpy_scalar_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'source_event_name': 'evt1',
                    }
                ),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_event_dir / 'stack_a.sac'), format='SAC')
            (stack_event_dir / 'stack_a.stack.json').write_text(
                json.dumps({'mode': 'stack', 'stack_wave_name': 'stack_a.sac'}),
                encoding='utf-8',
            )

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_event_dir)
            figure.stack_mode = True
            figure.markers = {str(idx): {'stack_a.sac': np.float32(idx)} for idx in range(10)}
            figure.user_markers = {
                key: {'stack_a.sac': np.float32(1.0)}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }

            figure._sync_stack_sidecar_from_markers('stack_a.sac')

            sidecar = json.loads((stack_event_dir / 'stack_a.stack.json').read_text(encoding='utf-8'))
            self.assertEqual(sidecar['markers']['t6'], 6.0)
            self.assertEqual(sidecar['user_markers']['user1'], 1.0)

    def test_build_stack_workspace_index_matches_manifest_counts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_event_dir / 'stack_a.sac'), format='SAC')

            manifest = build_stack_workspace_manifest(stack_event_dir)
            index = build_stack_workspace_index(stack_event_dir)

            self.assertEqual(index['stack_count'], manifest['stack_count'])
            self.assertEqual(index['valid_stack_count'], manifest['valid_stack_count'])
            self.assertEqual(index['health']['valid_sac_count'], 1)

    def test_repair_stack_event_metadata_rewrites_legacy_marker_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            marker_path = stack_event_dir / '.stack_event.json'
            sidecar_path = stack_event_dir / 'stack_a.stack.json'
            marker_path.write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': '/tmp/legacy-test-output',
                    }
                ),
                encoding='utf-8',
            )
            sidecar_path.write_text(
                json.dumps(
                    {
                        'stack_wave_name': 'stack_a.sac',
                        'outputs': {
                            'json': '/tmp/legacy-test-output/stack_a/meta.json',
                        },
                    }
                ),
                encoding='utf-8',
            )

            report = repair_stack_event_metadata(stack_event_dir, persist=True)

            self.assertTrue(report['marker_updated'])
            repaired_marker_path = stack_event_marker_path(stack_event_dir)
            self.assertEqual(report['sidecars_updated'], [str(sidecar_path)])
            repaired_marker = json.loads(repaired_marker_path.read_text(encoding='utf-8'))
            repaired_sidecar = json.loads(sidecar_path.read_text(encoding='utf-8'))
            self.assertEqual(repaired_marker['source_event_name'], 'evt1')
            self.assertEqual(
                repaired_marker['output_dir'],
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1',
            )
            self.assertEqual(repaired_sidecar['source_event_name'], 'evt1')
            self.assertEqual(
                repaired_sidecar['outputs']['json'],
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1/stack_a/meta.json',
            )
            self.assertEqual(set(repaired_sidecar['markers'].keys()), {f't{idx}' for idx in range(10)})
            self.assertEqual(set(repaired_sidecar['user_markers'].keys()), {'user1', 'user2', 'user3', 'user4', 'user5'})
            self.assertIsNone(repaired_sidecar['markers']['t6'])
            self.assertIsNone(repaired_sidecar['user_markers']['user1'])

    def test_repair_stack_event_metadata_reports_invalid_sidecar_without_rewriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            marker_path = stack_event_dir / '.stack_event.json'
            sidecar_path = stack_event_dir / 'stack_bad_meta.stack.json'
            marker_path.write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            sidecar_path.write_text('{bad json', encoding='utf-8')

            report = repair_stack_event_metadata(stack_event_dir, persist=True)

            self.assertEqual(report['sidecars_updated'], [])
            self.assertEqual(report['invalid_sidecars'][0]['path'], str(sidecar_path))
            self.assertIn('JSONDecodeError', report['invalid_sidecars'][0]['reason'])
            self.assertEqual(sidecar_path.read_text(encoding='utf-8'), '{bad json')

    def test_repair_stack_event_metadata_dry_run_does_not_rewrite_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            marker_path = stack_event_dir / '.stack_event.json'
            original = {
                'mode': 'stack',
                'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                'output_dir': '/tmp/legacy-test-output',
            }
            marker_path.write_text(json.dumps(original), encoding='utf-8')

            report = repair_stack_event_metadata(stack_event_dir, persist=False)

            self.assertTrue(report['marker_updated'])
            self.assertEqual(json.loads(marker_path.read_text(encoding='utf-8')), original)

    def test_inspect_stack_event_health_reports_invalid_and_missing_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_event_dir / 'bad.sac').write_bytes(b'SAC')
            (stack_event_dir / 'bad.stack.json').write_text(json.dumps({'stack_wave_name': 'bad.sac'}), encoding='utf-8')
            (stack_event_dir / 'good.sac').write_bytes(b'SACDATA')
            (stack_event_dir / 'orphan.stack.json').write_text(json.dumps({'stack_wave_name': 'orphan.sac'}), encoding='utf-8')

            report = inspect_stack_event_health(stack_event_dir)

            self.assertEqual(len(report['invalid_sac_files']), 2)
            self.assertEqual(len(report['missing_sidecars']), 1)
            self.assertEqual(len(report['orphan_sidecars']), 1)

    def test_inspect_stack_event_health_reports_invalid_and_repairable_sidecars(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            for wave_name in ('stack_bad_meta.sac', 'stack_legacy.sac'):
                trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
                trace.stats.delta = 0.05
                trace.write(str(stack_event_dir / wave_name), format='SAC')
            (stack_event_dir / 'stack_bad_meta.stack.json').write_text('{bad json', encoding='utf-8')
            (stack_event_dir / 'stack_legacy.stack.json').write_text(
                json.dumps(
                    {
                        'stack_wave_name': 'wrong_name.sac',
                        'outputs': {'json': '/tmp/legacy-test-output/stack_legacy/meta.json'},
                        'markers': {'t6': 0.0},
                    }
                ),
                encoding='utf-8',
            )

            report = inspect_stack_event_health(stack_event_dir)

            self.assertEqual(report['valid_sac_count'], 2)
            self.assertEqual(report['invalid_sidecars'][0]['wave_name'], 'stack_bad_meta.sac')
            self.assertIn('JSONDecodeError', report['invalid_sidecars'][0]['reason'])
            repair_item = report['sidecars_needing_repair'][0]
            self.assertEqual(repair_item['wave_name'], 'stack_legacy.sac')
            self.assertIn('stack_wave_name', repair_item['fields'])
            self.assertIn('user_markers', repair_item['fields'])

    def test_quarantine_invalid_stack_files_dry_run_reports_without_moving(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_event_dir / 'bad.sac').write_bytes(b'SAC')
            (stack_event_dir / 'bad.stack.json').write_text(json.dumps({'stack_wave_name': 'bad.sac'}), encoding='utf-8')

            report = quarantine_invalid_stack_files(stack_event_dir, persist=False)

            self.assertEqual(report['invalid_count'], 1)
            self.assertEqual(len(report['moved']), 1)
            self.assertEqual(len(report['sidecars_moved']), 1)
            self.assertTrue((stack_event_dir / 'bad.sac').exists())
            self.assertTrue((stack_event_dir / 'bad.stack.json').exists())

    def test_quarantine_invalid_stack_files_moves_bad_sac_and_sidecar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_event_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_event_dir.mkdir(parents=True, exist_ok=True)
            (stack_event_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_event_dir / 'bad.sac').write_bytes(b'SAC')
            (stack_event_dir / 'bad.stack.json').write_text(json.dumps({'stack_wave_name': 'bad.sac'}), encoding='utf-8')

            report = quarantine_invalid_stack_files(stack_event_dir, persist=True)

            quarantine_dir = Path(report['quarantine_dir'])
            self.assertEqual(report['invalid_count'], 1)
            self.assertFalse((stack_event_dir / 'bad.sac').exists())
            self.assertFalse((stack_event_dir / 'bad.stack.json').exists())
            self.assertTrue((quarantine_dir / 'bad.sac').exists())
            self.assertTrue((quarantine_dir / 'bad.stack.json').exists())

    def test_standard_export_event_name_prefers_semantic_event_name(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/stack/pick_jandy/2011_03_06_14_32_36'
        figure.runtime_event_dir = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'
        figure.stack_event_marker = {'source_event_name': '2011_03_06_14_32_36'}

        event_name = figure._standard_export_event_name(SimpleNamespace(evtname='2011.065.14.32.20'))

        self.assertEqual(event_name, '2011_03_06_14_32_36')

    def test_semantic_event_name_prefers_stack_marker_source_event_name(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.wavepath = _PROJ + '/data/stack/pick_jandy/2011_03_06_14_32_36'
        figure.runtime_event_dir = _PROJ + '/data/pick_jandy/2011_03_06_14_32_36'
        figure.stack_event_marker = {'source_event_name': '2011_03_06_14_32_36'}

        self.assertEqual(figure._semantic_event_name(), '2011_03_06_14_32_36')

    def test_alignment_status_summary_reports_stack_repairs(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'0': ('t0', '#000000')}
        figure.ori_sacnames = ['a.sac', 'b.sac']
        figure.stack_skipped_wave_files = []
        figure.stack_repair_report = {
            'marker_updated': True,
            'sidecars_updated': ['x.stack.json', 'y.stack.json'],
        }
        figure.stack_health_report = {'invalid_sac_files': []}
        figure._alignment_marker_key = lambda marker=None: '0'
        figure.missing_alignment_wave_names = lambda marker=None: []
        figure._phase_display_label = lambda marker_key: 'P'

        summary = figure.alignment_status_summary()

        self.assertIn('Stack repaired 3', summary)

    def test_alignment_status_summary_reports_invalid_stack_files(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'0': ('t0', '#000000')}
        figure.ori_sacnames = ['a.sac']
        figure.stack_skipped_wave_files = []
        figure.stack_repair_report = {}
        figure.stack_health_report = {'invalid_sac_files': [{'path': 'bad.sac', 'reason': 'bad'}]}
        figure._alignment_marker_key = lambda marker=None: '0'
        figure.missing_alignment_wave_names = lambda marker=None: []
        figure._phase_display_label = lambda marker_key: 'P'

        summary = figure.alignment_status_summary()

        self.assertIn('Stack invalid 1', summary)

    def test_alignment_status_summary_reports_stack_sidecar_health(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'0': ('t0', '#000000')}
        figure.ori_sacnames = ['a.sac']
        figure.stack_skipped_wave_files = []
        figure.stack_repair_report = {}
        figure.stack_health_report = {
            'invalid_sac_files': [],
            'invalid_sidecars': [{'path': 'bad.stack.json'}],
            'sidecars_needing_repair': [{'path': 'legacy.stack.json'}],
        }
        figure._alignment_marker_key = lambda marker=None: '0'
        figure.missing_alignment_wave_names = lambda marker=None: []
        figure._phase_display_label = lambda marker_key: 'P'

        summary = figure.alignment_status_summary()

        self.assertIn('Stack invalid sidecar 1', summary)
        self.assertIn('Stack sidecar repair 1', summary)

    def test_stack_health_action_requires_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(wavefig=SimpleNamespace(stack_mode=False))

        widget.inspect_stack_health()

        self.assertEqual(messages[0][0], 'Stack health is only available in stack mode')

    def test_stack_quarantine_action_requires_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(wavefig=SimpleNamespace(stack_mode=False))

        widget.quarantine_invalid_stack_files()

        self.assertEqual(messages[0][0], 'Stack quarantine is only available in stack mode')

    def test_stack_quarantine_preview_requires_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(wavefig=SimpleNamespace(stack_mode=False))

        widget.preview_invalid_stack_quarantine()

        self.assertEqual(messages[0][0], 'Stack quarantine preview is only available in stack mode')

    def test_close_topmost_window_closes_preview_before_finishing_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        events = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                stack_mode=True,
                finish=lambda: events.append('finish'),
                close_preview_window=lambda: events.append('close_preview') or True,
            )
        )

        with patch('ppk.QApplication.focusWidget', return_value=None):
            widget.close_topmost_window()

        self.assertEqual(events, ['close_preview'])
        self.assertEqual(messages[0][0], 'Closed preview window')

    def test_finish_saves_and_closes_only_current_window(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        events = []
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                finish=lambda: events.append('finish'),
                close_preview_window=lambda: events.append('close_preview') or False,
            )
        )
        widget.close = lambda: events.append('close_window')

        widget.finish()

        self.assertEqual(events, ['finish', 'close_preview', 'close_window'])

    def test_stack_quarantine_preview_reports_without_moving_files(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_dir / 'bad.sac').write_bytes(b'SAC')
            (stack_dir / 'bad.stack.json').write_text(json.dumps({'stack_wave_name': 'bad.sac'}), encoding='utf-8')
            widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
            wavefig = SimpleNamespace(stack_mode=True, wavepath=str(stack_dir), stack_health_report={})
            widget.mpl = SimpleNamespace(wavefig=wavefig)

            widget.preview_invalid_stack_quarantine()

            self.assertTrue((stack_dir / 'bad.sac').exists())
            self.assertTrue((stack_dir / 'bad.stack.json').exists())
            self.assertEqual(wavefig.stack_health_report['invalid_sac_files'][0]['path'], str(stack_dir / 'bad.sac'))
            self.assertIn('Would quarantine 1 invalid stack file(s): bad.sac', messages[0][0])

    def test_stack_index_action_requires_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(wavefig=SimpleNamespace(stack_mode=False))

        widget.refresh_stack_index()

        self.assertEqual(messages[0][0], 'Stack index is only available in stack mode')

    def test_stack_index_action_refreshes_workspace_index(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            stack_trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
            stack_trace.stats.delta = 0.05
            stack_trace.write(str(stack_dir / 'stack_a.sac'), format='SAC')
            widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
            wavefig = SimpleNamespace(stack_mode=True, wavepath=str(stack_dir), stack_health_report={})
            widget.mpl = SimpleNamespace(wavefig=wavefig)

            widget.refresh_stack_index()

            self.assertTrue(stack_index_path(stack_dir).exists())
            self.assertEqual(wavefig.stack_health_report['valid_sac_count'], 1)
            self.assertIn('Stack index refreshed: total 1, valid 1, invalid 0', messages[0][0])

    def test_open_stack_workspace_uses_source_event_mapping(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        opened = []
        messages = []
        source_event_dir = _PROJ + '/data/pick_jandy/test_stack_workspace_init_evt'
        stack_dir = stack_event_dir_for_source(source_event_dir)
        shutil.rmtree(stack_dir, ignore_errors=True)
        stack_dir.mkdir(parents=True, exist_ok=True)
        stack_trace = Trace(data=np.asarray([0.5, -0.25, 0.75], dtype=np.float32))
        stack_trace.stats.delta = 0.05
        stack_trace.write(str(stack_dir / 'existing_stack.sac'), format='SAC')
        widget.x1 = -50
        widget.x2 = 30
        widget.pending_tmarker = 't6'
        widget.suffix = '.sac'
        widget.preview_phases = ['t7', 't6']
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget._open_workspace_window = lambda wavepath: opened.append(wavepath) or object()
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                stack_mode=False,
                runtime_event_dir=source_event_dir,
                order='gcarc',
            )
        )

        try:
            widget.open_stack_workspace()

            self.assertEqual(opened[0], str(stack_dir))
            self.assertTrue(stack_event_marker_path(stack_dir).exists())
            self.assertIn('Opened stack workspace', messages[0][0])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)

    def test_open_stack_workspace_initializes_empty_workspace_without_opening(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        opened = []
        messages = []
        source_event_dir = _PROJ + '/data/pick_jandy/test_empty_stack_workspace_init_evt'
        stack_dir = stack_event_dir_for_source(source_event_dir)
        shutil.rmtree(stack_dir, ignore_errors=True)
        widget.x1 = -50
        widget.x2 = 30
        widget.pending_tmarker = 't6'
        widget.suffix = '.sac'
        widget.preview_phases = ['t7', 't6']
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget._open_workspace_window = lambda wavepath: opened.append(wavepath) or object()
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                stack_mode=False,
                runtime_event_dir=source_event_dir,
                order='gcarc',
            )
        )

        try:
            widget.open_stack_workspace()

            self.assertEqual(opened, [])
            self.assertTrue(stack_event_marker_path(stack_dir).exists())
            self.assertIn('Stack workspace not opened', messages[0][0])
            self.assertIn('no stack .sac yet', messages[0][0])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)

    def test_open_stack_workspace_does_not_open_when_only_invalid_stack_sac_exists(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        opened = []
        messages = []
        source_event_dir = _PROJ + '/data/pick_jandy/test_invalid_stack_workspace_evt'
        stack_dir = stack_event_dir_for_source(source_event_dir)
        shutil.rmtree(stack_dir, ignore_errors=True)
        stack_dir.mkdir(parents=True, exist_ok=True)
        (stack_dir / 'invalid_stack.sac').write_bytes(b'SAC')
        widget.x1 = -50
        widget.x2 = 30
        widget.pending_tmarker = 't6'
        widget.suffix = '.sac'
        widget.preview_phases = ['t7', 't6']
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget._open_workspace_window = lambda wavepath: opened.append(wavepath) or object()
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                stack_mode=False,
                runtime_event_dir=source_event_dir,
                order='gcarc',
            )
        )

        try:
            widget.open_stack_workspace()

            self.assertEqual(opened, [])
            self.assertTrue(stack_event_marker_path(stack_dir).exists())
            self.assertIn('Stack workspace not opened', messages[0][0])
            self.assertIn('no valid stack .sac yet', messages[0][0])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)

    def test_open_source_event_workspace_requires_stack_mode(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget.mpl = SimpleNamespace(wavefig=SimpleNamespace(stack_mode=False))

        widget.open_source_event_workspace()

        self.assertEqual(messages[0][0], 'Source event workspace is only available in stack mode')

    def test_open_source_event_workspace_uses_runtime_event_dir(self):
        widget = MatplotlibWidget.__new__(MatplotlibWidget)
        opened = []
        messages = []
        widget.show_status_message = lambda message, timeout_ms=3000: messages.append((message, timeout_ms))
        widget._open_workspace_window = lambda wavepath: opened.append(wavepath) or object()
        widget.mpl = SimpleNamespace(
            wavefig=SimpleNamespace(
                stack_mode=True,
                runtime_event_dir=_PROJ + '/data/pick_jandy/2011_03_06_14_32_36',
            )
        )

        widget.open_source_event_workspace()

        self.assertEqual(opened[0], _PROJ + '/data/pick_jandy/2011_03_06_14_32_36')
        self.assertIn('Opened source event workspace', messages[0][0])

    def test_resolve_stack_workspace_dir_maps_source_event_to_stack_dir(self):
        resolved = resolve_stack_workspace_dir(_PROJ + '/data/pick_jandy/2011_03_06_14_32_36')

        self.assertEqual(
            resolved,
            stack_event_dir_for_source(_PROJ + '/data/pick_jandy/2011_03_06_14_32_36'),
        )

    def test_source_event_with_stack_output_metadata_stays_source_mode(self):
        source_event_dir = Path(_PROJ + '/data/pick_jandy/test_stack_output_metadata_evt')
        stack_dir = stack_event_dir_for_source(source_event_dir)
        metadata_dir = stack_metadata_dir_for_event(source_event_dir)
        for path in (source_event_dir, stack_dir, metadata_dir):
            shutil.rmtree(path, ignore_errors=True)
        try:
            source_event_dir.mkdir(parents=True)
            marker_path = stack_event_marker_path(source_event_dir)
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(
                json.dumps(
                    {
                        "source_event_dir": str(source_event_dir),
                        "source_event_name": source_event_dir.name,
                    }
                ),
                encoding="utf-8",
            )

            self.assertTrue(marker_path.exists())
            self.assertFalse((source_event_dir / '.stack_event.json').exists())
            self.assertFalse(is_stack_event_dir(source_event_dir))
            self.assertEqual(resolve_stack_workspace_dir(source_event_dir), stack_dir)
        finally:
            for path in (source_event_dir, stack_dir, metadata_dir):
                shutil.rmtree(path, ignore_errors=True)

    def test_ensure_stack_workspace_dir_creates_marker_for_source_event(self):
        source_event_dir = _PROJ + '/data/pick_jandy/test_stack_ensure_evt'
        stack_dir = stack_event_dir_for_source(source_event_dir)
        metadata_dir = stack_metadata_dir_for_event(source_event_dir)
        shutil.rmtree(stack_dir, ignore_errors=True)
        shutil.rmtree(metadata_dir, ignore_errors=True)
        try:
            created_dir = ensure_stack_workspace_dir(source_event_dir)

            marker = load_stack_event_marker(created_dir)
            self.assertEqual(created_dir, stack_dir)
            self.assertTrue(stack_event_marker_path(created_dir).exists())
            self.assertEqual(marker['source_event_dir'], source_event_dir)
            self.assertEqual(marker['source_event_name'], 'test_stack_ensure_evt')
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)
            shutil.rmtree(metadata_dir, ignore_errors=True)

    def test_resolve_stack_workspace_dir_keeps_stack_dir(self):
        stack_dir = _PROJ + '/data/stack/pick_jandy/2011_03_06_14_32_36'

        resolved = resolve_stack_workspace_dir(stack_dir)

        self.assertEqual(
            str(resolved),
            _PROJ + '/data/output/stack/stack_files/pick_jandy/2011_03_06_14_32_36',
        )

    def test_ppk_stack_launcher_uses_stack_workspace_when_input_is_stack_dir(self):
        stack_dir = Path(_PROJ + '/data/output/stack/stack_files/pick_jandy/test_stack_launcher_evt')
        stack_dir.mkdir(parents=True, exist_ok=True)
        try:
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
            )

            config = resolve_launcher_config(args)

            self.assertEqual(
                config['wavepath'],
                str(stack_dir),
            )
            self.assertEqual(config['xlim'], [-40, 30])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)

    def test_ppk_stack_launcher_uses_stack_sac_time_window_when_available(self):
        stack_dir = Path(_PROJ + '/data/stack/pick_jandy/test_stack_launcher_window_evt')
        stack_dir.mkdir(parents=True, exist_ok=True)
        trace = Trace(data=np.zeros(3000, dtype=np.float32))
        trace.stats.delta = 0.02
        trace.stats.sac = obspy.core.AttribDict(b=0.0, e=59.98)
        trace.write(str(stack_dir / 'stack_a.sac'), format='SAC')
        try:
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
            )

            config = resolve_launcher_config(args)

            self.assertEqual(config['xlim'], [0, 60])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)
            shutil.rmtree(
                _PROJ + '/data/output/stack/stack_files/pick_jandy/test_stack_launcher_window_evt',
                ignore_errors=True,
            )

    def test_ppk_stack_launcher_requires_existing_stack_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, 'data', 'pick_jandy', 'evt1')
            source_dir.mkdir(parents=True, exist_ok=True)
            args = SimpleNamespace(
                event_path=str(source_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
            )

            with self.assertRaises(FileNotFoundError):
                resolve_launcher_config(args)

    def test_ppk_stack_open_report_blocks_empty_workspace_gui_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            config = {'wavepath': str(stack_dir)}

            report = stack_workspace_open_report(config)

            self.assertFalse(report['can_open_gui'])
            self.assertEqual(report['manifest']['stack_count'], 0)
            self.assertIn('no stack SAC files yet', report['message'])

    def test_ppk_stack_main_returns_nonzero_for_empty_workspace_gui_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )

            with patch('builtins.print') as print_mock:
                exit_code = ppk_stack_main([str(stack_dir), '-t', 't6'])

            self.assertEqual(exit_code, 2)
            printed_report = json.loads(print_mock.call_args.args[0])
            self.assertFalse(printed_report['can_open_gui'])
            self.assertIn('no stack SAC files yet', printed_report['message'])

    def test_ppk_stack_open_report_blocks_invalid_only_workspace_gui_launch(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_dir / 'bad.sac').write_bytes(b'SAC')
            config = {'wavepath': str(stack_dir)}

            report = stack_workspace_open_report(config)

            self.assertFalse(report['can_open_gui'])
            self.assertEqual(report['manifest']['invalid_stack_count'], 1)
            self.assertIn('no valid stack SAC files', report['message'])

    def test_ppk_stack_init_creates_workspace_and_manifest(self):
        source_event_dir = _PROJ + '/data/pick_jandy/test_stack_cli_init_evt'
        source_dir = Path(source_event_dir)
        stack_dir = stack_event_dir_for_source(source_event_dir)
        shutil.rmtree(source_dir, ignore_errors=True)
        shutil.rmtree(stack_dir, ignore_errors=True)
        source_dir.mkdir(parents=True, exist_ok=True)
        args = SimpleNamespace(
            event_path=source_event_dir,
            order='gcarc',
            xlim=None,
            tmarker='t6',
            suffix='.sac',
            ta_tb='t7,t6',
            xlim_preview=None,
            init=True,
            health=False,
            manifest=False,
            repair_metadata=False,
            quarantine_invalid=False,
            dry_run=False,
        )
        try:
            config = resolve_launcher_config(args)
            report = stack_workspace_maintenance_report(args, config=config)

            self.assertEqual(config['wavepath'], str(stack_dir))
            self.assertTrue(stack_event_marker_path(stack_dir).exists())
            self.assertEqual(report['manifest']['stack_count'], 0)
            self.assertFalse(report['health']['missing_marker'])
        finally:
            shutil.rmtree(stack_dir, ignore_errors=True)
            shutil.rmtree(source_dir, ignore_errors=True)

    def test_ppk_stack_refresh_index_writes_workspace_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_dir / 'stack_a.sac'), format='SAC')
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
                init=False,
                health=False,
                manifest=False,
                index=False,
                refresh_index=True,
                repair_metadata=False,
                quarantine_invalid=False,
                dry_run=False,
            )

            config = resolve_launcher_config(args)
            report = stack_workspace_maintenance_report(args, config=config)

            self.assertTrue(stack_index_path(stack_dir).exists())
            self.assertEqual(report['index']['valid_stack_count'], 1)
            self.assertEqual(report['index']['stacks'][0]['wave_name'], 'stack_a.sac')

    def test_ppk_stack_index_reports_without_writing_workspace_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
            trace.stats.delta = 0.05
            trace.write(str(stack_dir / 'stack_a.sac'), format='SAC')
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
                init=False,
                health=False,
                manifest=False,
                index=True,
                refresh_index=False,
                repair_metadata=False,
                quarantine_invalid=False,
                dry_run=False,
            )

            config = resolve_launcher_config(args)
            report = stack_workspace_maintenance_report(args, config=config)

            self.assertFalse(stack_index_path(stack_dir).exists())
            self.assertEqual(report['index']['valid_stack_count'], 1)
            self.assertEqual(report['index']['stacks'][0]['wave_name'], 'stack_a.sac')

    def test_ppk_stack_health_report_reads_existing_stack_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_dir / 'bad.sac').write_bytes(b'SAC')
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
                health=True,
                repair_metadata=False,
                quarantine_invalid=False,
                dry_run=False,
            )

            report = stack_workspace_maintenance_report(args)

            self.assertEqual(report['wavepath'], str(stack_dir.resolve()))
            self.assertFalse(report['dry_run'])
            self.assertEqual(len(report['health']['invalid_sac_files']), 1)
            self.assertIn('manifest', report)
            self.assertEqual(report['manifest']['invalid_stack_count'], 1)

    def test_ppk_stack_quarantine_dry_run_reports_without_moving(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            (stack_dir / '.stack_event.json').write_text(
                json.dumps({'mode': 'stack', 'source_event_dir': _PROJ + '/data/pick_jandy/evt1'}),
                encoding='utf-8',
            )
            (stack_dir / 'bad.sac').write_bytes(b'SAC')
            (stack_dir / 'bad.stack.json').write_text(json.dumps({'stack_wave_name': 'bad.sac'}), encoding='utf-8')
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
                health=False,
                repair_metadata=False,
                quarantine_invalid=True,
                dry_run=True,
            )

            report = stack_workspace_maintenance_report(args)

            self.assertTrue(report['dry_run'])
            self.assertEqual(report['quarantine_invalid']['invalid_count'], 1)
            self.assertTrue((stack_dir / 'bad.sac').exists())
            self.assertTrue((stack_dir / 'bad.stack.json').exists())
            self.assertEqual(len(report['health']['invalid_sac_files']), 1)

    def test_ppk_stack_repair_metadata_option_persists_normalized_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'data', 'stack', 'pick_jandy', 'evt1')
            stack_dir.mkdir(parents=True, exist_ok=True)
            marker_path = stack_dir / '.stack_event.json'
            marker_path.write_text(
                json.dumps(
                    {
                        'mode': 'stack',
                        'source_event_dir': _PROJ + '/data/pick_jandy/evt1',
                        'output_dir': '/tmp/legacy-test-output',
                    }
                ),
                encoding='utf-8',
            )
            args = SimpleNamespace(
                event_path=str(stack_dir),
                order='gcarc',
                xlim=None,
                tmarker='t6',
                suffix='.sac',
                ta_tb='t7,t6',
                xlim_preview=None,
                health=False,
                repair_metadata=True,
                quarantine_invalid=False,
                dry_run=False,
            )

            report = stack_workspace_maintenance_report(args)

            repaired_marker = json.loads(marker_path.read_text(encoding='utf-8'))
            self.assertTrue(report['repair_metadata']['marker_updated'])
            self.assertEqual(
                repaired_marker['output_dir'],
                _PROJ + '/data/output/stack/analysis/pick_jandy/evt1',
            )

    def test_apply_stack_sidecar_to_trace_restores_geometry_and_markers(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'geometry': {'gcarc_mean': 88.1, 'az_mean': 66.2, 'baz_mean': 212.4},
                'event': {'nzyear': 2011, 'nzjday': 65, 'nzhour': 14, 'nzmin': 32, 'nzsec': 36, 'evla': -56.39, 'evlo': -27.03, 'evdp': 92.0},
                'markers': {'t0': 0.0, 't6': 0.0, 't7': math.nan},
            }
        }
        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.network = 'DPK'
        trace.stats.station = 'STACK'
        trace.stats.delta = 0.02
        trace.stats.sampling_rate = 50.0
        trace.stats.sac = obspy.core.AttribDict(b=-40.0, e=-39.98)

        figure._apply_stack_sidecar_to_trace(trace, 'stack1.sac')

        self.assertAlmostEqual(float(trace.stats.sac.gcarc), 88.1)
        self.assertAlmostEqual(float(trace.stats.sac.az), 66.2)
        self.assertAlmostEqual(float(trace.stats.sac.baz), 212.4)
        self.assertAlmostEqual(float(trace.stats.sac.t0), 0.0)
        self.assertAlmostEqual(float(trace.stats.sac.t6), 0.0)
        self.assertTrue(math.isnan(float(trace.stats.sac.t7)))

    def test_apply_stack_sidecar_to_trace_preserves_manual_non_align_markers(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'align_marker': 't5',
                'markers': {'t5': 514.0, 't7': 14.14},
            }
        }
        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.network = 'DPK'
        trace.stats.station = 'STACK'
        trace.stats.delta = 0.02
        trace.stats.sampling_rate = 50.0
        trace.stats.sac = obspy.core.AttribDict(b=0.0, e=0.02)

        figure._apply_stack_sidecar_to_trace(trace, 'stack1.sac')

        self.assertAlmostEqual(float(trace.stats.sac.t5), 514.0)
        self.assertAlmostEqual(float(trace.stats.sac.t7), 14.14)

    def test_stack_auxiliary_marker_keys_add_t5_t6_partner(self):
        figure = WaveFigure.__new__(WaveFigure)

        self.assertEqual(figure._stack_auxiliary_marker_keys('5'), ['7', '6'])
        self.assertEqual(figure._stack_auxiliary_marker_keys('6'), ['7', '5'])
        self.assertEqual(figure._stack_auxiliary_marker_keys('7'), [])

    def test_stack_member_marker_mean_returns_nan_for_missing_partner_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {str(idx): {} for idx in range(10)}
        figure._preview_marker_reference_time = lambda marker_key, wave_name: math.nan

        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.dpk_wave_name = 'wave_a.sac'
        trace.stats.sac = obspy.core.AttribDict(t5=math.nan, t6=math.nan, t7=math.nan)
        evtdata = SimpleNamespace(wave_ori=[trace])

        self.assertTrue(math.isnan(figure._stack_member_marker_mean(['wave_a.sac'], '6', evtdata=evtdata)))

    def test_draw_preview_group_overlay_shows_group_numbers_only_when_enabled(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_group_overlay_enabled = True
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        figure._preview_group_color = WaveFigure._preview_group_color.__get__(figure, WaveFigure)
        fig, ax = plt.subplots()
        try:
            records = [
                SimpleNamespace(wave_name='group1/member_a.sac', longitude=10.0, latitude=20.0),
                SimpleNamespace(wave_name='group1/member_b.sac', longitude=12.0, latitude=22.0),
                SimpleNamespace(wave_name='group2/member_c.sac', longitude=30.0, latitude=40.0),
                SimpleNamespace(wave_name='ungrouped.sac', longitude=50.0, latitude=60.0),
            ]
            artists = figure._draw_preview_group_overlay(ax, records)
            texts = sorted(artist.get_text() for artist in artists if hasattr(artist, 'get_text'))
            self.assertEqual(texts, ['1', '2'])
            self.assertEqual(len(artists), 4)
        finally:
            plt.close(fig)

    def test_group_number_from_record_prefers_group_name_mapping(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)

        record = SimpleNamespace(wave_name='member_a.sac', group_name='group12')

        self.assertEqual(figure._group_number_from_record(record), 12)

    def test_group_number_from_wave_name_does_not_parse_year_as_group(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)

        self.assertIsNone(figure._group_number_from_wave_name('2022_04_28_01_07_48'))
        self.assertEqual(figure._group_number_from_wave_name('stack_group11_t6_xm40_p30.sac'), 11)

    def test_draw_preview_group_overlay_lists_group_numbers_in_lower_right_corner(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_group_overlay_enabled = True
        figure._preview_group_color = lambda group_name: '#ff0000'
        figure._group_number_from_record = lambda record: int(record.wave_name[-1])
        ax = Figure().add_subplot(111)
        records = [
            SimpleNamespace(wave_name='group1', longitude=10.0, latitude=20.0),
            SimpleNamespace(wave_name='group3', longitude=11.0, latitude=21.0),
        ]

        artists = figure._draw_preview_group_overlay(ax, records)

        scatter_artists = [artist for artist in artists if hasattr(artist, 'get_offsets')]
        text_artists = [artist for artist in artists if hasattr(artist, 'get_ha')]
        self.assertEqual(len(scatter_artists), 2)
        self.assertEqual(len(text_artists), 2)
        self.assertEqual(text_artists[0].get_text(), '1')
        self.assertEqual(text_artists[1].get_text(), '3')
        self.assertEqual(text_artists[0].get_ha(), 'right')
        self.assertEqual(text_artists[0].get_va(), 'bottom')
        self.assertEqual(text_artists[0].get_transform(), ax.transAxes)
        self.assertAlmostEqual(text_artists[0].get_position()[0], 0.98)
        self.assertAlmostEqual(text_artists[0].get_position()[1], 0.04)

    def test_apply_preview_selection_highlights_matching_group_label(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_wave_colors = lambda meta, is_selected: ('#ff5fa2' if is_selected else '#1f77b4', 1.0)
        figure._apply_preview_line_style = lambda line, meta, is_selected, line_color, line_width: None
        figure._sync_pick_highlight_from_preview_selection = lambda fig: None
        figure._preview_group_color = lambda group_name: '#00aa00'
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''
        figure.preview_trace_layout_mode = 'real'
        figure.preview_view_mode = 'wide'
        figure.stack_mode = False
        figure._normalize_preview_group_name = WaveFigure._normalize_preview_group_name.__get__(figure, WaveFigure)
        figure._group_number_from_wave_name = WaveFigure._group_number_from_wave_name.__get__(figure, WaveFigure)
        figure._group_number_from_record = WaveFigure._group_number_from_record.__get__(figure, WaveFigure)
        fig = Figure()
        axp = fig.add_subplot(111)
        label_artist = axp.text(
            0.98, 0.04, '1',
            transform=axp.transAxes,
            ha='right',
            va='bottom',
            bbox={'facecolor': 'white', 'edgecolor': '#00aa00', 'linewidth': 0.9, 'alpha': 0.92},
        )
        setattr(axp, '_dpk_group_label_artists', {1: label_artist})
        preview_state = {
            'lines': [DummyLine([0.0], [0.0])],
            'metadata': [{'wave_name': 'stack_group1.sac', 'name': 'grp1', 'gcarc': 10.0, 'az': 20.0}],
            'selected_indices': {0},
            'active_index': 0,
            'anchor_index': 0,
            'scatter': None,
            'selected_marker': None,
            'pierce_state': {
                'axes': axp,
                'records': [SimpleNamespace(wave_name='other_name.sac', group_name='group1', longitude=0.0, latitude=0.0)],
                'base_scatter': None,
                'highlight_scatter': None,
            },
            'evtdata': SimpleNamespace(gcarc=np.asarray([10.0]), az=np.asarray([20.0])),
            'info_text': SimpleNamespace(set_color=lambda value: None, set_text=lambda value: None),
            'control_status_text': None,
        }
        fig_wrapper = SimpleNamespace(_preview_state=preview_state, canvas=SimpleNamespace(draw_idle=lambda: None))

        figure._apply_preview_selection(fig_wrapper)

        self.assertEqual(label_artist.get_fontweight(), 'bold')
        facecolor = label_artist.get_bbox_patch().get_facecolor()
        self.assertAlmostEqual(facecolor[0], 0.0, places=3)
        self.assertAlmostEqual(facecolor[1], 170 / 255, places=3)
        self.assertAlmostEqual(facecolor[2], 0.0, places=3)

    def test_select_preview_ungrouped_waveforms_selects_only_unmapped_wave_names(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_group_wave_map = lambda: {'wave_a.sac': 'group1'}
        applied = {'called': False}
        figure._apply_preview_selection = lambda fig: applied.__setitem__('called', True)
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'wave_a.sac'},
                    {'wave_name': 'wave_b.sac'},
                    {'wave_name': 'wave_c.sac'},
                ],
                'selected_indices': set(),
                'active_index': 0,
                'anchor_index': 0,
            }
        )

        selected_count = figure._select_preview_ungrouped_waveforms(fig)

        self.assertEqual(selected_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {1, 2})
        self.assertTrue(applied['called'])

    def test_toggle_preview_ungrouped_only_is_one_shot_action(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_group_overlay_enabled = True
        figure.preview_ungrouped_only_enabled = True
        selected_counts = []
        statuses = []
        figure._select_preview_ungrouped_waveforms = lambda fig: selected_counts.append(2) or 2
        figure._update_preview_mode_button_styles = lambda fig: None
        figure._set_preview_search_status = lambda fig, text, color='#444444': statuses.append((text, color))

        figure._toggle_preview_ungrouped_only(SimpleNamespace(), 3)

        self.assertFalse(figure.preview_group_overlay_enabled)
        self.assertFalse(figure.preview_ungrouped_only_enabled)
        self.assertEqual(statuses[-1][0], 'Selected 2 ungrouped waveform(s)')

    def test_read_sac_syncs_stack_workspace_headers_from_sidecar_markers(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'stack_evt')
            stack_dir.mkdir()
            trace = Trace(data=np.ones(100, dtype=np.float32))
            trace.stats.network = 'DPK'
            trace.stats.station = 'STACK'
            trace.stats.channel = 'BHZ'
            trace.stats.delta = 0.05
            trace.stats.sac = obspy.core.AttribDict(
                b=0.0,
                e=4.95,
                t5=math.nan,
                t7=math.nan,
                gcarc=39.85,
                az=10.0,
                baz=20.0,
                nzyear=2002,
                nzjday=41,
                nzhour=1,
                nzmin=47,
                nzsec=7,
                evla=-55.97,
                evlo=-29.15,
                evdp=198.0,
            )
            trace.write(str(stack_dir / 'stack_group3_t5_xm50_p20.sac'), format='SAC')

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(stack_dir)
            figure.stack_mode = True
            figure.stack_event_marker = {}
            figure.stack_sidecars = {
                'stack_group3_t5_xm50_p20.sac': {
                    'markers': {'t5': 21.0, 't7': 11.0},
                }
            }
            figure.suffix = '.sac'
            figure.maxidx = 5
            figure.tmarker = 't5'
            figure.fig = Figure()
            figure.markers = {str(idx): {} for idx in range(10)}
            figure.user_markers = {
                key: {}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }

            figure.read_sac(order='gcarc')

            synced_trace = obspy.read(str(stack_dir / 'stack_group3_t5_xm50_p20.sac'), headonly=True)[0]
            self.assertAlmostEqual(float(synced_trace.stats.sac.t5), 21.0)
            self.assertAlmostEqual(float(synced_trace.stats.sac.t7), 11.0)

    def test_stack_wave_summary_formats_scope_type_normalize_and_count(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }

        summary = figure._stack_wave_summary('stack1.sac')

        self.assertEqual(summary, 'group:group2 | linear | rms | N=27')

    def test_stack_wave_summary_returns_empty_for_non_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = False
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }

        summary = figure._stack_wave_summary('stack1.sac')

        self.assertEqual(summary, '')

    def test_wave_display_name_appends_stack_summary_in_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }

        display_name = figure._wave_display_name('stack1.sac', 'DPK.STACK')

        self.assertEqual(display_name, 'DPK.STACK [group:group2 | linear | rms | N=27]')

    def test_remember_pick_wave_uses_stack_aware_display_name(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }
        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.network = 'DPK'
        trace.stats.station = 'STACK'
        figure.wave = [trace]
        figure.ori_sacnames = ['stack1.sac']

        figure._remember_pick_wave(0)

        self.assertEqual(figure.current_pick_wave_name, 'stack1.sac')
        self.assertEqual(figure.current_pick_station_name, 'DPK.STACK [group:group2 | linear | rms | N=27]')

    def test_stack_wave_summary_describes_group_and_stack_type(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }
        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.network = 'DPK'
        trace.stats.station = 'STACK'
        trace.stats.delta = 0.02
        trace.stats.sac = obspy.core.AttribDict(gcarc=88.1, baz=212.4)
        trace.stats.dpk_wave_name = 'stack1.sac'

        trace_metadata = [{
            'name': f"{trace.stats.network}.{trace.stats.station}",
            'gcarc': float(trace.stats.sac.gcarc),
            'baz': float(trace.stats.sac.baz),
            'wave_name': getattr(trace.stats, 'dpk_wave_name', ''),
            'stack_summary': figure._stack_wave_summary(getattr(trace.stats, 'dpk_wave_name', '')),
        }]

        self.assertEqual(trace_metadata[0]['stack_summary'], 'group:group2 | linear | rms | N=27')

    def test_preview_pierce_points_falls_back_to_stack_sidecar_average_location(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'geometry': {
                    'pierce_lon_mean': -27.1,
                    'pierce_lat_mean': -56.4,
                }
            }
        }
        figure.preview_pierce_cache = {}
        figure._load_pierce_points_for_current_event = lambda auto_generate=False, phase=None, model=None: {}
        preview_state = {
            'metadata': [{'wave_name': 'stack1.sac'}],
            'selected_indices': {0},
        }

        records = figure._preview_pierce_points(preview_state, selected_only=False)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].wave_name, 'stack1.sac')
        self.assertAlmostEqual(records[0].longitude, -27.1)
        self.assertAlmostEqual(records[0].latitude, -56.4)

    def test_standard_export_pierce_records_falls_back_to_stack_sidecar_average_location(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'geometry': {
                    'pierce_lon_mean': -27.2,
                    'pierce_lat_mean': -56.5,
                }
            }
        }
        figure._load_pierce_points_for_current_event = lambda auto_generate=True, phase=None, model=None: {}
        trace = Trace(data=np.asarray([1.0, 2.0], dtype=np.float32))
        trace.stats.dpk_wave_name = 'stack1.sac'
        evtdata = SimpleNamespace(wave_ori=[trace])

        records = figure._standard_export_pierce_records_from_evtdata(evtdata)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].wave_name, 'stack1.sac')
        self.assertAlmostEqual(records[0].longitude, -27.2)
        self.assertAlmostEqual(records[0].latitude, -56.5)

    def test_standard_pierce_record_label_marks_stack_average_point(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }

        label = figure._standard_pierce_record_label('stack1.sac')

        self.assertEqual(label, 'STACK')

    def test_standard_pierce_record_label_omits_non_stack_points(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = False
        figure.stack_sidecars = {
            'stack1.sac': {
                'scope': 'group:group2',
                'stack_type': 'linear',
                'normalize': 'rms',
                'wave_count_used': 27,
            }
        }

        label = figure._standard_pierce_record_label('stack1.sac')

        self.assertEqual(label, '')

    def test_save_standard_pierce_plot_uses_python_renderer_without_external_script(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = False
        figure.preview_pierce_phase = 'pP'
        figure.preview_pierce_model = 'iasp91'
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure._is_user1_wave = lambda wave_name: False
        figure._is_user5_wave = lambda wave_name: False
        figure._is_preview_purple_wave = lambda wave_name: False
        figure._standard_export_pierce_records_from_evtdata = lambda evtdata: [
            SimpleNamespace(wave_name='wave_a.sac', longitude=-26.2, latitude=-58.3),
            SimpleNamespace(wave_name='wave_b.sac', longitude=-25.9, latitude=-58.1),
        ]
        figure._standard_export_header_lines = lambda evtdata, export_options=None: ['evt:test']
        figure._standard_pierce_record_color = lambda wave_name: '#111111'
        figure._standard_pierce_record_label = lambda wave_name: ''
        figure._is_user4_wave = lambda wave_name: False
        evtdata = SimpleNamespace(sta_num=2, evlo=-26.0, evla=-58.0, wave_ori=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path, pierce_count, zoom_path = figure._save_standard_pierce_plot(
                evtdata,
                temp_dir,
                '20260624_000000',
            )

            self.assertEqual(pierce_count, 2)
            self.assertTrue(Path(output_path).is_file())
            self.assertTrue(Path(zoom_path).is_file())

    def test_save_standard_pierce_group_plot_colors_records_by_group(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.preview_pierce_phase = 'pP'
        figure.preview_pierce_model = 'iasp91'
        figure.plotfig = None
        figure.standard_export_options = {'event_name': True}
        figure._standard_export_header_lines = lambda evtdata, export_options=None: ['evt:test']
        figure._current_stack_preview_wave_name = lambda: 'stack_group3_t6_xm40_p20.sac'
        # wave_a is a stack trace whose name embeds group 3; wave_b is a member
        # trace that should inherit the previewed group (3) when its name has none.
        records = [
            SimpleNamespace(wave_name='stack_group3_t6_xm40_p20.sac', longitude=-26.2, latitude=-58.3),
            SimpleNamespace(wave_name='member_wave.sac', longitude=-25.9, latitude=-58.1),
        ]
        evtdata = SimpleNamespace(sta_num=2, evlo=-26.0, evla=-58.0, wave_ori=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = figure._save_standard_pierce_group_plot(
                evtdata,
                records,
                temp_dir,
                '20260624_000000',
            )

            self.assertIsNotNone(output_path)
            self.assertTrue(Path(output_path).is_file())
            self.assertIn('pierce_group_', Path(output_path).name)
            # Both records resolve to group 3 (stack name embeds it; member inherits it).
            self.assertEqual(figure._standard_pierce_group_number('stack_group3_t6_xm40_p20.sac'), 3)
            self.assertEqual(figure._standard_pierce_group_number('member_wave.sac'), 3)
            # Group color comes from the shared palette.
            self.assertEqual(
                figure._preview_group_color('group3'),
                figure._preview_group_color('group3'),
            )

    def test_save_standard_pierce_plot_reuses_cached_outputs(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = False
        figure.preview_pierce_phase = 'pP'
        figure.preview_pierce_model = 'iasp91'
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure._is_user1_wave = lambda wave_name: False
        figure._is_user5_wave = lambda wave_name: False
        figure._is_preview_purple_wave = lambda wave_name: False
        figure._is_user4_wave = lambda wave_name: False
        figure._standard_export_pierce_records_from_evtdata = lambda evtdata: [
            SimpleNamespace(wave_name='wave_a.sac', longitude=-26.2, latitude=-58.3),
        ]
        figure._standard_export_header_lines = lambda evtdata, export_options=None: ['evt:test']
        figure._standard_pierce_record_label = lambda wave_name: ''
        figure._semantic_event_dir = lambda: '/tmp/pick_x/evt1'
        evtdata = SimpleNamespace(sta_num=1, evlo=-26.0, evla=-58.0, wave_ori=[])

        with tempfile.TemporaryDirectory() as temp_dir:
            signature = figure._standard_pierce_cache_signature(evtdata, figure._standard_export_pierce_records_from_evtdata(evtdata))
            cache_main = Path(temp_dir, f'pierce_cache_{signature}.png')
            cache_zoom = Path(temp_dir, f'pierce_zoom_cache_{signature}.png')
            cache_main.write_bytes(b'PNG')
            cache_zoom.write_bytes(b'PNG')

            output_path, pierce_count, zoom_path = figure._save_standard_pierce_plot(
                evtdata,
                temp_dir,
                '20260624_000001',
            )

            self.assertEqual(pierce_count, 1)
            self.assertTrue(Path(output_path).is_file())
            self.assertTrue(Path(zoom_path).is_file())
            self.assertEqual(Path(output_path).read_bytes(), b'PNG')
            self.assertEqual(Path(zoom_path).read_bytes(), b'PNG')

    def _preview_action_figure(self, line, curve_points, selected_indices=None):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {
            '0': ('t0', '#d62728'),
            '7': ('t7', '#d62728'),
        }
        figure.markers = {
            '0': {'waveA': 50.0},
            '7': {'waveA': 55.0},
        }
        figure.ori_sacnames = ['waveA']
        figure.filenames = ['NET.STA']
        figure.t0 = np.asarray([50.0], dtype=float)
        figure.t7 = np.asarray([55.0], dtype=float)
        figure.tmarker = 't0'
        figure.tmarker_t = np.asarray([50.0], dtype=float)
        figure.preview_modes = [['0', -10.0, 10.0]]
        figure.preview_peak_half_window_default = 1.0
        figure.current_pick_wave_name = None
        figure.current_pick_station_name = None
        figure._refresh_preview_figure = lambda fig, preview_index: None
        figure._refresh_pick_window_if_available = lambda focus_current_wave=True: None
        figure._set_preview_search_status = lambda fig, text, color=None: setattr(fig, 'last_status', (text, color))

        fig = SimpleNamespace(
            _preview_state={
                'tmarker': '0',
                'metadata': [{'wave_name': 'waveA', 'name': 'NET.STA'}],
                'selected_indices': set({0} if selected_indices is None else selected_indices),
                'active_index': 0,
                'y_values': np.asarray([0.0], dtype=float),
                'lines': [line],
            },
            _preview_curve_pick={
                'active': False,
                'finished': True,
                'points': list(curve_points),
                'artist': None,
            },
            _preview_controls={},
        )
        return figure, fig

    def test_parse_preview_curve_pick_request_accepts_plain_digit_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'7': ('P', '#000000')}

        marker_key, half_window, mode, error_message = figure._parse_preview_curve_pick_request('7', default_half_window=1.0)

        self.assertEqual(marker_key, '7')
        self.assertEqual(half_window, 1.0)
        self.assertEqual(mode, 'pk')
        self.assertIsNone(error_message)

    def test_parse_preview_curve_pick_request_accepts_t_prefixed_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'7': ('P', '#000000')}

        marker_key, half_window, mode, error_message = figure._parse_preview_curve_pick_request('t7 0.5', default_half_window=1.0)

        self.assertEqual(marker_key, '7')
        self.assertEqual(half_window, 0.5)
        self.assertEqual(mode, 'pk')
        self.assertIsNone(error_message)

    def test_preview_line_peak_time_near_refines_to_parabolic_vertex(self):
        figure = WaveFigure.__new__(WaveFigure)
        xdata = np.array([-1.0, -0.4, 0.2, 0.8, 1.4], dtype=float)
        ydata = -(xdata - 0.5) ** 2 + 1.0
        line = DummyLine(xdata, ydata)

        peak_time = figure._preview_line_peak_time_near(
            line,
            baseline_y=0.0,
            reference_time=10.0,
            center_x=0.5,
            half_window_seconds=1.0,
        )

        self.assertAlmostEqual(peak_time, 10.5, places=2)

    def test_preview_line_peak_time_near_refines_to_parabolic_trough(self):
        figure = WaveFigure.__new__(WaveFigure)
        xdata = np.array([-1.0, -0.4, 0.2, 0.8, 1.4], dtype=float)
        ydata = (xdata - 0.5) ** 2 - 1.0
        line = DummyLine(xdata, ydata)

        peak_time = figure._preview_line_peak_time_near(
            line,
            baseline_y=0.0,
            reference_time=20.0,
            center_x=0.5,
            half_window_seconds=1.0,
        )

        self.assertAlmostEqual(peak_time, 20.5, places=2)

    def test_preview_line_peak_time_near_visual_prefers_nearest_local_extremum_over_far_larger_one(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0, 1.4],
            [0.0, 0.3, 0.0, 0.7, 0.0, 1.4, 0.0],
        )

        peak_time = figure._preview_line_peak_time_near_visual(
            line=line,
            baseline_y=0.0,
            reference_time=50.0,
            center_x=0.15,
            half_window_seconds=1.5,
        )

        self.assertAlmostEqual(peak_time, 50.2, places=1)

    def test_preview_line_peak_time_near_visual_prefers_peak_when_reference_is_above_baseline(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
            [-0.6, -0.2, 0.15, 0.55, 0.1, -0.7],
        )

        peak_time = figure._preview_line_peak_time_near_visual(
            line=line,
            baseline_y=0.0,
            reference_time=80.0,
            center_x=0.18,
            half_window_seconds=0.5,
        )

        self.assertAlmostEqual(peak_time, 80.2, places=1)

    def test_preview_line_peak_time_near_visual_prefers_trough_when_reference_is_below_baseline(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6],
            [0.5, 0.15, -0.1, -0.55, -0.05, 0.7],
        )

        peak_time = figure._preview_line_peak_time_near_visual(
            line=line,
            baseline_y=0.0,
            reference_time=90.0,
            center_x=0.18,
            half_window_seconds=0.5,
        )

        self.assertAlmostEqual(peak_time, 90.2, places=1)

    def test_preview_line_peak_time_near_refines_small_local_peak_near_reference(self):
        figure = WaveFigure.__new__(WaveFigure)
        xdata = np.array([-0.3, -0.15, 0.0, 0.15, 0.3], dtype=float)
        ydata = -(xdata - 0.03) ** 2 + 0.2
        line = DummyLine(xdata, ydata)

        peak_time = figure._preview_line_peak_time_near(
            line=line,
            baseline_y=0.0,
            reference_time=30.0,
            center_x=0.03,
            half_window_seconds=0.3,
        )

        self.assertAlmostEqual(peak_time, 30.03, places=2)

    def test_standard_phase_legend_uses_latin_subscript_for_actual_phases(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.phase_display_labels = {
            '0': 'P',
            '6': 'pP',
            '7': 'P',
        }
        figure.phase_label_prefixes = {
            '0': 'Theory',
            '6': 'Actual',
            '7': 'Actual',
        }

        self.assertEqual(figure._standard_phase_legend_label('0'), r'P$_{t}$')
        self.assertEqual(figure._standard_phase_legend_label('6'), r'pP$_{a}$')
        self.assertEqual(figure._standard_phase_legend_label('7'), r'P$_{a}$')

    def test_standard_phase_label_placements_stack_close_labels_from_cluster_top(self):
        figure = WaveFigure.__new__(WaveFigure)
        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(-10, 70)
        ax.set_ylim(77, 82)
        label_specs = [
            {'text': 'sP$_{t}$', 'x': 31.0, 'y': 81.0, 'color': 'orange'},
            {'text': 'sP$_{a}$', 'x': 35.0, 'y': 81.2, 'color': 'brown'},
            {'text': 'P$_{t}$', 'x': 0.0, 'y': 80.0, 'color': 'red'},
        ]

        placements = figure._standard_phase_label_placements(ax, label_specs, xmin=-10, xmax=70)

        close_labels = [item for item in placements if item['text'].startswith('sP')]
        far_label = [item for item in placements if item['text'].startswith('P')][0]
        self.assertEqual([item['lane'] for item in close_labels], [0, 1])
        self.assertGreater(close_labels[1]['label_y'], close_labels[0]['label_y'])
        self.assertGreater(close_labels[0]['label_y'], 81.2)
        self.assertGreater(far_label['label_y'], 80.0)

    def test_standard_phase_label_placements_do_not_chain_far_labels(self):
        figure = WaveFigure.__new__(WaveFigure)
        fig = Figure(figsize=(4, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_xlim(-10, 70)
        ax.set_ylim(68, 73)
        label_specs = [
            {'text': 'pP$_{t}$', 'x': 19.5, 'y': 71.0, 'color': 'blue'},
            {'text': 'pP$_{a}$', 'x': 23.5, 'y': 71.3, 'color': 'purple'},
            {'text': 'pmP$_{t}$', 'x': 27.5, 'y': 71.2, 'color': 'cyan'},
            {'text': 'sP$_{t}$', 'x': 31.5, 'y': 71.4, 'color': 'green'},
        ]

        placements = figure._standard_phase_label_placements(ax, label_specs, xmin=-10, xmax=70)

        p_labels = placements[:2]
        later_labels = placements[2:]
        self.assertEqual([item['lane'] for item in p_labels], [0, 1])
        self.assertEqual([item['lane'] for item in later_labels], [0, 1])
        self.assertLess(later_labels[1]['label_y'] - later_labels[0]['label_y'], 0.35)

    def test_preview_marker_reference_time_reads_marker_only(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {
            '0': {'waveA': 12.5},
        }

        self.assertEqual(figure._preview_marker_reference_time('t0', 'waveA'), 12.5)

    def test_preview_relative_phase_time_uses_marker_times(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {
            '0': {'waveA': 12.5},
            '7': {'waveA': 15.0},
        }

        relative_time = figure._preview_relative_phase_time('t0', '7', 'waveA')

        self.assertEqual(relative_time, 2.5)

    def test_preview_relative_phase_time_uses_frozen_alignment_reference(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {
            '7': {'waveA': 55.4},
        }

        relative_time = figure._preview_relative_phase_time(
            '7',
            '7',
            'waveA',
            reference_times={'waveA': 55.0},
        )

        self.assertAlmostEqual(relative_time, 0.4)

    def test_preview_line_peak_time_near_uses_displayed_waveform_amplitude(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [-0.2, -0.1, 0.0, 0.1, 0.2],
            [10.1, 10.4, 10.2, 9.1, 10.3],
        )

        peak_time = figure._preview_line_peak_time_near(
            line=line,
            baseline_y=10.0,
            reference_time=100.0,
            center_x=0.0,
            half_window_seconds=0.2,
        )

        self.assertAlmostEqual(peak_time, 100.098, places=3)

    def test_preview_curve_wave_intersection_x_prefers_intersection_near_curve_center(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 0.0],
        )
        curve_points = [
            (0.8, 1.2),
            (1.8, 1.2),
        ]

        intersection_x = figure._preview_curve_wave_intersection_x(line, curve_points)

        self.assertAlmostEqual(intersection_x, 1.4)

    def test_preview_curve_wave_intersection_x_uses_first_curve_contact(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 0.0],
        )
        curve_points = [
            (0.0, 1.2),
            (2.0, 1.2),
        ]

        intersection_x = figure._preview_curve_wave_intersection_x(line, curve_points)

        self.assertAlmostEqual(intersection_x, 0.6)

    def test_wave_polarity_factor_uses_user4_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user_markers = {
            'user4': {'waveA': -1.0},
        }

        self.assertEqual(figure._wave_polarity_factor('waveA'), -1.0)
        self.assertEqual(figure._wave_polarity_factor('waveB'), 1.0)

    def test_preview_wave_colors_prioritize_user4(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user4_mark_color = '#ff9f1c'
        figure.user4_selected_color = '#ffd166'
        figure.user1_mark_color = '#008f5a'
        figure.user1_selected_color = '#ffd400'
        figure.preview_mark_color = '#9a1fff'
        figure.preview_selected_mark_color = '#d2691e'

        color, width = figure._preview_wave_colors({'is_user4_marked': True, 'is_user5_marked': True}, False)

        self.assertEqual(color, '#ff9f1c')
        self.assertAlmostEqual(width, 0.95)

    def test_toggle_user4_marker_clears_lower_tier_user_states(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user_markers = {
            'user1': {},
            'user2': {'waveA': 1.0},
            'user3': {},
            'user4': {},
            'user5': {'waveA': 1.0},
        }

        changed = figure._toggle_user4_marker('waveA')

        self.assertTrue(changed)
        self.assertFalse(math.isnan(figure.user_markers['user4']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user2']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user5']['waveA']))

    def test_set_user_marker_keeps_low_tier_states_mutually_exclusive(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user_markers = {
            'user1': {},
            'user2': {},
            'user3': {},
            'user4': {},
            'user5': {'waveA': 1.0},
        }

        changed = figure._set_user_marker('waveA', 'user2', True)

        self.assertTrue(changed)
        self.assertFalse(math.isnan(figure.user_markers['user2']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user5']['waveA']))

    def test_select_preview_user4_waveforms_selects_flipped_entries(self):
        figure = WaveFigure.__new__(WaveFigure)
        selected_wave_names = []
        figure._update_preview_selection_by_wave_names = lambda fig, wave_names: selected_wave_names.extend(wave_names)
        figure._apply_preview_selection = lambda fig: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_user4_marked': True},
                    {'wave_name': 'waveB', 'is_user4_marked': False},
                    {'wave_name': 'waveC', 'is_user4_marked': True},
                ]
            }
        )

        count = figure._select_preview_user4_waveforms(fig)

        self.assertEqual(count, 2)
        self.assertEqual(selected_wave_names, ['waveA', 'waveC'])

    def test_toggle_preview_selection_by_selected_user_state_repeats_complement(self):
        figure = WaveFigure.__new__(WaveFigure)
        apply_calls = []
        figure._apply_preview_selection = lambda fig: apply_calls.append(set(fig._preview_state['selected_indices']))
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_user1_marked': True},
                    {'wave_name': 'waveB', 'is_user1_marked': False},
                    {'wave_name': 'waveC', 'is_user1_marked': True},
                    {'wave_name': 'waveD', 'is_user5_marked': True},
                ],
                'selected_indices': {0},
            }
        )

        first_count, first_label, first_scope, first_error = figure._toggle_preview_selection_by_selected_user_states(fig)
        self.assertEqual(first_error, None)
        self.assertEqual(first_label, 'user1')
        self.assertEqual(first_scope, 'matching')
        self.assertEqual(first_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {0, 2})

        second_count, second_label, second_scope, second_error = figure._toggle_preview_selection_by_selected_user_states(fig)
        self.assertEqual(second_error, None)
        self.assertEqual(second_label, 'user1')
        self.assertEqual(second_scope, 'complement')
        self.assertEqual(second_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {1, 3})

        third_count, third_label, third_scope, third_error = figure._toggle_preview_selection_by_selected_user_states(fig)
        self.assertEqual(third_error, None)
        self.assertEqual(third_label, 'user1')
        self.assertEqual(third_scope, 'matching')
        self.assertEqual(third_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {0, 2})
        self.assertEqual(apply_calls, [{0, 2}, {1, 3}, {0, 2}])

    def test_toggle_preview_selection_by_multiple_selected_user_states(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._apply_preview_selection = lambda fig: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_user1_marked': True},
                    {'wave_name': 'waveB', 'is_marked_m': True},
                    {'wave_name': 'waveC', 'is_user4_marked': True},
                    {'wave_name': 'waveD', 'is_user5_marked': True},
                    {'wave_name': 'waveE'},
                ],
                'selected_indices': {0, 3},
            }
        )

        selected_count, state_label, target_scope, error_message = figure._toggle_preview_selection_by_selected_user_states(fig)

        self.assertIsNone(error_message)
        self.assertEqual(state_label, 'user1 + user5')
        self.assertEqual(target_scope, 'matching')
        self.assertEqual(selected_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {0, 3})

    def test_toggle_preview_selection_uses_only_selected_primary_user_states(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._apply_preview_selection = lambda fig: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_user1_marked': True, 'is_user4_marked': True},
                    {'wave_name': 'waveB', 'is_user1_marked': True},
                    {'wave_name': 'waveC', 'is_user4_marked': True},
                    {'wave_name': 'waveD', 'is_user5_marked': True},
                ],
                'selected_indices': {0},
            }
        )

        selected_count, state_label, target_scope, error_message = figure._toggle_preview_selection_by_selected_user_states(fig)

        self.assertIsNone(error_message)
        self.assertEqual(state_label, 'user1')
        self.assertEqual(target_scope, 'matching')
        self.assertEqual(selected_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {0, 1})

    def test_toggle_preview_selection_prioritizes_user4_over_lower_states(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._apply_preview_selection = lambda fig: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_user4_marked': True, 'is_user5_marked': True},
                    {'wave_name': 'waveB', 'is_user4_marked': True},
                    {'wave_name': 'waveC', 'is_user5_marked': True},
                ],
                'selected_indices': {0},
            }
        )

        selected_count, state_label, target_scope, error_message = figure._toggle_preview_selection_by_selected_user_states(fig)

        self.assertIsNone(error_message)
        self.assertEqual(state_label, 'user4')
        self.assertEqual(target_scope, 'matching')
        self.assertEqual(selected_count, 2)
        self.assertEqual(fig._preview_state['selected_indices'], {0, 1})

    def test_toggle_selected_preview_user1_repeats_apply_and_clear(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user_markers = {
            'user1': {},
            'user2': {'waveA': 1.0},
            'user3': {},
            'user4': {},
            'user5': {'waveA': 1.0},
        }
        figure._apply_preview_selection = lambda fig: None
        figure._refresh_pick_window_if_available = lambda *args, **kwargs: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_marked_m': True, 'is_user5_marked': True},
                ],
                'selected_indices': {0},
            }
        )

        first_count, first_enabled = figure._toggle_selected_preview_user1(fig)
        self.assertEqual((first_count, first_enabled), (1, True))
        self.assertFalse(math.isnan(figure.user_markers['user1']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user2']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user5']['waveA']))
        self.assertTrue(fig._preview_state['metadata'][0]['is_user1_marked'])
        self.assertFalse(fig._preview_state['metadata'][0]['is_marked_m'])
        self.assertFalse(fig._preview_state['metadata'][0]['is_user5_marked'])

        second_count, second_enabled = figure._toggle_selected_preview_user1(fig)
        self.assertEqual((second_count, second_enabled), (1, False))
        self.assertTrue(math.isnan(figure.user_markers['user1']['waveA']))
        self.assertFalse(fig._preview_state['metadata'][0]['is_user1_marked'])

    def test_toggle_selected_preview_user5_repeats_apply_and_clear(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.user_markers = {
            'user1': {},
            'user2': {'waveA': 1.0},
            'user3': {},
            'user4': {},
            'user5': {},
        }
        figure._apply_preview_selection = lambda fig: None
        figure._refresh_pick_window_if_available = lambda *args, **kwargs: None
        fig = SimpleNamespace(
            _preview_state={
                'metadata': [
                    {'wave_name': 'waveA', 'is_marked_m': True},
                    {'wave_name': 'waveB', 'is_user5_marked': True},
                ],
                'selected_indices': {0},
            }
        )

        first_count, first_enabled = figure._toggle_selected_preview_user5(fig)
        self.assertEqual((first_count, first_enabled), (1, True))
        self.assertFalse(math.isnan(figure.user_markers['user5']['waveA']))
        self.assertTrue(math.isnan(figure.user_markers['user2']['waveA']))
        self.assertTrue(fig._preview_state['metadata'][0]['is_user5_marked'])

        second_count, second_enabled = figure._toggle_selected_preview_user5(fig)
        self.assertEqual((second_count, second_enabled), (1, False))
        self.assertTrue(math.isnan(figure.user_markers['user5']['waveA']))
        self.assertFalse(fig._preview_state['metadata'][0]['is_user5_marked'])

    def test_preview_curve_peak_uses_first_contact_when_curve_crosses_same_wave_twice(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [0.5, 0.6, 0.7, 1.3, 1.4, 1.5],
            [0.0, 1.5, 0.0, 0.0, 2.0, 0.0],
        )
        curve_points = [
            (0.6, -0.5),
            (0.6, 2.5),
            (1.4, 2.5),
            (1.4, -0.5),
        ]

        marker_time = figure._curve_marker_time_for_preview_line(
            line=line,
            baseline_y=0.0,
            reference_time=50.0,
            curve_points=curve_points,
            half_window=0.2,
        )

        self.assertEqual(marker_time, 50.6)

    def test_preview_curve_peak_pick_writes_peak_time_to_marker_state(self):
        line = DummyLine(
            [5.2, 5.3, 5.4, 5.5],
            [0.0, 0.0, 1.0, 0.0],
        )
        curve_points = [(5.3, -1.0), (5.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)

        applied = figure._apply_preview_curve_peak_pick(fig, preview_index=0, request_text='7')

        self.assertTrue(applied)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)
        self.assertEqual(fig._preview_curve_pick['points'], [])
        self.assertIn('Picked t7', fig.last_status[0])

    def test_preview_curve_peak_pick_uses_frozen_preview_alignment_reference(self):
        line = DummyLine(
            [0.2, 0.3, 0.4, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        )
        curve_points = [(0.3, -1.0), (0.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)
        figure.markers['7']['waveA'] = 55.4
        figure.t7[0] = 55.4
        figure.tmarker = 't7'
        figure.tmarker_t[0] = 55.4
        figure.preview_modes = [['7', -10.0, 10.0]]
        fig._preview_state['tmarker'] = '7'
        fig._preview_state['reference_times'] = {'waveA': 55.0}

        applied = figure._apply_preview_curve_peak_pick(fig, preview_index=0, request_text='7')

        self.assertTrue(applied)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)
        self.assertEqual(float(figure.tmarker_t[0]), 55.4)

    def test_preview_curve_peak_pick_does_not_move_current_alignment_reference(self):
        line = DummyLine(
            [0.2, 0.3, 0.4, 0.5],
            [0.0, 0.0, 1.0, 0.0],
        )
        curve_points = [(0.3, -1.0), (0.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)
        figure.tmarker = 't7'
        figure.tmarker_t[0] = 55.0
        figure.preview_modes = [['7', -10.0, 10.0]]
        fig._preview_state['tmarker'] = '7'
        fig._preview_state['reference_times'] = {'waveA': 55.0}

        applied = figure._apply_preview_curve_peak_pick(fig, preview_index=0, request_text='7')

        self.assertTrue(applied)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)
        self.assertEqual(float(figure.tmarker_t[0]), 55.0)

    def test_preview_curve_alignment_writes_current_preview_marker_state(self):
        line = DummyLine(
            [-0.2, 0.3, 0.45, 0.9],
            [0.0, 0.2, -1.5, 3.0],
        )
        curve_points = [(0.3, -1.0), (0.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)

        applied = figure._apply_preview_curve_alignment(fig, preview_index=0)

        self.assertTrue(applied)
        self.assertEqual(figure.markers['0']['waveA'], 50.45)
        self.assertEqual(float(figure.t0[0]), 50.45)
        self.assertEqual(float(figure.tmarker_t[0]), 50.0)
        self.assertEqual(fig._preview_curve_pick['points'], [])
        self.assertIn('Aligned t0', fig.last_status[0])

    def test_preview_peak_pick_uses_visible_waveforms_when_selection_is_empty(self):
        line = DummyLine(
            [-0.1, 0.0, 0.1],
            [0.0, 0.0, 1.0],
        )
        figure, fig = self._preview_action_figure(line, curve_points=[], selected_indices=[])

        applied = figure._apply_preview_reference_peak_pick(fig, preview_index=0, request_text='7')

        self.assertTrue(applied)
        self.assertEqual(figure.markers['7']['waveA'], 50.1)
        self.assertIn('visible', fig.last_status[0])

    def test_preview_peak_action_does_not_fallback_to_zero_when_curve_is_unfinished(self):
        line = DummyLine(
            [-0.1, 0.0, 0.1],
            [0.0, 0.0, 1.0],
        )
        figure, fig = self._preview_action_figure(line, curve_points=[(5.3, 0.0)], selected_indices=[])
        fig._preview_curve_pick['active'] = True
        fig._preview_curve_pick['finished'] = False

        applied = figure._apply_preview_peak_action(fig, preview_index=0, request_text='7')

        self.assertFalse(applied)
        self.assertEqual(figure.markers['7']['waveA'], 55.0)
        self.assertIn('finish the P curve', fig.last_status[0])

    def test_preview_peak_action_ignores_duplicate_submit_after_curve_pick(self):
        line = DummyLine(
            [-0.1, 0.0, 0.15, 5.2, 5.3, 5.4, 5.5],
            [0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        )
        curve_points = [(5.3, -1.0), (5.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)

        first_submit = figure._apply_preview_peak_action(fig, preview_index=0, request_text='7')
        second_submit = figure._apply_preview_peak_action(fig, preview_index=0, request_text='7')

        self.assertTrue(first_submit)
        self.assertFalse(second_submit)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)

    def test_set_wave_marker_time_updates_marker_array(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {'7': {'waveA': 55.0}}
        figure.ori_sacnames = ['waveA']
        figure.t7 = np.asarray([55.0], dtype=float)
        figure.tmarker = 't0'
        figure.tmarker_t = np.asarray([50.0], dtype=float)

        changed = figure._set_wave_marker_time('waveA', '7', 55.4)

        self.assertTrue(changed)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)

    def test_set_wave_marker_time_can_leave_alignment_reference_unchanged(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.markers = {'7': {'waveA': 55.0}}
        figure.ori_sacnames = ['waveA']
        figure.t7 = np.asarray([55.0], dtype=float)
        figure.tmarker = 't7'
        figure.tmarker_t = np.asarray([55.0], dtype=float)

        changed = figure._set_wave_marker_time(
            'waveA',
            '7',
            55.4,
            update_alignment_reference=False,
        )

        self.assertTrue(changed)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(float(figure.t7[0]), 55.4)
        self.assertEqual(float(figure.tmarker_t[0]), 55.0)

    def test_relative_display_uses_current_view_reference_not_updated_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.axis_mode = 'relative'
        figure.tmarker_t = np.asarray([55.0], dtype=float)
        figure.t0 = np.asarray([50.0], dtype=float)

        display_x = figure._display_x_value(55.4, wave_index=0)

        self.assertAlmostEqual(display_x, 0.4)

    def test_relative_reference_time_falls_back_to_first_available_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.tmarker_t = np.asarray([math.nan], dtype=float)
        figure.t0 = np.asarray([math.nan], dtype=float)
        figure.t1 = np.asarray([math.nan], dtype=float)
        figure.t2 = np.asarray([math.nan], dtype=float)
        figure.t3 = np.asarray([33.3], dtype=float)
        figure.t4 = np.asarray([math.nan], dtype=float)
        figure.t5 = np.asarray([55.5], dtype=float)
        figure.t6 = np.asarray([math.nan], dtype=float)
        figure.t7 = np.asarray([math.nan], dtype=float)
        figure.t8 = np.asarray([math.nan], dtype=float)
        figure.t9 = np.asarray([math.nan], dtype=float)

        reference_time = figure._relative_reference_time_for_wave(0)

        self.assertEqual(reference_time, 33.3)

    def test_relative_reference_time_returns_nan_when_no_marker_exists(self):
        figure = WaveFigure.__new__(WaveFigure)
        nan_array = np.asarray([math.nan], dtype=float)
        figure.tmarker_t = nan_array.copy()
        figure.t0 = nan_array.copy()
        figure.t1 = nan_array.copy()
        figure.t2 = nan_array.copy()
        figure.t3 = nan_array.copy()
        figure.t4 = nan_array.copy()
        figure.t5 = nan_array.copy()
        figure.t6 = nan_array.copy()
        figure.t7 = nan_array.copy()
        figure.t8 = nan_array.copy()
        figure.t9 = nan_array.copy()

        reference_time = figure._relative_reference_time_for_wave(0)

        self.assertTrue(math.isnan(reference_time))

    def test_bandpass_zerophase_matches_pass_semantics(self):
        figure = WaveFigure.__new__(WaveFigure)

        self.assertFalse(figure._bandpass_zerophase_enabled(1))
        self.assertTrue(figure._bandpass_zerophase_enabled(2))
        self.assertTrue(figure._bandpass_zerophase_enabled(3))

    def test_apply_bandpass_to_trace_changes_result_between_p1_and_p2(self):
        figure = WaveFigure.__new__(WaveFigure)
        dt = 0.05
        t = np.arange(4000, dtype=float) * dt
        data = np.sin(2 * np.pi * 0.12 * t)
        onset = 1600
        data[onset:] += 1.2 * np.exp(-((t[:-onset] - t[onset]) * 3.0) ** 2)

        trace_p1 = Trace(data=data.copy())
        trace_p1.stats.delta = dt
        trace_p2 = Trace(data=data.copy())
        trace_p2.stats.delta = dt

        settings_p1 = {'freqmin': 0.05, 'freqmax': 0.4, 'corners': 2, 'passes': 1}
        settings_p2 = {'freqmin': 0.05, 'freqmax': 0.4, 'corners': 2, 'passes': 2}

        figure._apply_bandpass_to_trace(trace_p1, settings_p1)
        figure._apply_bandpass_to_trace(trace_p2, settings_p2)

        self.assertFalse(np.allclose(trace_p1.data, trace_p2.data))

    def test_reapplying_same_view_marker_keeps_existing_reference_snapshot(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.tmarker = 't7'
        figure.tmarker_t = np.asarray([55.0], dtype=float)
        figure.t7 = np.asarray([55.4], dtype=float)
        figure.xlim = [-1.0, 1.0]
        figure.axis_mode = 'relative'
        figure.ipage = 3
        figure.Change_time_window = lambda: None

        figure.set_view_settings('t7', xlim=[-2.0, 2.0], axis_mode='relative')

        self.assertEqual(float(figure.tmarker_t[0]), 55.0)
        self.assertEqual(figure.xlim, [-2.0, 2.0])
        self.assertEqual(figure.ipage, 0)

    def test_pick_window_refresh_draws_updated_marker_against_current_view_reference(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.sta_num = 1
        figure.maxidx = 5
        figure.ipage = 0
        figure.axis_mode = 'relative'
        figure.xlim = [-1.0, 1.0]
        figure.tmarker = 't7'
        figure.tmarker_t = np.asarray([55.0], dtype=float)
        figure.t0 = np.asarray([50.0], dtype=float)
        figure.ori_sacnames = ['waveA']
        figure.filenames = ['NET.STA']
        figure.gcarc = np.asarray([10.0], dtype=float)
        figure.az = np.asarray([20.0], dtype=float)
        figure.baz = np.asarray([30.0], dtype=float)
        figure.wave = [DummyTrace(np.zeros(2400), delta=0.05, b=0.0)]
        figure.enf = 1.0
        figure.A1lines = [[] for _ in range(5)]
        figure.preview_hidden_wave_names = set()
        figure.preview_selected_wave_names = set()
        figure.preview_jump_highlight_wave_name = None
        figure.current_pick_wave_name = 'waveA'
        figure.current_pick_station_name = 'NET.STA'
        figure.user_markers = {'user1': {}, 'user2': {}, 'user3': {}}
        figure.user1_mark_color = '#008000'
        figure.preview_mark_color = '#5b2a86'
        figure.markers = {'7': {'waveA': 55.4}}
        figure.marker_styles = {'7': ('t7', '#d62728')}
        figure.theory_time_model = 'iasp91'
        figure._current_wave_theory_delta = lambda wave_name=None, model=None: {}
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''

        figure.refresh_current_page()

        marker_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.4, 0.4])
        ]
        zero_marker_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [0.0, 0.0])
        ]
        self.assertTrue(marker_lines)
        self.assertFalse(zero_marker_lines)

    def test_stack_pick_window_draws_sac_time_without_negative_slice_wraparound(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.stack_mode = True
        figure.sta_num = 1
        figure.maxidx = 5
        figure.ipage = 0
        figure.axis_mode = 'relative'
        figure.xlim = [-10.0, 70.0]
        figure.tmarker = 't6'
        figure.tmarker_t = np.asarray([0.0], dtype=float)
        for marker_key in range(10):
            setattr(figure, f't{marker_key}', np.asarray([math.nan], dtype=float))
        figure.t6 = np.asarray([0.0], dtype=float)
        figure.ori_sacnames = ['stack_a.sac']
        figure.filenames = ['DPK.STACK']
        figure.gcarc = np.asarray([88.0], dtype=float)
        figure.az = np.asarray([65.0], dtype=float)
        figure.baz = np.asarray([211.0], dtype=float)
        figure.wave = [DummyTrace(np.arange(3000), delta=0.02, b=0.0)]
        figure.enf = 1.0
        figure.A1lines = [[] for _ in range(5)]
        figure.preview_hidden_wave_names = set()
        figure.preview_selected_wave_names = set()
        figure.preview_jump_highlight_wave_name = None
        figure.current_pick_wave_name = 'stack_a.sac'
        figure.current_pick_station_name = 'DPK.STACK'
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure.user1_mark_color = '#008000'
        figure.user5_mark_color = '#00bcd4'
        figure.user4_mark_color = '#9a4f00'
        figure.preview_mark_color = '#5b2a86'
        figure.markers = {str(idx): {'stack_a.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_a.sac'] = 0.0
        figure.marker_styles = {
            str(idx): (f't{idx}', '#800080')
            for idx in range(10)
        }
        figure.theory_time_model = 'iasp91'
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''

        figure.refresh_current_page()

        wave_line = figure.ax1.lines[0]
        xdata = np.asarray(wave_line.get_xdata(), dtype=float)
        self.assertAlmostEqual(float(xdata[0]), 0.0)
        self.assertAlmostEqual(float(xdata[-1]), 59.98)
        self.assertGreater(len(xdata), 2500)

    def test_stack_pick_window_draws_alignment_marker_at_stack_relative_position(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.fig = Figure()
        figure.ax1 = figure.fig.add_subplot(5, 1, 1)
        figure.ax2 = figure.fig.add_subplot(5, 1, 2)
        figure.ax3 = figure.fig.add_subplot(5, 1, 3)
        figure.ax4 = figure.fig.add_subplot(5, 1, 4)
        figure.ax5 = figure.fig.add_subplot(5, 1, 5)
        figure.stack_mode = True
        figure.sta_num = 1
        figure.maxidx = 5
        figure.ipage = 0
        figure.axis_mode = 'absolute'
        figure.xlim = [0.0, 50.0]
        figure.tmarker = 't6'
        figure.tmarker_t = np.asarray([791.0], dtype=float)
        for marker_key in range(10):
            setattr(figure, f't{marker_key}', np.asarray([math.nan], dtype=float))
        figure.t6 = np.asarray([791.0], dtype=float)
        figure.ori_sacnames = ['stack_a/stack_a.sac']
        figure.filenames = ['DPK.STACK']
        figure.gcarc = np.asarray([88.0], dtype=float)
        figure.az = np.asarray([65.0], dtype=float)
        figure.baz = np.asarray([211.0], dtype=float)
        figure.wave = [DummyTrace(np.ones(1000), delta=0.05, b=0.0)]
        figure.enf = 1.0
        figure.A1lines = [[] for _ in range(5)]
        figure.preview_hidden_wave_names = set()
        figure.preview_selected_wave_names = set()
        figure.preview_jump_highlight_wave_name = None
        figure.current_pick_wave_name = 'stack_a/stack_a.sac'
        figure.current_pick_station_name = 'DPK.STACK'
        figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
        figure.user1_mark_color = '#008000'
        figure.user5_mark_color = '#00bcd4'
        figure.user4_mark_color = '#9a4f00'
        figure.preview_mark_color = '#5b2a86'
        figure.markers = {str(idx): {'stack_a/stack_a.sac': math.nan} for idx in range(10)}
        figure.markers['6']['stack_a/stack_a.sac'] = 791.0
        figure.marker_styles = {str(idx): (f't{idx}', '#800080') for idx in range(10)}
        figure.stack_sidecars = {
            'stack_a/stack_a.sac': {
                'align_marker': 't6',
                'window': [-30.0, 20.0],
                'markers': {'t6': 791.0},
            }
        }
        figure.theory_time_model = 'iasp91'
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''

        figure.refresh_current_page()

        marker_lines = [
            line for line in figure.ax1.lines
            if len(line.get_xdata()) == 2 and np.allclose(line.get_xdata(), [30.0, 30.0])
        ]
        self.assertTrue(marker_lines)
        figure.markers['6']['stack_a/stack_a.sac'] = math.nan
        self.assertAlmostEqual(
            figure._stack_marker_display_x_value(791.0, 0),
            30.0,
        )
        figure.markers['6']['stack_a/stack_a.sac'] = 17.5
        self.assertAlmostEqual(
            figure._stack_marker_display_x_value(17.5, 0),
            17.5,
        )

    def test_stack_read_sac_keeps_wave_names_aligned_after_obspy_sort(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            stack_dir = Path(temp_dir, 'stack_evt')
            stack_dir.mkdir()

            def write_stack(path, network, t6, gcarc):
                trace = Trace(data=np.ones(100, dtype=np.float32))
                trace.stats.network = network
                trace.stats.station = 'STACK'
                trace.stats.channel = 'BHZ'
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(
                    b=0.0,
                    e=4.95,
                    t6=t6,
                    gcarc=gcarc,
                    az=10.0,
                    baz=20.0,
                    nzyear=2002,
                    nzjday=41,
                    nzhour=1,
                    nzmin=47,
                    nzsec=7,
                    evla=-55.97,
                    evlo=-29.15,
                    evdp=198.0,
                )
                trace.write(str(path), format='SAC')

            write_stack(stack_dir / 'stack_group1.sac', 'ZZZ', 11.0, 39.85)
            write_stack(stack_dir / 'stack_group2.sac', 'AAA', 22.0, 82.29)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(stack_dir)
            figure.stack_mode = True
            figure.stack_event_marker = {}
            figure.stack_sidecars = {}
            figure.suffix = '.sac'
            figure.maxidx = 5
            figure.tmarker = 't6'
            figure.fig = Figure()
            figure.markers = {str(idx): {} for idx in range(10)}
            figure.user_markers = {
                key: {}
                for key in ('user1', 'user2', 'user3', 'user4', 'user5')
            }

            figure.read_sac(order='gcarc')

            self.assertEqual(list(figure.ori_sacnames), ['stack_group1.sac', 'stack_group2.sac'])
            self.assertEqual(
                [tr.stats.dpk_wave_name for tr in figure.wave],
                ['stack_group1.sac', 'stack_group2.sac'],
            )
            self.assertAlmostEqual(figure.markers['6']['stack_group1.sac'], 11.0)
            self.assertAlmostEqual(figure.markers['6']['stack_group2.sac'], 22.0)

    def test_saved_stack_options_for_group_deduplicates_same_configuration(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.stack_sidecars = {
            'stack_group1.sac': {
                'group_name': 'group1',
                'scope': 'group:group1',
                'align_marker': 't6',
                'window': [-40.0, 20.0],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'linear',
                'moveout_mode': 'off',
                'moveout_phase': None,
                'label': '',
            },
            'stack_group1_alt.sac': {
                'group_name': 'group1',
                'scope': 'group:group1',
                'align_marker': 't6',
                'window': [-40.0, 20.0],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'linear',
                'moveout_mode': 'off',
                'moveout_phase': None,
                'label': '',
            },
            'stack_group1_pws.sac': {
                'group_name': 'group1',
                'scope': 'group:group1',
                'align_marker': 't6',
                'window': [-40.0, 20.0],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'pws',
                'moveout_mode': 'off',
                'moveout_phase': None,
                'label': '',
            },
        }
        figure._normalize_preview_group_name = lambda raw: 'group1'

        saved = figure._saved_stack_options_for_group('group1')

        self.assertEqual(len(saved), 2)
        stack_types = sorted(option['stack_type'] for option in saved)
        self.assertEqual(stack_types, ['linear', 'pws'])

    def test_saved_stack_options_for_group_loads_stack_workspace_from_source_window(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = False
        figure._stack_data_event_directory = lambda: '/tmp/stack_evt'
        sidecars = {
            'stack_group1.sac': {
                'scope': 'group:group1',
                'align_marker': 't6',
                'window': [-55.0, 25.0],
                'polarity': 'apply_user4',
                'normalize': 'rms',
                'stack_type': 'linear',
                'moveout_mode': 'off',
                'moveout_phase': None,
                'label': '',
                'stack_wave_name': 'stack_group1.sac',
            },
        }

        with patch('WaveFigure.load_stack_sidecar_map', return_value=sidecars) as load_mock:
            saved = figure._saved_stack_options_for_group('group1')

        load_mock.assert_called_once_with('/tmp/stack_evt')
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]['align_marker'], '6')
        self.assertEqual(saved[0]['x1'], -55.0)
        self.assertEqual(saved[0]['x2'], 25.0)

    def test_saved_stack_config_defaults_to_first_saved_option(self):
        figure = WaveFigure.__new__(WaveFigure)

        self.assertEqual(figure._saved_stack_config_default_combo_index([]), 0)
        self.assertEqual(figure._saved_stack_config_default_combo_index([{'stack_type': 'linear'}]), 1)
        self.assertEqual(
            figure._saved_stack_config_default_combo_index([
                {'stack_type': 'linear'},
                {'stack_type': 'pws'},
            ]),
            1,
        )

    def test_stack_saved_option_payload_reads_window_and_polarity(self):
        figure = WaveFigure.__new__(WaveFigure)

        payload = figure._stack_saved_option_payload({
            'scope': 'group:group2',
            'align_marker': 't5',
            'window': [-10.0, 70.0],
            'polarity': 'reject_mixed',
            'normalize': 'peak',
            'stack_type': 'pws',
            'moveout_mode': 'phase',
            'moveout_phase': 't3',
            'label': 'test_one',
            'group_name': 'group2',
        })

        self.assertEqual(payload['scope'], 'group:group2')
        self.assertEqual(payload['align_marker'], '5')
        self.assertEqual(payload['x1'], -10.0)
        self.assertEqual(payload['x2'], 70.0)
        self.assertEqual(payload['polarity'], 'reject_mixed')
        self.assertEqual(payload['normalize'], 'peak')
        self.assertEqual(payload['stack_type'], 'pws')
        self.assertEqual(payload['moveout_mode'], 'phase')
        self.assertEqual(payload['moveout_phase'], '3')

    def test_preview_selection_does_not_refresh_pick_window(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_selected_wave_names = set()
        figure.preview_hidden_wave_names = set()
        figure.preview_hidden_batches = []
        figure.ori_sacnames = ['waveA', 'waveB']
        figure.preview_trace_layout_mode = 'real'
        figure.preview_even_spacing_step = 1.0
        figure.preview_amplitude_scale = 1.0
        figure.standard_export_phase_tokens = ''
        figure.user1_selected_color = '#ffd400'
        figure.user1_mark_color = '#008f5a'
        figure.preview_selected_mark_color = '#d2691e'
        figure.preview_mark_color = '#9a1fff'
        figure.theory_time_model = 'iasp91'
        figure._current_wave_theory_delta = lambda wave_name=None, model=None: {}
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''
        refresh_calls = []
        figure._refresh_pick_window_if_available = lambda *args, **kwargs: refresh_calls.append((args, kwargs))

        fig = Figure()
        axr = fig.add_subplot(1, 2, 1)
        axb = fig.add_subplot(1, 2, 2)
        line_a, = axr.plot([0.0, 1.0], [10.0, 10.0])
        line_b, = axr.plot([0.0, 1.0], [20.0, 20.0])
        scatter = axb.scatter([15.0, 30.0], [10.0, 20.0])
        selected_marker, = axb.plot([], [], 'o')
        fig._preview_state = {
            'tmarker': '0',
            'evtdata': SimpleNamespace(
                az=np.asarray([15.0, 30.0], dtype=float),
                gcarc=np.asarray([10.0, 20.0], dtype=float),
            ),
            'lines': [line_a, line_b],
            'scatter': scatter,
            'selected_marker': selected_marker,
            'info_text': fig.text(0.5, 0.5, ''),
            'control_status_text': fig.text(0.5, 0.4, ''),
            'metadata': [
                {'wave_name': 'waveA', 'name': 'NET.A', 'gcarc': 10.0, 'az': 15.0},
                {'wave_name': 'waveB', 'name': 'NET.B', 'gcarc': 20.0, 'az': 30.0},
            ],
            'selected_indices': {0},
            'active_index': 0,
            'anchor_index': 0,
            'y_values': np.asarray([10.0, 20.0], dtype=float),
            'window_width': 80.0,
            'tick_interval': 10.0,
            'tick_mode': 'auto',
        }

        figure._update_preview_selection(fig, 1, mode='single')

        self.assertEqual(refresh_calls, [])
        self.assertEqual(figure.preview_selected_wave_names, {'waveB'})

    def test_event_theory_delta_summary_reads_cache_file_without_regeneration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            figure = WaveFigure.__new__(WaveFigure)
            figure.theory_time_model = 'iasp91'
            figure.theory_time_cache = {}
            figure.theory_delta_summary_cache = None
            figure.ori_sacnames = ['waveA.sac']
            figure.gcarc = np.asarray([88.5], dtype=float)
            figure.wave = [DummyTrace([0.0, 1.0])]
            figure._analysis_output_directory = lambda: tmpdir
            summary_path = Path(tmpdir) / 'theory_time_summary_iasp91.json'
            payload = {
                'model': 'iasp91',
                'evdp': 35.0,
                'pP-P_mean': 28.2,
                'sP-P_mean': 43.1,
                'per_wave': {
                    'waveA.sac': {
                        'model': 'iasp91',
                        'P': 400.0,
                        'pP': 428.2,
                        'sP': 443.1,
                        'pP-P': 28.2,
                        'sP-P': 43.1,
                    }
                },
            }
            summary_path.write_text(json.dumps(payload), encoding='utf-8')

            with patch.object(figure, '_theory_phase_deltas_for_gcarc', side_effect=AssertionError('should not regenerate')):
                summary = figure._event_theory_delta_summary(model='iasp91')

            self.assertEqual(summary['pP-P_mean'], 28.2)
            cache_key = figure._theory_time_cache_key('iasp91', 'P,pP,sP', 35.0, 88.5)
            self.assertEqual(figure.theory_time_cache[cache_key]['pP'], 428.2)

    def test_event_theory_delta_summary_returns_none_when_cache_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            figure = WaveFigure.__new__(WaveFigure)
            figure.theory_time_model = 'iasp91'
            figure.theory_time_cache = {}
            figure.theory_delta_summary_cache = None
            figure.ori_sacnames = ['waveA.sac']
            figure.gcarc = np.asarray([88.5], dtype=float)
            figure.wave = [DummyTrace([0.0, 1.0])]
            figure._analysis_output_directory = lambda: tmpdir

            with patch.object(figure, '_theory_phase_deltas_for_gcarc', side_effect=AssertionError('should not regenerate')):
                summary = figure._event_theory_delta_summary(model='iasp91')

            self.assertIsNone(summary)

    def test_theory_summary_cache_path_uses_source_event_dir_in_stack_mode(self):
        with tempfile.TemporaryDirectory() as project_root, \
                tempfile.TemporaryDirectory() as stack_dir:
            figure = WaveFigure.__new__(WaveFigure)
            figure.theory_time_model = 'iasp91'
            figure.stack_mode = True
            source_event_dir = os.path.join(
                project_root, 'data', 'pick_other', '2013_03_19_03_29_02'
            )
            figure.runtime_event_dir = source_event_dir
            # Generic analysis dir points at the stack workspace, which must NOT be used.
            figure._analysis_output_directory = lambda: stack_dir

            with patch("WaveFigure.PROJECT_ROOT", Path(project_root)):
                cache_path = figure._theory_summary_cache_path(model='iasp91')

            expected_dir = os.path.join(
                project_root, 'data', 'output', 'phases', 'pick_other', '2013_03_19_03_29_02'
            )
            self.assertEqual(os.path.dirname(cache_path), expected_dir)
            self.assertNotIn(stack_dir, cache_path)
            self.assertTrue(cache_path.endswith('theory_time_summary_iasp91.json'))

    def test_event_theory_delta_summary_reads_source_event_summary_in_stack_mode(self):
        with tempfile.TemporaryDirectory() as project_root:
            figure = WaveFigure.__new__(WaveFigure)
            figure.theory_time_model = 'iasp91'
            figure.theory_time_cache = {}
            figure.theory_delta_summary_cache = None
            figure.stack_mode = True
            figure.suffix = '.sac'
            figure.preview_pierce_taup_bin = 'taup'
            source_event_dir = os.path.join(
                project_root, 'data', 'pick_other', '2013_03_19_03_29_02'
            )
            figure.runtime_event_dir = source_event_dir
            figure.ori_sacnames = ['waveA.sac']
            figure.gcarc = np.asarray([88.5], dtype=float)
            figure.wave = [DummyTrace([0.0, 1.0])]
            summary_dir = os.path.join(
                project_root, 'data', 'output', 'phases', 'pick_other', '2013_03_19_03_29_02'
            )
            summary_path = os.path.join(summary_dir, 'theory_time_summary_iasp91.json')
            os.makedirs(summary_dir, exist_ok=True)
            payload = {
                'model': 'iasp91',
                'evdp': 35.0,
                'pP-P_mean': 28.2,
                'sP-P_mean': 43.1,
                'per_wave': {
                    'waveA.sac': {
                        'model': 'iasp91',
                        'P': 400.0,
                        'pP': 428.2,
                        'sP': 443.1,
                        'pP-P': 28.2,
                        'sP-P': 43.1,
                    }
                },
            }
            with open(summary_path, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)

            with patch("WaveFigure.PROJECT_ROOT", Path(project_root)):
                # On-demand generation must not fire when the summary already exists.
                with patch.object(figure, '_ensure_event_theory_summary', side_effect=AssertionError('should not regenerate')):
                    summary = figure._event_theory_delta_summary(model='iasp91')

            self.assertIsNotNone(summary)
            self.assertEqual(summary['pP-P_mean'], 28.2)

    def test_preview_curve_peak_pick_refreshes_pick_window_after_marker_update(self):
        line = DummyLine(
            [5.2, 5.3, 5.4, 5.5],
            [0.0, 0.0, 1.0, 0.0],
        )
        curve_points = [(5.3, -1.0), (5.3, 1.0)]
        figure, fig = self._preview_action_figure(line, curve_points)
        refresh_calls = []
        figure._refresh_pick_window_if_available = lambda focus_current_wave=True: refresh_calls.append(focus_current_wave)

        applied = figure._apply_preview_curve_peak_pick(fig, preview_index=0, request_text='7')

        self.assertTrue(applied)
        self.assertEqual(figure.markers['7']['waveA'], 55.4)
        self.assertEqual(refresh_calls, [True])

    def test_preview_curve_marker_time_returns_none_when_curve_does_not_touch_wave(self):
        figure = WaveFigure.__new__(WaveFigure)
        line = DummyLine(
            [0.0, 1.0],
            [0.0, 0.0],
        )
        curve_points = [(0.2, 1.0), (0.8, 1.0)]

        marker_time = figure._curve_marker_time_for_preview_line(
            line=line,
            baseline_y=0.0,
            reference_time=50.0,
            curve_points=curve_points,
            half_window=None,
        )

        self.assertIsNone(marker_time)

    def test_stack_preview_collects_current_stack_with_source_members(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir, 'source_evt')
            stack_dir = Path(temp_dir, 'stack_evt')
            source_dir.mkdir()
            stack_dir.mkdir()

            def write_sac(path, network, station, data, gcarc, az, t6):
                trace = Trace(data=np.asarray(data, dtype=np.float32))
                trace.stats.network = network
                trace.stats.station = station
                trace.stats.delta = 0.05
                trace.stats.sac = obspy.core.AttribDict(
                    b=0.0,
                    e=(len(data) - 1) * 0.05,
                    gcarc=gcarc,
                    az=az,
                    baz=az + 180.0,
                    t6=t6,
                    nzyear=2011,
                    nzjday=65,
                    nzhour=14,
                    nzmin=32,
                    nzsec=36,
                    evla=-56.0,
                    evlo=-27.0,
                    evdp=92.0,
                )
                trace.write(str(path), format='SAC')

            write_sac(source_dir / 'member_a.sac', 'AA', 'A', np.linspace(0.0, 1.0, 400), 20.0, 10.0, 12.5)
            write_sac(source_dir / 'member_b.sac', 'BB', 'B', np.linspace(1.0, 0.0, 400), 30.0, 20.0, 15.0)
            # 真实的现代帧 stack SAC：长度与窗口一致（50 s），对齐头段位于 -x1=30.0。
            write_sac(stack_dir / 'stack_a.sac', 'DPK', 'STACK', np.ones(1000), 25.0, 15.0, 30.0)

            figure = WaveFigure.__new__(WaveFigure)
            figure.wavepath = str(stack_dir)
            figure.runtime_event_dir = str(source_dir)
            figure.stack_mode = True
            figure.stack_sidecars = {
                'stack_a.sac': {
                    'align_marker': 't6',
                    'window': [-30.0, 20.0],
                    'wave_names_used': ['member_a.sac', 'member_b.sac'],
                    'geometry': {
                        'gcarc_mean': 25.0,
                        'az_mean': 15.0,
                        'baz_mean': 195.0,
                        'pierce_lon_mean': -26.5,
                        'pierce_lat_mean': -56.5,
                    },
                    'event': {
                        'evla': -56.0,
                        'evlo': -27.0,
                        'evdp': 92.0,
                    },
                    'markers': {'t6': 30.0},
                }
            }
            figure.ori_sacnames = ['stack_a.sac']
            figure.current_pick_wave_name = 'stack_a.sac'
            figure.dt = 0.05
            figure.bandpass_settings = {
                'freqmin': None,
                'freqmax': None,
                'corners': 2,
                'passes': 2,
            }
            figure.user_markers = {key: {} for key in ('user1', 'user2', 'user3', 'user4', 'user5')}
            figure.preview_hidden_wave_names = set()
            figure.preview_modes = [['6', -50.0, 30.0]]

            waves, reference_times, active_reference_times = figure._collect_stack_preview_stream('6', 'stack_a.sac')

            self.assertEqual([tr.stats.dpk_wave_name for tr in waves], ['stack_a.sac', 'member_a.sac', 'member_b.sac'])
            self.assertEqual([tr.stats.dpk_stack_preview_role for tr in waves], ['stack', 'member', 'member'])
            # stack 道走窗口相对帧，对齐参考恒为 -x1；成员道保持各自的绝对到时。
            np.testing.assert_allclose(reference_times, np.asarray([30.0, 12.5, 15.0]))
            self.assertEqual(active_reference_times['member_a.sac'], 12.5)
            self.assertEqual(figure._stack_preview_display_mode(), 'overlay')
            self.assertLessEqual(float(waves[0].stats.sac.gcarc), 30.0)
            self.assertEqual(figure.stack_preview_active_wave_name, 'stack_a.sac')

            figure.preview_stack_display_mode = 'top'
            top_waves, top_reference_times, _active_reference_times = figure._collect_stack_preview_stream('6', 'stack_a.sac')
            self.assertGreater(float(top_waves[0].stats.sac.gcarc), 30.0)

            evtdata = EvtData(top_waves, top_reference_times, x1=-1.0, x2=1.0, dt=0.05)
            self.assertEqual(evtdata.wave_ori[-1].stats.dpk_wave_name, 'stack_a.sac')
            self.assertEqual(_stack_member_visible_mask(evtdata).tolist(), [True, True, False])

            applied_window = figure._apply_stack_preview_window(0, 'stack_a.sac')
            self.assertEqual(applied_window, (-30.0, 20.0))
            self.assertEqual(figure.preview_modes[0][1:], [-30.0, 20.0])

    def _pierce_points_figure(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.ori_sacnames = ['stack_a.sac', 'stack_b.sac']
        figure.stack_sidecars = {
            'stack_a.sac': {
                'wave_names_used': ['member_a.sac', 'member_b.sac'],
                'geometry': {
                    'pierce_lon_mean': -26.5,
                    'pierce_lat_mean': -56.5,
                }
            },
            'stack_b.sac': {
                'geometry': {
                    'pierce_lon_mean': -25.5,
                    'pierce_lat_mean': -55.5,
                }
            },
        }
        return figure

    def test_stack_preview_pierce_points_show_all_means_when_no_active_preview(self):
        # 没有活动预览时回退：显示全部 group 的均值点。
        figure = self._pierce_points_figure()
        figure._current_stack_preview_wave_name = lambda: ''

        records = figure._stack_preview_pierce_points()

        self.assertEqual([record.wave_name for record in records], ['stack_a.sac', 'stack_b.sac'])
        self.assertEqual(
            [(record.longitude, record.latitude) for record in records],
            [(-26.5, -56.5), (-25.5, -55.5)],
        )

    def test_stack_preview_pierce_points_track_active_group_only(self):
        # 有活动预览时穿透点面板跟随当前 group，不再固定显示所有 group 的均值，
        # 否则切换 group 时图上的点不动，无法判断看的是哪一组。
        figure = self._pierce_points_figure()
        figure._current_stack_preview_wave_name = lambda: 'stack_a.sac'

        records = figure._stack_preview_pierce_points(include_members=False)

        self.assertEqual([record.wave_name for record in records], ['stack_a.sac'])

    def test_stack_preview_pierce_points_prepend_members_of_active_group(self):
        # 打开成员点时，成员在前、该 group 均值在后；其它 group 不出现。
        figure = self._pierce_points_figure()
        figure._current_stack_preview_wave_name = lambda: 'stack_a.sac'
        figure._load_pierce_points_for_current_event = lambda auto_generate=False: {
            'member_a.sac': SimpleNamespace(longitude=-26.1, latitude=-56.1),
            'member_b.sac': SimpleNamespace(longitude=-26.2, latitude=-56.2),
        }

        records = figure._stack_preview_pierce_points(
            active_stack_wave_name='stack_a.sac',
            include_members=True,
        )

        self.assertEqual(
            [record.wave_name for record in records],
            ['member::member_a.sac', 'member::member_b.sac', 'stack_a.sac'],
        )

    def test_preview_index_from_pierce_click_matches_stack_member_record_name(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_wave_names_for_pierce_record = WaveFigure._preview_wave_names_for_pierce_record.__get__(figure, WaveFigure)
        figure._is_stack_preview_member_pierce_name = WaveFigure._is_stack_preview_member_pierce_name.__get__(figure, WaveFigure)
        preview_state = {
            'metadata': [
                {'wave_name': 'member_a.sac'},
                {'wave_name': 'stack_a.sac'},
            ],
            'pierce_state': {
                'records': [
                    SimpleNamespace(wave_name='member::member_a.sac', longitude=10.0, latitude=20.0),
                    SimpleNamespace(wave_name='stack_a.sac', longitude=30.0, latitude=40.0),
                ],
            },
        }
        event = SimpleNamespace(xdata=10.05, ydata=20.05)

        selected_index = WaveFigure._preview_index_from_pierce_click(figure, preview_state, event)

        self.assertEqual(selected_index, 0)

    def test_apply_preview_selection_highlights_stack_member_pierce_record(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.preview_stack_show_member_pierce = True
        figure._preview_wave_colors = lambda meta, is_selected: ('#ff5fa2' if is_selected else '#1f77b4', 1.0)
        figure._apply_preview_line_style = lambda line, meta, is_selected, line_color, line_width: None
        figure._sync_pick_highlight_from_preview_selection = WaveFigure._sync_pick_highlight_from_preview_selection.__get__(figure, WaveFigure)
        figure._preview_selected_record_wave_names = WaveFigure._preview_selected_record_wave_names.__get__(figure, WaveFigure)
        figure._stack_preview_pierce_member_record_name = WaveFigure._stack_preview_pierce_member_record_name.__get__(figure, WaveFigure)
        figure._is_stack_preview_member_pierce_name = WaveFigure._is_stack_preview_member_pierce_name.__get__(figure, WaveFigure)
        figure._group_number_from_record = lambda record: None
        figure._group_number_from_wave_name = lambda wave_name: None
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''
        figure.preview_trace_layout_mode = 'real'
        figure.preview_view_mode = 'wide'
        figure.standard_export_phase_tokens = ''
        figure._preview_layout_summary = lambda: 'real'
        figure._preview_view_mode_label = lambda: 'wide'
        figure._stack_crustal_summary_text = lambda wave_name: ''
        figure._preview_hidden_summary = lambda: (0, 0)
        figure._preview_reference_mode_label = lambda tmarker: 'marker'
        figure.preview_amplitude_scale = 1.0
        figure.preview_even_spacing_step = 1.0
        # 与 WaveFigure.__init__ 的默认值一致（WaveFigure.py 中 preview_hidden_wave_names = set()）；
        # 这些用例走 __new__ 绕过 __init__，属性需手工补齐。
        figure.preview_hidden_wave_names = set()
        figure._pierce_record_style = lambda wave_name, selected=False: (
            '#ff5fa2' if selected else '#1f77b4',
            '#ff5fa2' if selected else '#1f77b4',
        )
        fig = Figure()
        axp = fig.add_subplot(111)
        highlight_scatter = axp.scatter([], [])
        preview_state = {
            'lines': [DummyLine([0.0], [0.0]), DummyLine([0.0], [1.0])],
            'metadata': [
                {'wave_name': 'member_a.sac', 'name': 'member', 'gcarc': 10.0, 'az': 20.0, 'stack_preview_role': 'member'},
                {'wave_name': 'stack_a.sac', 'name': 'stack', 'gcarc': 30.0, 'az': 40.0, 'stack_preview_role': 'stack'},
            ],
            'selected_indices': {0},
            'active_index': 0,
            'anchor_index': 0,
            'scatter': None,
            'selected_marker': None,
            'pierce_state': {
                'axes': axp,
                'records': [
                    SimpleNamespace(wave_name='member::member_a.sac', longitude=11.0, latitude=21.0),
                    SimpleNamespace(wave_name='stack_a.sac', longitude=31.0, latitude=41.0),
                ],
                'base_scatter': None,
                'highlight_scatter': highlight_scatter,
            },
            'evtdata': SimpleNamespace(gcarc=np.asarray([10.0, 30.0]), az=np.asarray([20.0, 40.0])),
            'info_text': SimpleNamespace(set_color=lambda value: None, set_text=lambda value: None),
            'control_status_text': None,
            'y_values': np.asarray([10.0, 30.0]),
            'azimuth_y_values': np.asarray([10.0, 30.0]),
        }
        fig_wrapper = SimpleNamespace(
            _preview_state=preview_state,
            _stack_preview_wave_name='stack_a.sac',
            canvas=SimpleNamespace(draw_idle=lambda: None),
        )

        figure._apply_preview_selection(fig_wrapper)

        offsets = highlight_scatter.get_offsets()
        self.assertEqual(offsets.shape[0], 1)
        self.assertTrue(np.allclose(offsets[0], [11.0, 21.0]))

    def test_apply_preview_selection_highlights_only_selected_stack_mean_point(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.stack_mode = True
        figure.preview_stack_show_member_pierce = True
        figure._preview_wave_colors = lambda meta, is_selected: ('#ff5fa2' if is_selected else '#1f77b4', 1.0)
        figure._apply_preview_line_style = lambda line, meta, is_selected, line_color, line_width: None
        figure._sync_pick_highlight_from_preview_selection = WaveFigure._sync_pick_highlight_from_preview_selection.__get__(figure, WaveFigure)
        figure._preview_selected_record_wave_names = WaveFigure._preview_selected_record_wave_names.__get__(figure, WaveFigure)
        figure._stack_preview_pierce_member_record_name = WaveFigure._stack_preview_pierce_member_record_name.__get__(figure, WaveFigure)
        figure._is_stack_preview_member_pierce_name = WaveFigure._is_stack_preview_member_pierce_name.__get__(figure, WaveFigure)
        figure._group_number_from_record = lambda record: None
        figure._group_number_from_wave_name = lambda wave_name: None
        figure.current_wave_theory_delta_text = lambda wave_name=None, model=None: ''
        figure.preview_trace_layout_mode = 'real'
        figure.preview_view_mode = 'wide'
        figure.standard_export_phase_tokens = ''
        figure._preview_layout_summary = lambda: 'real'
        figure._preview_view_mode_label = lambda: 'wide'
        figure._stack_crustal_summary_text = lambda wave_name: ''
        figure._preview_hidden_summary = lambda: (0, 0)
        figure._preview_reference_mode_label = lambda tmarker: 'marker'
        figure.preview_amplitude_scale = 1.0
        figure.preview_even_spacing_step = 1.0
        # 与 WaveFigure.__init__ 的默认值一致（WaveFigure.py 中 preview_hidden_wave_names = set()）；
        # 这些用例走 __new__ 绕过 __init__，属性需手工补齐。
        figure.preview_hidden_wave_names = set()
        figure._pierce_record_style = lambda wave_name, selected=False: (
            '#ff5fa2' if selected else '#1f77b4',
            '#ff5fa2' if selected else '#1f77b4',
        )
        fig = Figure()
        axp = fig.add_subplot(111)
        highlight_scatter = axp.scatter([], [])
        preview_state = {
            'lines': [DummyLine([0.0], [0.0]), DummyLine([0.0], [1.0])],
            'metadata': [
                {'wave_name': 'member_a.sac', 'name': 'member', 'gcarc': 10.0, 'az': 20.0, 'stack_preview_role': 'member'},
                {'wave_name': 'stack_a.sac', 'name': 'stack', 'gcarc': 30.0, 'az': 40.0, 'stack_preview_role': 'stack'},
            ],
            'selected_indices': {1},
            'active_index': 1,
            'anchor_index': 1,
            'scatter': None,
            'selected_marker': None,
            'pierce_state': {
                'axes': axp,
                'records': [
                    SimpleNamespace(wave_name='member::member_a.sac', longitude=11.0, latitude=21.0),
                    SimpleNamespace(wave_name='stack_a.sac', longitude=31.0, latitude=41.0),
                ],
                'base_scatter': None,
                'highlight_scatter': highlight_scatter,
            },
            'evtdata': SimpleNamespace(gcarc=np.asarray([10.0, 30.0]), az=np.asarray([20.0, 40.0])),
            'info_text': SimpleNamespace(set_color=lambda value: None, set_text=lambda value: None),
            'control_status_text': None,
            'y_values': np.asarray([10.0, 30.0]),
            'azimuth_y_values': np.asarray([10.0, 30.0]),
        }
        fig_wrapper = SimpleNamespace(
            _preview_state=preview_state,
            _stack_preview_wave_name='stack_a.sac',
            canvas=SimpleNamespace(draw_idle=lambda: None),
        )

        figure._apply_preview_selection(fig_wrapper)

        offsets = highlight_scatter.get_offsets()
        self.assertEqual(offsets.shape[0], 1)
        self.assertTrue(np.allclose(offsets[0], [31.0, 41.0]))

    def test_plot_waves_with_masked_azimuth_keeps_all_lines_but_masks_scatter(self):
        fig = Figure()
        axr = fig.add_subplot(1, 2, 1)
        axb = fig.add_subplot(1, 2, 2)
        evtdata = SimpleNamespace(
            sta_num=3,
            data=np.asarray([
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 3.0, 0.0],
            ], dtype=float),
            time_axis=np.asarray([-0.05, 0.0, 0.05], dtype=float),
            az=np.asarray([10.0, 20.0, 30.0], dtype=float),
            gcarc=np.asarray([1.0, 2.0, 3.0], dtype=float),
        )

        lines, scatter = plot_waves_with_masked_azimuth(
            axr,
            axb,
            evtdata,
            enf=1.0,
            y_values=evtdata.gcarc,
            azimuth_mask=np.asarray([True, False, True]),
        )

        self.assertEqual(len(lines), 3)
        self.assertEqual(scatter.get_offsets().shape[0], 2)
        self.assertEqual(scatter._dpk_preview_full_indices.tolist(), [0, 2])

    def test_stack_preview_line_uses_distinct_deep_red_style(self):
        fig = Figure()
        axr = fig.add_subplot(1, 2, 1)
        axb = fig.add_subplot(1, 2, 2)
        stack_trace = Trace(data=np.asarray([1.0], dtype=np.float32))
        stack_trace.stats.dpk_stack_preview_role = 'stack'
        member_trace = Trace(data=np.asarray([1.0], dtype=np.float32))
        member_trace.stats.dpk_stack_preview_role = 'member'
        evtdata = SimpleNamespace(
            sta_num=2,
            data=np.asarray([
                [0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0],
            ], dtype=float),
            time_axis=np.asarray([-0.05, 0.0, 0.05], dtype=float),
            az=np.asarray([10.0, 20.0], dtype=float),
            gcarc=np.asarray([1.0, 2.0], dtype=float),
            wave_ori=[stack_trace, member_trace],
        )

        lines, _scatter = plot_waves_with_masked_azimuth(axr, axb, evtdata, enf=1.0)

        self.assertEqual(lines[0].get_color(), STACK_TRACE_COLOR)
        self.assertEqual(lines[0].get_linewidth(), STACK_TRACE_LINEWIDTH)
        self.assertEqual(lines[1].get_color(), MEMBER_TRACE_COLOR)
        self.assertEqual(lines[1].get_linewidth(), MEMBER_TRACE_LINEWIDTH)

    def test_preview_should_async_pierce_panel_uses_threshold(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_pierce_async_threshold = 140

        self.assertFalse(figure._preview_should_async_pierce_panel(140))
        self.assertTrue(figure._preview_should_async_pierce_panel(141))

    def test_preview_should_defer_side_panels_uses_large_event_threshold(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_deferred_panel_threshold = 220

        self.assertFalse(figure._preview_should_defer_side_panels(220))
        self.assertTrue(figure._preview_should_defer_side_panels(221))

    def test_preview_resource_warmup_loads_once_without_opening_preview_window(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.preview_modes = [['7', -10.0, 70.0]]
        figure.stack_mode = False
        figure._preview_warm_cache_keys = set()
        figure._current_bandpass_profile = lambda: None
        backend_warm_calls = []
        figure._warm_preview_backend_resources = lambda: backend_warm_calls.append(True) or True
        collect_calls = []
        load_calls = []
        figure._collect_preview_display_stream = lambda tmarker: collect_calls.append(tmarker) or (
            [SimpleNamespace(stats=SimpleNamespace(dpk_wave_name='wave_a.sac'))],
            np.asarray([12.0], dtype=float),
            {'wave_a.sac': 12.0},
        )
        figure._maybe_generate_current_preview_pierce_cache = lambda metadata: False
        figure._load_pierce_points_for_current_event = lambda auto_generate=False: load_calls.append(auto_generate) or {}

        self.assertTrue(figure.warm_preview_resources(0))
        self.assertFalse(figure.warm_preview_resources(0))

        self.assertEqual(collect_calls, ['7'])
        self.assertEqual(load_calls, [False])
        self.assertEqual(backend_warm_calls, [True])

    def test_preview_backend_warmup_runs_once_and_closes_hidden_figure(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure._preview_backend_warmed = False
        before = set(plt.get_fignums())

        self.assertTrue(figure._warm_preview_backend_resources())
        after_first = set(plt.get_fignums())
        self.assertFalse(figure._warm_preview_backend_resources())
        after_second = set(plt.get_fignums())

        self.assertEqual(after_first, before)
        self.assertEqual(after_second, before)

    def test_set_wave_marker_time_syncs_trace_sac_header(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'7': ('P', '#ff0000')}
        figure.markers = {'7': {'wave_a.sac': 12.5}}
        figure.ori_sacnames = ['wave_a.sac']
        figure.t7 = [12.5]
        figure.tmarker = 't6'
        figure.tmarker_t = [0.0]
        trace = Trace(data=np.asarray([0.0, 1.0], dtype=np.float32))
        trace.stats.delta = 0.05
        trace.stats.sac = obspy.core.AttribDict(t7=12.5)
        raw_trace = Trace(data=np.asarray([0.0, 1.0], dtype=np.float32))
        raw_trace.stats.delta = 0.05
        raw_trace.stats.sac = obspy.core.AttribDict(t7=12.5)
        figure.wave = [trace]
        figure.wave_raw = [raw_trace]

        changed = figure._set_wave_marker_time('wave_a.sac', '7', math.nan)

        self.assertTrue(changed)
        self.assertTrue(math.isnan(figure.markers['7']['wave_a.sac']))
        self.assertTrue(math.isnan(figure.t7[0]))
        self.assertTrue(math.isnan(float(figure.wave[0].stats.sac.t7)))
        self.assertTrue(math.isnan(float(figure.wave_raw[0].stats.sac.t7)))

    def test_clear_selected_preview_marker_clears_frozen_reference_for_align_marker(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.marker_styles = {'7': ('P', '#ff0000')}
        figure.markers = {'7': {'wave_a.sac': 12.5}}
        figure.preview_modes = [['7', -10.0, 10.0]]
        figure._normalize_marker_key = WaveFigure._normalize_marker_key.__get__(figure, WaveFigure)
        figure._parse_preview_delete_marker_keys = WaveFigure._parse_preview_delete_marker_keys.__get__(figure, WaveFigure)
        figure._set_wave_marker_time = lambda wave_name, marker_key, marker_time: figure.markers[marker_key].__setitem__(wave_name, marker_time) or True
        figure._update_preview_selection_by_wave_names = lambda fig, wave_names: None
        figure._apply_preview_selection = lambda fig: None
        figure._refresh_preview_figure = lambda fig, preview_index: None
        figure._refresh_pick_window_if_available = lambda focus_current_wave=False: None
        fig = SimpleNamespace(
            _preview_reference_times={'wave_a.sac': 12.5},
            _preview_reference_tmarker='7',
            _preview_state={
                'selected_indices': {0},
                'metadata': [{'wave_name': 'wave_a.sac'}],
                'reference_times': {'wave_a.sac': 12.5},
            },
        )

        cleared_count, marker_keys, error_message = figure._clear_selected_preview_marker(fig, 0, 't7')

        self.assertIsNone(error_message)
        self.assertEqual(cleared_count, 1)
        self.assertEqual(marker_keys, ['7'])
        self.assertIsNone(fig._preview_reference_times)
        self.assertIsNone(fig._preview_reference_tmarker)
        self.assertIsNone(fig._preview_state['reference_times'])

    def test_filtered_trace_for_preview_prefers_loaded_wave_for_non_stack_mode(self):
        figure = WaveFigure.__new__(WaveFigure)
        figure.dt = 0.05
        figure.stack_mode = False
        figure.bandpass_settings = {'freqmin': None, 'freqmax': None, 'corners': 2, 'passes': 2}
        figure._wave_polarity_factor = lambda wave_name: 1.0
        figure._apply_bandpass_to_trace = lambda tr, bandpass_settings: tr
        figure.ori_sacnames = ['wave_a.sac']
        trace = Trace(data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        trace.stats.delta = 0.05
        trace.stats.sampling_rate = 20.0
        trace.stats.sac = obspy.core.AttribDict(t7=math.nan)
        figure.wave = [trace]
        figure._trace_from_runtime_dir = lambda wave_name: (_ for _ in ()).throw(AssertionError('should not read from disk'))

        preview_trace = figure._filtered_trace_for_preview('wave_a.sac')

        self.assertEqual(getattr(preview_trace.stats, 'dpk_wave_name', ''), 'wave_a.sac')
        np.testing.assert_allclose(preview_trace.data, np.asarray([1.0, 2.0, 3.0], dtype=float))


class LegacyStackFileRecognitionTests(unittest.TestCase):
    """早期版本写出的叠加文件其 knetwk 与当前代码不同，必须仍被识别为叠加道。

    识别失败的后果不是报错而是静默降级：叠加道会被当成普通台站道，
    对齐帧、成员联动与厚度标注全部失效，且不易察觉。
    识别依赖 kstnm='STACK' 这条路径——历史文件的 kstnm 一直是它。
    """

    @staticmethod
    def _evtdata(network, station):
        trace = Trace(data=np.ones(400, dtype=np.float32))
        trace.stats.network = network
        trace.stats.station = station
        trace.stats.delta = 0.05
        trace.stats.sac = obspy.core.AttribDict(
            b=0.0, e=399 * 0.05, gcarc=30.0, az=10.0, baz=190.0
        )
        return EvtData(
            obspy.Stream([trace]), np.asarray([10.0]), x1=-1.0, x2=1.0, dt=0.05
        )

    def test_current_network_code_is_recognised(self):
        self.assertTrue(self._evtdata('DPK', 'NOTSTACK').is_stack_mode)

    def test_legacy_network_code_still_recognised_via_station_code(self):
        # 旧文件的网络代码是别的值，但台站代码一直是 STACK。
        self.assertTrue(self._evtdata('OLDCODE', 'STACK').is_stack_mode)

    def test_ordinary_trace_is_not_stack(self):
        self.assertFalse(self._evtdata('IU', 'ANMO').is_stack_mode)

    def test_writer_codes_are_exposed_as_constants(self):
        from WaveFigure import STACK_NETWORK_CODE, STACK_STATION_CODE
        self.assertEqual(STACK_NETWORK_CODE, 'DPK')
        self.assertEqual(STACK_STATION_CODE, 'STACK')
