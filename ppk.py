from os.path import exists, dirname, join
import json
import math
import os
import sys
import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, QEvent, Qt, QThread, Signal
from PySide6.QtGui import QCursor, QIcon, QKeySequence, QAction, QShortcut, QKeyEvent,QIntValidator, QDoubleValidator
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout,QLabel, \
                            QSizePolicy, QWidget, \
                            QPushButton, QHBoxLayout, QFileDialog,QLineEdit,QComboBox
from PySide6.QtWidgets import QLayout
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

from window_geometry import center_widget_on_workarea, maximize_on_workarea, screen_workarea_rect


class ClearTheoryMarkersThread(QThread):
    """Background thread to clear theory markers from multiple SAC files"""
    finished = Signal(int, int)  # success_count, error_count

    def __init__(self, wavepath, suffix, markers_dict):
        super().__init__()
        self.wavepath = wavepath
        self.suffix = suffix
        self.markers_dict = markers_dict

    def run(self):
        sac_files = sorted([f for f in os.listdir(self.wavepath) if f.lower().endswith(self.suffix.lower())])
        success_count = 0
        error_count = 0

        for sac_file in sac_files:
            sac_path = os.path.join(self.wavepath, sac_file)
            try:
                st = obspy.read(sac_path)
                sac_headers = st[0].stats.sac
                # Clear theory marker headers (t0-t9)
                should_write = False
                for t_marker in range(10):
                    header_key = f't{t_marker}'
                    # Only clear if it's not in the current picked markers
                    if sac_file not in self.markers_dict.get(str(t_marker), {}):
                        if hasattr(sac_headers, header_key):
                            current_val = getattr(sac_headers, header_key)
                            # Only write if the value is set (not undefined)
                            if current_val != -12345.0:
                                sac_headers[header_key] = -12345.0
                                should_write = True

                if should_write:
                    st.write(sac_path, format='SAC')
                success_count += 1
            except Exception:
                error_count += 1

        self.finished.emit(success_count, error_count)


class CleanUserHeadersThread(QThread):
    """Background thread to clean user headers (user0, user2, user3) after taup setsac"""
    finished = Signal(int, int)  # success_count, error_count

    def __init__(self, wavepath, suffix):
        super().__init__()
        self.wavepath = wavepath
        self.suffix = suffix

    def run(self):
        sac_files = sorted([f for f in os.listdir(self.wavepath) if f.lower().endswith(self.suffix.lower())])
        success_count = 0
        error_count = 0

        for sac_file in sac_files:
            sac_path = os.path.join(self.wavepath, sac_file)
            try:
                st = obspy.read(sac_path)
                sac_headers = st[0].stats.sac
                # Clear user headers that taup setsac writes (user0, user2, user3)
                should_write = False
                for user_header in ['user0', 'user2', 'user3']:
                    if user_header in sac_headers:
                        current_val = sac_headers[user_header]
                        if current_val != -12345.0:
                            sac_headers[user_header] = -12345.0
                            should_write = True

                if should_write:
                    st.write(sac_path, format='SAC')
                success_count += 1
            except Exception:
                error_count += 1

        self.finished.emit(success_count, error_count)

warnings.filterwarnings(
    "ignore",
    message=r"pkg_resources is deprecated as an API\..*",
    category=UserWarning,
)

import obspy

from WaveFigure import WaveFigure, _sac_float
from pierce_point_cache import DEFAULT_TAUP_BIN, PROJECT_ROOT, ensure_event_pierce_files, relative_event_path
from stack_system import (
    ensure_stack_workspace_dir,
    inspect_stack_event_health,
    is_stack_event_dir,
    quarantine_invalid_stack_files,
    resolve_stack_workspace_dir,
    source_event_dir_for_runtime,
    stack_sac_time_window,
    write_stack_workspace_index,
)
from dsm_fit_compare_dialog import DSMFitCompareWindow
from dsm_fit_compare_dialog import DSMFitContext as DSMFitCompareContext
from dsm_fit_compare_core import station_key as _dsm_station_key
from stack_thickness_review_dialog import ThicknessReviewContext, ThicknessReviewWindow


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _theory_summary_path_for_event(event_dir, model='iasp91'):
    output_root = Path(PROJECT_ROOT) / 'data' / 'output' / 'phases'
    return output_root / relative_event_path(event_dir) / f'theory_time_summary_{str(model).lower()}.json'


def _event_sac_paths(event_dir, suffix='.sac'):
    event_path = Path(event_dir).expanduser().resolve()
    suffix_lower = str(suffix or '.sac').lower()
    return sorted(
        path for path in event_path.iterdir()
        if path.is_file() and path.name.lower().endswith(suffix_lower)
    )


def ensure_event_theory_summary(event_dir, model='iasp91', suffix='.sac', taup_bin=DEFAULT_TAUP_BIN):
    cache_path = _theory_summary_path_for_event(event_dir, model=model)
    sac_paths = _event_sac_paths(event_dir, suffix=suffix)
    if cache_path.exists():
        try:
            with cache_path.open('r', encoding='utf-8') as handle:
                cached = json.load(handle)
            per_wave = cached.get('per_wave', {}) if isinstance(cached, dict) else {}
            cached_model = str(cached.get('model', '')).lower() if isinstance(cached, dict) else ''
            missing_waves = [path.name for path in sac_paths if path.name not in per_wave]
            if cached_model == str(model).lower() and isinstance(per_wave, dict) and not missing_waves:
                return cache_path
        except (OSError, ValueError, TypeError):
            pass

    if not sac_paths:
        return cache_path

    print(f'Preparing theory summary: {cache_path}')
    per_wave = {}
    pp_values = []
    sp_values = []
    evdp = math.nan
    gcarc_values = []
    model_key = str(model).lower()
    taup_exe = str(Path(taup_bin).expanduser().resolve())

    for sac_path in sac_paths:
        traces = obspy.read(str(sac_path), headonly=True)
        if len(traces) == 0:
            continue
        trace = traces[0]
        sac = getattr(trace.stats, 'sac', None)
        if sac is None:
            continue
        wave_name = sac_path.name
        gcarc = _sac_float(trace, 'gcarc', math.nan)
        local_evdp = _sac_float(trace, 'evdp', math.nan)
        if math.isnan(gcarc) or math.isnan(local_evdp):
            continue
        if math.isnan(evdp):
            evdp = local_evdp
        gcarc_values.append(gcarc)
        try:
            completed = subprocess.run(
                [
                    taup_exe,
                    'time',
                    '--onlytime',
                    '-h',
                    f'{local_evdp:g}',
                    '-p',
                    'P,pP,sP',
                    '--deg',
                    f'{gcarc:g}',
                    '--mod',
                    model_key,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (subprocess.SubprocessError, OSError, ValueError):
            continue
        values = []
        for token in (completed.stdout or '').replace('\n', ' ').split():
            try:
                values.append(float(token))
            except ValueError:
                continue
        if len(values) < 3:
            continue
        p_time, pp_time, sp_time = values[:3]
        delta_info = {
            'model': model_key,
            'P': p_time,
            'pP': pp_time,
            'sP': sp_time,
            'pP-P': pp_time - p_time,
            'sP-P': sp_time - p_time,
        }
        per_wave[wave_name] = delta_info
        pp_values.append(delta_info['pP-P'])
        sp_values.append(delta_info['sP-P'])

    summary = {
        'model': model_key,
        'event_name': Path(event_dir).expanduser().resolve().name,
        'evdp': evdp,
        'gcarc_min': float(np.min(gcarc_values)) if gcarc_values else math.nan,
        'gcarc_max': float(np.max(gcarc_values)) if gcarc_values else math.nan,
        'pP-P_mean': float(np.mean(pp_values)) if pp_values else math.nan,
        'sP-P_mean': float(np.mean(sp_values)) if sp_values else math.nan,
        'per_wave': per_wave,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2)
    print(f'Prepared theory summary: {cache_path}')
    return cache_path

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None,wavepath='', width=21, height=11,
                 dpi=100,xlim=None,order=None,tmarker=None,suffix=None,ta_tb=None,xlim_preview=None,axis_mode='absolute', member_filter=None):
        plt.rcParams['axes.unicode_minus'] = False

        self.wavefig = WaveFigure(wavepath, width=width, height=height, dpi=dpi, xlim=xlim,
                                  tmarker=tmarker,suffix=suffix,ta_tb=ta_tb,xlim_preview=xlim_preview, axis_mode=axis_mode, member_filter=member_filter)
        self.wavefig.init_canvas(order=order)


        FigureCanvas.__init__(self, self.wavefig.fig)
        self.setParent(parent)
        #
        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Expanding)
        # The canvas otherwise advertises its full figure size (width*dpi x
        # height*dpi) as its minimum, which pushes the window wider than the
        # screen on WSLg. Allow it to shrink so the window can fit the screen.
        self.setMinimumSize(0, 0)
        FigureCanvas.updateGeometry(self)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._force_visible_cursor()
        self.setFocus()

    def _force_visible_cursor(self):
        # Matplotlib changes Qt cursor shapes while hovering/dragging. On some
        # WSLg + Qt/XCB combinations, those themed cursor shapes can become
        # invisible. Keep the pick canvas on a plain arrow first; picking still
        # uses the real mouse event coordinates.
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def set_cursor(self, cursor):
        self._force_visible_cursor()

    def enterEvent(self, event):
        self._force_visible_cursor()
        return super().enterEvent(event)




class MatplotlibWidget(QMainWindow):
    ALIGN_OPTIONS = ['t0', 't2', 't3', 't7', 't6', 't5', 't1', 't4', 't8', 't9']
    MODE_OPTIONS = [('absolute', 'Absolute'), ('relative', 'Relative')]
    PICK_SHORTCUTS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'd', 'q']
    BP_PRESET_PLACEHOLDER = 'BP presets'
    PHASE_PRESET_PLACEHOLDER = 'Phase presets'
    PREVIEW_SHORTCUTS = {
        't7': 'u',
        't6': 'y',
        't5': 't',
        't0': 'p',
        't2': 'w',
        't3': 'e',
    }

    def __init__(self, wavepath, xlim=[-10,10],
                 order='gcarc', tmarker=None ,parent=None, suffix=None,ta_tb=None,xlim_preview=None, member_filter=None):
        super(MatplotlibWidget, self).__init__(parent)
        self.initUi(wavepath, xlim, order=order, tmarker=tmarker,suffix=suffix,ta_tb=ta_tb,xlim_preview=xlim_preview, member_filter=member_filter)

    def _force_visible_cursor(self, *widgets):
        cursor = QCursor(Qt.CursorShape.ArrowCursor)
        self.setCursor(cursor)
        for widget in widgets:
            if widget is not None:
                widget.setCursor(cursor)

    def initUi(self, wavepath, xlim, order, tmarker, suffix,ta_tb,xlim_preview, member_filter=None):
        self.layout = QVBoxLayout()
        self.pending_tmarker = tmarker or 't0'
        self.stack_mode_for_controls = is_stack_event_dir(wavepath)
        self.pending_axis_mode = 'absolute' if self.stack_mode_for_controls else 'relative'
        self.x1 = xlim[0]
        self.x2 = xlim[1]
        self.bp_freqmin = 0.05
        self.bp_freqmax = 0.4
        self.bp_corners = 2
        self.bp_passes = 2
        self.bp_preset_path = join(dirname(__file__), 'bp_presets.json')
        self.phase_preset_path = join(dirname(__file__), 'phase_presets.json')
        self.preview_phases = [item.strip() for item in ta_tb.split(',') if item.strip()]
        self.bp_presets = self._load_bp_presets()
        self.phase_presets = self._load_phase_presets()
        self.tmaker = self.pending_tmarker
        self.axis_mode = self.pending_axis_mode
        self.suffix = suffix
        self.preview_shortcuts = []
        self.station_search_visible = False
        # Theory phase marker settings
        self.theory_model = 'iasp91'
        self.theory_phases = 'pP-2,sP-3'
        self.taup_bin = str(DEFAULT_TAUP_BIN)
        self.add_btn()
        self.mpl = MyMplCanvas(self, wavepath= wavepath, width=21, height=11,
                               dpi=100, xlim=xlim, order=order, tmarker= tmarker,
                               suffix=suffix,ta_tb=ta_tb,xlim_preview=xlim_preview, axis_mode=self.axis_mode, member_filter=member_filter)
        self.mpl.wavefig.jump_status_callback = self.show_jump_status
        self.mpl.wavefig.status_callback = self.show_status_message
        self.mpl.wavefig.phase_tokens_change_callback = self.sync_phase_preset_from_wavefig
        self.mpl.wavefig.stack_review_refresh_callback = self._refresh_open_stack_thickness_reviews
        self.layout.addWidget(self.mpl, 2)
        self.mpl.mpl_connect('button_press_event', self.on_click)
        self.mpl.mpl_connect('key_press_event', self.keyPressEvent)
        QTimer.singleShot(0, self.mpl.setFocus)
        #
        main_frame = QWidget()
        self.setCentralWidget(main_frame)
        main_frame.setLayout(self.layout)
        self._force_visible_cursor(main_frame)
        # Keep the root layout unconstrained so the main window can still
        # maximize correctly on narrower screens. The key fix was not a custom
        # geometry hack but making the control area shrinkable: long control
        # rows are split across multiple rows, and large line edits / preset
        # combos use minimum-width + expanding size policies instead of only
        # fixed widths.
        self.layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        saveAction = QAction('&Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save this figure')
        saveAction.triggered.connect(self.plot_save)

        stackHealthAction = QAction('Stack Health', self)
        stackHealthAction.setStatusTip('Inspect current stack event health')
        stackHealthAction.triggered.connect(self.inspect_stack_health)

        stackIndexAction = QAction('Refresh Stack Index', self)
        stackIndexAction.setStatusTip('Rewrite the current stack workspace index')
        stackIndexAction.triggered.connect(self.refresh_stack_index)

        stackQuarantinePreviewAction = QAction('Preview Invalid Stack Quarantine', self)
        stackQuarantinePreviewAction.setStatusTip('Preview invalid stack SAC files that would be quarantined')
        stackQuarantinePreviewAction.triggered.connect(self.preview_invalid_stack_quarantine)

        stackQuarantineAction = QAction('Quarantine Invalid Stack Files', self)
        stackQuarantineAction.setStatusTip('Move invalid stack SAC files into a quarantine directory')
        stackQuarantineAction.triggered.connect(self.quarantine_invalid_stack_files)

        openStackWorkspaceAction = QAction('Open Stack Workspace', self)
        openStackWorkspaceAction.setStatusTip('Open the stack workspace for this event')
        openStackWorkspaceAction.triggered.connect(self.open_stack_workspace)

        openSourceEventAction = QAction('Open Source Event', self)
        openSourceEventAction.setStatusTip('Open the source event workspace for this stack event')
        openSourceEventAction.triggered.connect(self.open_source_event_workspace)

        dsmFitAction = QAction('DSM 拟合对比…', self)
        dsmFitAction.setStatusTip('对比当前事件观测波形与 DSM 正演模拟波形 (黑=观测 / 红=理论)')
        dsmFitAction.triggered.connect(self.open_dsm_fit_compare)

        thicknessReviewAction = QAction('Stack 厚度审阅…', self)
        thicknessReviewAction.setStatusTip('汇集目录下所有 stack group 的穿透点 + 地壳厚度，三视图联动审阅')
        thicknessReviewAction.triggered.connect(self.open_stack_thickness_review)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(saveAction)
        fileMenu.addSeparator()
        if self.mpl.wavefig.stack_mode:
            fileMenu.addAction(openSourceEventAction)
        else:
            fileMenu.addAction(openStackWorkspaceAction)
            fileMenu.addAction(dsmFitAction)
            fileMenu.addAction(thicknessReviewAction)
        if self.mpl.wavefig.stack_mode:
            fileMenu.addSeparator()
            fileMenu.addAction(stackHealthAction)
            fileMenu.addAction(stackIndexAction)
            fileMenu.addAction(stackQuarantinePreviewAction)
            fileMenu.addAction(stackQuarantineAction)

        self._set_geom_center()
        self._define_global_shortcuts()
        self.setWindowTitle('DePhaseKit Stack' if self.mpl.wavefig.stack_mode else 'DePhaseKit')
        # print(dirname(__file__))
        self.setWindowIcon(QIcon(join(dirname(__file__), 'dpk.png')))
        QTimer.singleShot(0, lambda: self.show_status_message('', timeout_ms=0))
        QTimer.singleShot(0, self._populate_event_combo)
        QTimer.singleShot(250, self._warm_default_preview_resources)

    def _warm_default_preview_resources(self):
        try:
            self.mpl.wavefig.warm_preview_resources(0)
        except Exception:
            pass

    # ----- same-directory event switching -----
    def _sibling_event_dirs(self):
        """Sibling event directories (same parent) that contain waveforms of
        the current suffix. Returns naturally-sorted absolute paths."""
        try:
            wavepath = self.mpl.wavefig.wavepath
        except Exception:
            return []
        if not wavepath:
            return []
        parent = os.path.dirname(os.path.abspath(str(wavepath).rstrip(os.sep)))
        if not parent or not os.path.isdir(parent):
            return []
        suffix = str(getattr(self, 'suffix', '.sac') or '.sac').lower()
        siblings = []
        for name in os.listdir(parent):
            full = os.path.join(parent, name)
            if not os.path.isdir(full):
                continue
            try:
                has_wave = any(
                    fn.lower().endswith(suffix) for fn in os.listdir(full)
                )
            except OSError:
                has_wave = False
            if has_wave:
                siblings.append(full)
        from stack_system import _natural_sort_key
        return sorted(siblings, key=lambda p: _natural_sort_key(os.path.basename(p)))

    def _populate_event_combo(self):
        combo = getattr(self, 'event_combo', None)
        if combo is None:
            return
        try:
            current = os.path.abspath(str(self.mpl.wavefig.wavepath).rstrip(os.sep))
        except Exception:
            current = ''
        siblings = self._sibling_event_dirs()
        self._event_combo_updating = True
        try:
            combo.blockSignals(True)
            combo.clear()
            for path in siblings:
                combo.addItem(os.path.basename(path), path)
            idx = combo.findData(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        finally:
            combo.blockSignals(False)
            self._event_combo_updating = False

    def _switch_event_by_offset(self, offset):
        siblings = self._sibling_event_dirs()
        if not siblings:
            self.show_status_message('No sibling event directories found', timeout_ms=5000)
            return
        try:
            current = os.path.abspath(str(self.mpl.wavefig.wavepath).rstrip(os.sep))
        except Exception:
            return
        try:
            index = siblings.index(current)
        except ValueError:
            index = -1
        next_index = (index + offset) % len(siblings) if siblings else 0
        self._switch_to_event(siblings[next_index])

    def _switch_to_event_combo_index(self, index):
        if self._event_combo_updating:
            return
        combo = getattr(self, 'event_combo', None)
        if combo is None or index < 0 or index >= combo.count():
            return
        path = combo.itemData(index)
        if not path:
            return
        self._switch_to_event(path)

    def _switch_to_event(self, path):
        if not path or not exists(path):
            return
        try:
            current = os.path.abspath(str(self.mpl.wavefig.wavepath).rstrip(os.sep))
        except Exception:
            current = ''
        if os.path.abspath(path) == current:
            return
        # Flush this window's markers to disk first (sync medium between windows).
        window = self._open_workspace_window(path)
        if window is None:
            return
        # Close the current window so it feels like switching events (the new
        # window keeps the app alive).
        QTimer.singleShot(0, self.close)


    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QEvent.Type.ActivationChange and self.isActiveWindow():
            QTimer.singleShot(0, self.restore_pick_focus)

    def restore_pick_focus(self):
        if getattr(self, 'station_search_visible', False):
            return
        if QApplication.focusWidget() in {
            getattr(self, 'x1_input', None),
            getattr(self, 'x2_input', None),
            getattr(self, 'bp_freqmin_input', None),
            getattr(self, 'bp_freqmax_input', None),
            getattr(self, 'bp_corners_input', None),
            getattr(self, 'bp_passes_input', None),
        }:
            self.mpl.setFocus()

    def on_click(self, event):
        wavefig = self.mpl.wavefig
        changed = wavefig.onclick(event)
        if changed:
            if getattr(wavefig, '_last_click_refresh_needed', True):
                wavefig.refresh_current_page()
            self.show_status_message('', timeout_ms=0)
        # Stack 子系统：点击主图波形后联动预览窗的穿透点。onclick 里
        # _remember_pick_wave 已更新 current_pick_wave_name，但 stack 预览
        # 的穿透点集合由 plotfig._stack_preview_wave_name 决定——不刷新,
        # 高亮不跟点击走；换行点另一 stack wave 也不换 group。
        if (getattr(wavefig, 'stack_mode', False)
                and wavefig.plotfig is not None
                and plt.fignum_exists(wavefig.plotfig.number)
                and wavefig.current_pick_wave_name
                and wavefig.current_pick_wave_name in wavefig._stack_preview_stack_wave_names()):
            preview_index = getattr(
                wavefig.plotfig, '_preview_controls', {}).get('preview_index', 0)
            current = wavefig.current_pick_wave_name
            if current != getattr(wavefig.plotfig, '_stack_preview_wave_name', None):
                # 切到另一 stack group：同步 wave name 并重载预览（含穿透点）。
                wavefig.plotfig._stack_preview_wave_name = current
                wavefig._refresh_preview_figure(wavefig.plotfig, preview_index)
            else:
                # 同 group：只同步选中高亮（轻量，不重载 member stream）。
                wavefig._sync_preview_to_current_pick(wavefig.plotfig)
        self.mpl.draw()

    def keyPressEvent(self, event: QKeyEvent):
        try:
            key_text = event.text().strip().lower()
        except Exception:
            key_text = ''

        key_value = getattr(event, 'key', None)
        if callable(key_value):
            try:
                key_value = key_value()
            except Exception:
                key_value = None

        if key_text == ' ' or key_value in (' ', 'space', 'Space', Qt.Key.Key_Space):
            accept = getattr(event, 'accept', None)
            if callable(accept):
                accept()
            return
        self.mpl.wavefig.onkeypress(event)


    def previous_connect(self):
        self.mpl.wavefig.butprevious()
        self.mpl.draw()

    def next_connect(self):
        self.mpl.wavefig.butnext()
        self.mpl.draw()

    def show_status_message(self, message, timeout_ms=3000):
        alignment_message = self.mpl.wavefig.alignment_status_summary()
        message = str(message or '').strip()
        mode_label = 'Stack mode' if self.mpl.wavefig.stack_mode else ''
        parts = [part for part in (message, alignment_message, mode_label) if part]
        full_message = '  |  '.join(parts)
        self.statusBar().showMessage(full_message, timeout_ms)

    def inspect_stack_health(self):
        if not self.mpl.wavefig.stack_mode:
            self.show_status_message('Stack health is only available in stack mode', timeout_ms=5000)
            return
        report = inspect_stack_event_health(self.mpl.wavefig.wavepath)
        self.mpl.wavefig.stack_health_report = report
        invalid_count = len(report.get('invalid_sac_files', []))
        missing_sidecars = len(report.get('missing_sidecars', []))
        orphan_sidecars = len(report.get('orphan_sidecars', []))
        invalid_sidecars = len(report.get('invalid_sidecars', []))
        repairable_sidecars = len(report.get('sidecars_needing_repair', []))
        valid_count = int(report.get('valid_sac_count', 0))
        message = (
            f'Stack health: valid {valid_count}, invalid {invalid_count}, '
            f'missing sidecar {missing_sidecars}, orphan sidecar {orphan_sidecars}, '
            f'invalid sidecar {invalid_sidecars}, repairable sidecar {repairable_sidecars}'
        )
        self.show_status_message(message, timeout_ms=8000)

    def refresh_stack_index(self):
        if not self.mpl.wavefig.stack_mode:
            self.show_status_message('Stack index is only available in stack mode', timeout_ms=5000)
            return
        index = write_stack_workspace_index(self.mpl.wavefig.wavepath)
        self.mpl.wavefig.stack_health_report = index.get('health', {})
        stack_count = int(index.get('stack_count', 0) or 0)
        valid_count = int(index.get('valid_stack_count', 0) or 0)
        invalid_count = int(index.get('invalid_stack_count', 0) or 0)
        self.show_status_message(
            f'Stack index refreshed: total {stack_count}, valid {valid_count}, invalid {invalid_count}',
            timeout_ms=8000,
        )

    def preview_invalid_stack_quarantine(self):
        if not self.mpl.wavefig.stack_mode:
            self.show_status_message('Stack quarantine preview is only available in stack mode', timeout_ms=5000)
            return
        report = quarantine_invalid_stack_files(self.mpl.wavefig.wavepath, persist=False)
        invalid_count = int(report.get('invalid_count', 0) or 0)
        self.mpl.wavefig.stack_health_report = inspect_stack_event_health(self.mpl.wavefig.wavepath)
        if invalid_count == 0:
            self.show_status_message('No invalid stack files would be quarantined', timeout_ms=5000)
            return
        preview_names = []
        for item in report.get('moved', [])[:3]:
            preview_names.append(os.path.basename(str(item.get('path', ''))))
        preview_text = ', '.join(name for name in preview_names if name)
        if invalid_count > len(preview_names):
            preview_text = f'{preview_text}, ...' if preview_text else '...'
        message = f'Would quarantine {invalid_count} invalid stack file(s)'
        if preview_text:
            message = f'{message}: {preview_text}'
        self.show_status_message(message, timeout_ms=10000)

    def quarantine_invalid_stack_files(self):
        if not self.mpl.wavefig.stack_mode:
            self.show_status_message('Stack quarantine is only available in stack mode', timeout_ms=5000)
            return
        report = quarantine_invalid_stack_files(self.mpl.wavefig.wavepath, persist=True)
        moved_count = int(report.get('invalid_count', 0))
        quarantine_dir = report.get('quarantine_dir', '')
        self.mpl.wavefig.stack_health_report = inspect_stack_event_health(self.mpl.wavefig.wavepath)
        write_stack_workspace_index(self.mpl.wavefig.wavepath)
        if moved_count == 0:
            self.show_status_message('No invalid stack files to quarantine', timeout_ms=5000)
            return
        self.show_status_message(
            f'Quarantined {moved_count} invalid stack file(s) to {quarantine_dir}',
            timeout_ms=10000,
        )

    def _open_workspace_window(self, wavepath, member_filter=None):
        wavepath = str(wavepath)
        if not exists(wavepath):
            self.show_status_message(f'No such directory: {wavepath}', timeout_ms=6000)
            return None
        # Flush this window's in-memory markers to disk so the new window (a
        # separate WaveFigure sharing the same source SAC files) reads fresh
        # values. Without this, original-event edits made this session stay in
        # memory only (finish() runs on quit) and the stack window sees stale disk.
        try:
            self.mpl.wavefig._persist_markers_to_disk()
        except Exception as exc:
            print(f'warn: marker flush before opening workspace failed: {exc}')
        child_xlim = [self.x1, self.x2]
        if is_stack_event_dir(wavepath):
            child_xlim = stack_sac_time_window(wavepath, suffix='.sac') or child_xlim
            # stack 工作区目录固定输出 .sac 文件，必须用 .sac 后缀匹配；
            # 不能继承父窗口的 suffix（如 .bhz），否则 iter_stack_sac_paths 匹配不到文件。
            child_suffix = '.sac'
        elif Path(wavepath).resolve().parent.name in {'pick_jandy', 'pick_other'}:
            # 源事件目录同样固定存放 *.sac。这里若继承父窗口 suffix（例如
            # .bhz / .hhz 上下文），按 group 直跳主 ppk 时会先把事件目录筛空，
            # 直接报 "No valid waveforms in ...".
            child_suffix = '.sac'
        else:
            child_suffix = self.suffix or '.sac'
        new_window = MatplotlibWidget(
            wavepath,
            xlim=child_xlim,
            order=self.mpl.wavefig.order,
            tmarker=self.pending_tmarker,
            suffix=child_suffix,
            ta_tb=','.join(self.preview_phases),
            xlim_preview=None,
            member_filter=member_filter,
        )
        new_window.show()
        # 与主拾取窗一致：show 后强制铺满工作区，避免子窗位置/尺寸与主窗不一致。
        try:
            new_window._set_geom_center()
        except Exception:
            try:
                maximize_on_workarea(new_window, frac=0.98)
            except Exception:
                pass
        # 激活新窗口并抢占焦点：从厚度审阅窗等非主 ppk 路径打开时，show() 不会
        # 让新窗口成为活动窗口，其 QShortcut(WindowShortcut) 不触发 → 预览/拾取
        # 快捷键失效。raise_+activateWindow+延迟 setFocus 确保焦点落到 canvas。
        try:
            new_window.raise_()
            new_window.activateWindow()
        except Exception:
            pass
        def _grab_focus():
            try:
                new_window.raise_()
                new_window.activateWindow()
                if getattr(new_window, 'mpl', None) is not None:
                    new_window.mpl.setFocus()
            except Exception:
                pass
        QTimer.singleShot(0, _grab_focus)
        if not hasattr(self, '_child_windows'):
            self._child_windows = []
        self._child_windows.append(new_window)
        return new_window

    def _refresh_open_stack_thickness_reviews(self):
        app = QApplication.instance()
        if app is None:
            return
        for widget in app.topLevelWidgets():
            if isinstance(widget, ThicknessReviewWindow):
                try:
                    widget.request_external_refresh()
                except Exception:
                    pass

    def open_stack_workspace(self):
        source_event_dir = self.mpl.wavefig.runtime_event_dir
        stack_dir = str(ensure_stack_workspace_dir(source_event_dir))
        health = inspect_stack_event_health(stack_dir)
        valid_stack_count = int(health.get('valid_sac_count', 0) or 0)
        if valid_stack_count == 0:
            invalid_stack_count = len(health.get('invalid_sac_files', []) or [])
            suffix = str(self.suffix or '.sac')
            detail = (
                f'no valid stack {suffix} yet'
                if invalid_stack_count
                else f'no stack {suffix} yet'
            )
            message = f'Stack workspace not opened: {stack_dir} ({detail})'
            self.show_status_message(message, timeout_ms=10000)
            print(message)
            return
        window = self._open_workspace_window(stack_dir)
        if window is None:
            return
        message = f'Opened stack workspace: {stack_dir}'
        self.show_status_message(message, timeout_ms=6000)
        print(message)

    def open_source_event_workspace(self):
        if not self.mpl.wavefig.stack_mode:
            self.show_status_message('Source event workspace is only available in stack mode', timeout_ms=5000)
            return
        source_event_dir = self.mpl.wavefig.runtime_event_dir
        window = self._open_workspace_window(source_event_dir)
        if window is None:
            return
        self.show_status_message(f'Opened source event workspace: {source_event_dir}', timeout_ms=6000)

    def open_dsm_fit_compare(self):
        """Open the DSM forward-modeling fit comparison dialog for this event.

        根据主窗当前状态自动配置：
          - 场景A：打开原始事件且有可见集合(预览/拾取窗隐藏后留下的波形) → 按 NET.STA
            Jaccard 自动选最匹配的 dsm group，obs 限定为可见集合台站。
          - 场景B：主窗用 -s .bhz 直接打开某 dsm group 目录 → synth=该 group，obs=原始事件目录。
          - manual：无可见集合且非 dsm 树 → 保留旧手动流程。
        """
        wf = self.mpl.wavefig
        wavepath = str(getattr(wf, 'wavepath', '') or '')
        runtime_event_dir = str(getattr(wf, 'runtime_event_dir', '') or wavepath)
        if not wavepath or not exists(wavepath):
            self.show_status_message(f'No valid event directory for DSM fit compare: {wavepath}', timeout_ms=6000)
            return
        suffix = self.suffix or '.sac'

        # dsm 树检测（场景B）：wavepath 在 data/dsm 下。
        dsm_root = (PROJECT_ROOT / "data" / "dsm").resolve()
        try:
            Path(wavepath).resolve().relative_to(dsm_root)
            is_dsm_tree = True
        except (ValueError, OSError):
            is_dsm_tree = False

        # event_id：场景B 取 group 的父目录名（runtime_event_dir 此时=wavepath，
        # 直接 .name 会错取 group 名）；否则取 runtime_event_dir 的 basename。
        if is_dsm_tree:
            event_id = Path(wavepath).parent.name
        else:
            event_id = Path(runtime_event_dir).name

        # 可见集合（场景A 的 obs 过滤器）。
        hidden = {str(n) for n in getattr(wf, 'preview_hidden_wave_names', set())}
        ori = getattr(wf, 'ori_sacnames', None)
        visible_keys: set[str] = set()
        if ori is not None:
            for name in ori:
                sname = str(name)
                if sname in hidden:
                    continue
                visible_keys.add(_dsm_station_key(Path(os.path.basename(sname))))

        # obs_dir / synth_dir / scenario
        if is_dsm_tree:
            synth_dir = wavepath
            obs_dir = None
            for pick_root in (PROJECT_ROOT / "data" / "pick_jandy", PROJECT_ROOT / "data" / "pick_other"):
                cand = pick_root / event_id
                if cand.is_dir():
                    obs_dir = str(cand)
                    break
            scenario = 'B'
            visible_keys = set()
        elif visible_keys:
            synth_dir = None
            obs_dir = runtime_event_dir
            scenario = 'A'
        else:
            synth_dir = None
            obs_dir = runtime_event_dir
            scenario = 'manual'

        context = DSMFitCompareContext(
            wavepath=wavepath,
            runtime_event_dir=runtime_event_dir,
            suffix=suffix,
            is_dsm_tree=is_dsm_tree,
            event_id=event_id,
            obs_dir=obs_dir,
            synth_dir=synth_dir,
            visible_station_keys=visible_keys,
            scenario=scenario,
        )
        dialog = DSMFitCompareWindow(context, parent=self)
        dialog.show()
        if not hasattr(self, '_child_windows'):
            self._child_windows = []
        self._child_windows.append(dialog)
        self.show_status_message(f'DSM 拟合对比: {event_id} ({scenario})', timeout_ms=5000)

    def open_stack_thickness_review(self):
        """打开 Stack 厚度审阅窗：汇集目录下所有 stack group 的穿透点 + 地壳厚度。

        默认扫描 ``data/output/stack/analysis``；可在窗内"选目录"切换。右键某 group
        可"在主 ppk 打开此事件"——打开该事件的拾取窗并自动隐藏非该 group 的台站，
        一开窗就只看该 group 成员。
        """
        default_root = str(PROJECT_ROOT / "data" / "output" / "stack" / "analysis")

        def _open_event_in_ppk(event_dir, group_name):
            members = self._read_group_members(event_dir, group_name)
            self._open_workspace_window(event_dir, member_filter=members)
            if members:
                self.show_status_message(
                    f'已打开 {group_name}（{len(members)} 台）', timeout_ms=4000)

        def _open_stack_in_ppk(event_dir, stack_wave_name):
            # 打开该 group 的 stack 工作区拾取窗，主图只显示该 stack.sac
            stack_dir = str(ensure_stack_workspace_dir(event_dir))
            self._open_workspace_window(
                stack_dir, member_filter={stack_wave_name})
            self.show_status_message(
                f'已打开 stack: {stack_wave_name}', timeout_ms=4000)

        context = ThicknessReviewContext(
            scan_root=default_root,
            open_event_callback=_open_event_in_ppk,
            open_stack_callback=_open_stack_in_ppk,
        )
        dialog = ThicknessReviewWindow(context, parent=self)
        dialog.show()
        if not hasattr(self, '_child_windows'):
            self._child_windows = []
        self._child_windows.append(dialog)
        self.show_status_message('Stack 厚度审阅已打开', timeout_ms=4000)

    def _read_group_members(self, event_dir, group_name):
        """读 ``data/output/process/group/<event>/groupN.txt`` 的成员 wave_name。

        返回 set[str]；文件不存在或 group 名非法时返回 None（=不过滤，读全部）。
        """
        try:
            raw = str(group_name or '').strip().lower()
            if raw.startswith('group'):
                raw = raw[5:]
            if not raw.isdigit():
                return None
            normalized = f'group{int(raw)}'
            group_dir = PROJECT_ROOT / 'data' / 'output' / 'process' / 'group' / str(
                relative_event_path(event_dir))
            txt_path = group_dir / f'{normalized}.txt'
            if not txt_path.is_file():
                self.show_status_message(
                    f'group 文件不存在: {normalized}', timeout_ms=5000)
                return None
            members = set()
            with open(txt_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    wave_name = line.split('\t', 1)[0].strip()
                    if wave_name:
                        members.add(wave_name)
            return members or None
        except Exception as exc:
            print(f'warn: read group members failed: {exc}')
            return None

    def _marked_status_text(self, wave_name):
        mode_label = self.mpl.wavefig.jump_target_mode_label()
        position, total = self.mpl.wavefig.marked_wave_position(wave_name)
        if total == 0:
            return f'{mode_label} 0/0'
        if position is None:
            return f'{mode_label} -/{total}'
        return f'{mode_label} {position}/{total}'

    def set_jump_target_mode(self, mode):
        if not self.mpl.wavefig.set_jump_target_mode(mode):
            return
        target_index = self.mpl.wavefig.focus_first_jump_target_wave(mode=mode, refresh=True)
        mode_label = self.mpl.wavefig.jump_target_mode_label()
        if target_index is None:
            self.show_status_message(f'No {mode_label} waveforms found')
            self.mpl.draw()
            return
        wave_name = self.mpl.wavefig.ori_sacnames[target_index]
        station_name = self.mpl.wavefig.current_pick_station_name or wave_name
        page_number = (target_index // self.mpl.wavefig.maxidx) + 1
        marked_text = self._marked_status_text(wave_name)
        self.show_status_message(f'{marked_text}  {station_name} on page {page_number}')
        self.mpl.draw()

    def show_jump_status(self, wave_name):
        if not wave_name:
            return
        try:
            target_index = self.mpl.wavefig.ori_sacnames.index(wave_name)
        except ValueError:
            return
        station_name = self.mpl.wavefig.current_pick_station_name or wave_name
        page_number = (target_index // self.mpl.wavefig.maxidx) + 1
        marked_text = self._marked_status_text(wave_name)
        self.show_status_message(f'{marked_text}  {station_name} on page {page_number}')

    def jump_to_marked_wave(self, step):
        target_index = self.mpl.wavefig.jump_to_marked_wave(step=step)
        if target_index is None:
            mode_label = self.mpl.wavefig.jump_target_mode_label()
            self.show_status_message(f'No {mode_label} waveforms found')
            return
        station_name = self.mpl.wavefig.current_pick_station_name or self.mpl.wavefig.ori_sacnames[target_index]
        page_number = (target_index // self.mpl.wavefig.maxidx) + 1
        marked_text = self._marked_status_text(self.mpl.wavefig.ori_sacnames[target_index])
        self.show_status_message(f'{marked_text}  {station_name} on page {page_number}')
        self.mpl.draw()

    def jump_to_missing_alignment_wave(self, step=1):
        target_index = self.mpl.wavefig.jump_to_missing_alignment_wave(step=step)
        if target_index is None:
            phase_label = self.mpl.wavefig._phase_display_label(self.mpl.wavefig.tmarker)
            self.show_status_message(f'No missing {phase_label} picks')
            return
        wave_name = self.mpl.wavefig.ori_sacnames[target_index]
        station_name = self.mpl.wavefig.current_pick_station_name or wave_name
        page_number = (target_index // self.mpl.wavefig.maxidx) + 1
        phase_label = self.mpl.wavefig._phase_display_label(self.mpl.wavefig.tmarker)
        self.show_status_message(f'Missing {phase_label}: {station_name} on page {page_number}')
        self.mpl.draw()

    def focus_station_search(self):
        self.station_search_visible = True
        self.station_search_input.show()
        self.station_search_input.setFocus()
        self.station_search_input.selectAll()
        self.show_status_message('Type station name and press Enter')

    def close_topmost_window(self):
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, QLineEdit):
            return
        if self.mpl.wavefig.close_preview_window():
            self.show_status_message('Closed preview window')
            return
        self.show_status_message('Finish and close DePhaseKit')
        self.finish()

    def hide_station_search(self):
        self.station_search_visible = False
        self.station_search_input.clearFocus()
        self.station_search_input.hide()

    def jump_to_station_search(self):
        query = self.station_search_input.text().strip()
        if query == '':
            self.show_status_message('Station search is empty')
            self.hide_station_search()
            return
        target_index = self.mpl.wavefig.find_wave_index(query)
        if target_index is None:
            self.show_status_message(f'No match for "{query}"')
            self.station_search_input.setFocus()
            self.station_search_input.selectAll()
            return
        wave_name = self.mpl.wavefig.ori_sacnames[target_index]
        self.mpl.wavefig.jump_to_wave_name(wave_name, refresh=True)
        station_name = self.mpl.wavefig.current_pick_station_name or wave_name
        page_number = (target_index // self.mpl.wavefig.maxidx) + 1
        marked_text = self._marked_status_text(wave_name)
        self.show_status_message(f'Found {station_name} on page {page_number}  {marked_text}')
        self.hide_station_search()
        self.mpl.draw()

    def getx1(self):
        x1_text = self.x1_input.text().strip() # get text
        if x1_text in {"", "-", "+"}:
            return
        try:
            self.x1 = int(x1_text)
        except ValueError:
            return

    def getx2(self):
        x2_text = self.x2_input.text().strip()  # get text
        if x2_text in {"", "-", "+"}:
            return
        try:
            self.x2 = int(x2_text)
        except ValueError:
            return

    def C_time_window(self):
        x1 = self.x1
        x2 = self.x2
        self.tmaker = self.pending_tmarker
        self.axis_mode = self.pending_axis_mode
        self.mpl.wavefig.set_view_settings(self.tmaker, [x1, x2], self.axis_mode)
        self.mpl.wavefig.sync_preview_window_for_marker(self.tmaker, [x1, x2])
        self.show_status_message(
            f'View refreshed: {self.tmaker} [{x1}, {x2}] {self.axis_mode}',
            timeout_ms=4000,
        )
        self.mpl.draw()

    def apply_bandpass_settings(self):
        x1 = self.x1
        x2 = self.x2
        self.tmaker = self.pending_tmarker
        self.axis_mode = self.pending_axis_mode
        success, message = self.mpl.wavefig.apply_sac_bandpass_and_reload(
            freqmin=self.bp_freqmin,
            freqmax=self.bp_freqmax,
            corners=self.bp_corners,
            passes=self.bp_passes,
            order=self.mpl.wavefig.order,
        )
        if not success:
            self.show_status_message(message, timeout_ms=5000)
            return
        self.mpl.wavefig.set_view_settings(self.tmaker, [x1, x2], self.axis_mode)
        self.mpl.wavefig.sync_preview_window_for_marker(self.tmaker, [x1, x2])
        self.show_status_message(message, timeout_ms=5000)
        self.mpl.draw()

    def show_raw_waveforms(self):
        x1 = self.x1
        x2 = self.x2
        self.tmaker = self.pending_tmarker
        self.axis_mode = self.pending_axis_mode
        self.mpl.wavefig.read_sac(order=self.mpl.wavefig.order)
        self.mpl.wavefig.refresh_current_page()
        self.mpl.wavefig.set_view_settings(self.tmaker, [x1, x2], self.axis_mode)
        self.mpl.wavefig.sync_preview_window_for_marker(self.tmaker, [x1, x2])
        self.show_status_message('Reloaded current SAC files from disk', timeout_ms=5000)
        self.mpl.draw()

    def restore_from_backup_path(self):
        backup_path = self.backup_restore_input.text().strip()
        if backup_path == '':
            self.show_status_message('Backup event path is empty', timeout_ms=5000)
            return
        success, message, match_summary = self.mpl.wavefig.restore_event_from_backup(
            backup_path,
            order=self.mpl.wavefig.order,
        )
        if match_summary:
            self.show_status_message(f'{match_summary}  {message}', timeout_ms=5000)
        else:
            self.show_status_message(message, timeout_ms=5000)
        if success:
            self.mpl.draw()

    def get_bp_freqmin(self):
        text = self.bp_freqmin_input.text().strip()
        if text in {"", "-", "+", ".", "-.", "+."}:
            return
        try:
            self.bp_freqmin = float(text)
        except ValueError:
            return

    def get_bp_freqmax(self):
        text = self.bp_freqmax_input.text().strip()
        if text in {"", "-", "+", ".", "-.", "+."}:
            return
        try:
            self.bp_freqmax = float(text)
        except ValueError:
            return

    def get_bp_corners(self):
        text = self.bp_corners_input.text().strip()
        if text in {"", "-", "+"}:
            return
        try:
            self.bp_corners = int(text)
        except ValueError:
            return

    def get_bp_passes(self):
        text = self.bp_passes_input.text().strip()
        if text in {"", "-", "+"}:
            return
        try:
            self.bp_passes = int(text)
        except ValueError:
            return

    def _default_bp_preset(self):
        return {
            'freqmin': 0.05,
            'freqmax': 0.4,
            'corners': 2,
            'passes': 2,
        }

    def _default_phase_preset(self):
        preview_tokens = ','.join(self.preview_phases)
        normalized = self._normalize_phase_preset(preview_tokens)
        return normalized or ''

    def _normalize_bp_preset(self, profile):
        if not isinstance(profile, dict):
            return None
        try:
            freqmin = float(profile['freqmin'])
            freqmax = float(profile['freqmax'])
            corners = int(profile['corners'])
            passes = int(profile['passes'])
        except (KeyError, TypeError, ValueError):
            return None
        if freqmin <= 0 or freqmax <= freqmin or corners < 1 or passes < 1:
            return None
        return {
            'freqmin': freqmin,
            'freqmax': freqmax,
            'corners': corners,
            'passes': passes,
        }

    def _bp_preset_key(self, profile):
        return (
            round(float(profile['freqmin']), 6),
            round(float(profile['freqmax']), 6),
            int(profile['corners']),
            int(profile['passes']),
        )

    def _bp_preset_label(self, profile):
        return (
            f"BP c {profile['freqmin']:g} {profile['freqmax']:g} "
            f"n {int(profile['corners'])} p {int(profile['passes'])}"
        )

    def _current_bp_preset(self):
        return self._normalize_bp_preset({
            'freqmin': self.bp_freqmin,
            'freqmax': self.bp_freqmax,
            'corners': self.bp_corners,
            'passes': self.bp_passes,
        })

    def _normalize_phase_preset(self, preset_text):
        raw_tokens = [item.strip() for item in str(preset_text or '').split(',') if item.strip()]
        if not raw_tokens:
            return None
        phase_keys = []
        for token in raw_tokens:
            token_lower = token.lower()
            if token_lower.startswith('t') and token_lower[1:].isdigit():
                marker_key = token_lower[1:]
            elif token_lower.isdigit():
                marker_key = token_lower
            else:
                return None
            if marker_key not in {str(i) for i in range(10)}:
                return None
            if marker_key not in phase_keys:
                phase_keys.append(marker_key)
        return ','.join(f"t{marker_key}" for marker_key in phase_keys)

    def _phase_preset_label(self, preset_text):
        return str(preset_text)

    def _load_bp_presets(self):
        if not exists(self.bp_preset_path):
            return [self._default_bp_preset()]
        try:
            with open(self.bp_preset_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            return [self._default_bp_preset()]
        if isinstance(loaded, dict):
            raw_presets = loaded.get('presets', [])
        else:
            raw_presets = loaded
        presets = []
        seen = set()
        if not isinstance(raw_presets, list):
            return presets
        for item in raw_presets:
            normalized = self._normalize_bp_preset(item)
            if normalized is None:
                continue
            key = self._bp_preset_key(normalized)
            if key in seen:
                continue
            presets.append(normalized)
            seen.add(key)
        if not presets:
            return [self._default_bp_preset()]
        return presets

    def _save_bp_presets(self):
        try:
            with open(self.bp_preset_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {'presets': self.bp_presets},
                    f,
                    ensure_ascii=True,
                    indent=2,
                )
        except OSError as exc:
            self.show_status_message(f'Failed to save BP presets: {exc}', timeout_ms=5000)

    def _load_phase_presets(self):
        if not exists(self.phase_preset_path):
            default_preset = self._default_phase_preset()
            return [default_preset] if default_preset else []
        try:
            with open(self.phase_preset_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        except (OSError, json.JSONDecodeError):
            default_preset = self._default_phase_preset()
            return [default_preset] if default_preset else []
        if not isinstance(loaded, list):
            return []
        presets = []
        seen = set()
        for item in loaded:
            normalized = self._normalize_phase_preset(item)
            if normalized is None or normalized in seen:
                continue
            presets.append(normalized)
            seen.add(normalized)
        if not presets:
            default_preset = self._default_phase_preset()
            return [default_preset] if default_preset else []
        return presets

    def _save_phase_presets(self):
        try:
            with open(self.phase_preset_path, 'w', encoding='utf-8') as f:
                json.dump(self.phase_presets, f, ensure_ascii=True, indent=2)
        except OSError as exc:
            self.show_status_message(f'Failed to save phase presets: {exc}', timeout_ms=5000)

    def _refresh_bp_preset_combo(self, select_key=None):
        if not hasattr(self, 'bp_preset_combo'):
            return
        combo = self.bp_preset_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self.BP_PRESET_PLACEHOLDER, None)
        selected_index = 0
        for index, profile in enumerate(self.bp_presets, start=1):
            combo.addItem(self._bp_preset_label(profile), profile)
            if select_key is not None and self._bp_preset_key(profile) == select_key:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def _refresh_phase_preset_combo(self, select_text=None):
        if not hasattr(self, 'phase_preset_combo'):
            return
        combo = self.phase_preset_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem(self.PHASE_PRESET_PLACEHOLDER, None)
        selected_index = 0
        for index, preset_text in enumerate(self.phase_presets, start=1):
            combo.addItem(self._phase_preset_label(preset_text), preset_text)
            if select_text is not None and preset_text == select_text:
                selected_index = index
        combo.setCurrentIndex(selected_index)
        combo.blockSignals(False)

    def apply_bp_preset_selection(self, index):
        if index <= 0:
            return
        profile = self.bp_preset_combo.itemData(index)
        normalized = self._normalize_bp_preset(profile)
        if normalized is None:
            return
        self.bp_freqmin_input.setText(f"{normalized['freqmin']:g}")
        self.bp_freqmax_input.setText(f"{normalized['freqmax']:g}")
        self.bp_corners_input.setText(str(normalized['corners']))
        self.bp_passes_input.setText(str(normalized['passes']))
        self.show_status_message(
            f"Loaded BP preset only: {self._bp_preset_label(normalized)}",
            timeout_ms=4000,
        )

    def apply_phase_preset_selection(self, index):
        if index <= 0:
            return
        preset_text = self.phase_preset_combo.itemData(index)
        normalized = self._normalize_phase_preset(preset_text)
        if normalized is None:
            return
        _canonical_tokens, error_message = self.mpl.wavefig.set_standard_phase_tokens(
            normalized,
            refresh=True,
        )
        if error_message is not None:
            self.show_status_message(error_message, timeout_ms=5000)
            return
        self.show_status_message(f"Loaded phase preset: {normalized}")

    def on_theory_model_changed(self, model):
        self.theory_model = model

    def apply_theory_markers(self):
        """Apply theory phase markers using taup setsac command"""
        phases_text = self.theory_phase_input.text().strip()
        if not phases_text:
            self.show_status_message('Please enter phase definitions', timeout_ms=5000)
            return

        model = self.theory_model
        wavepath = self.mpl.wavefig.wavepath

        # Get SAC files
        sac_files = sorted([f for f in os.listdir(wavepath) if f.lower().endswith(self.suffix.lower())])

        if not sac_files:
            self.show_status_message('No SAC files found', timeout_ms=5000)
            return

        # Run taup setsac command on ALL SAC files at once (much faster than one by one)
        # Format: P-0 means write P time to t0 header, pP-2 means write pP time to t2 header, etc.
        cmd = [
            self.taup_bin,
            'setsac',
            '--evdpkm',
            '-p',
            phases_text,
            '--mod',
            model,
        ] + sac_files  # Pass all files at once!

        try:
            completed = subprocess.run(
                cmd,
                check=False,  # Don't raise exception on errors, check return code instead
                capture_output=True,
                text=True,
                cwd=wavepath,
            )

            # Check for warnings/errors in output
            stderr = completed.stderr or ''
            stdout = completed.stdout or ''

            # Count skipped files (those without O marker)
            skipped = stderr.count('O marker not set')
            total = len(sac_files)
            success_count = total - skipped

            # Clean user headers (user0, user2, user3) that taup setsac writes
            # This is done immediately and saved to disk to avoid user2 interference
            self.clean_thread = CleanUserHeadersThread(wavepath, self.suffix)
            self.clean_thread.finished.connect(self._on_clean_user_headers_finished)
            self.clean_thread.start()

            if skipped > 0:
                status = f"Applied theory markers: {success_count}/{total} processed ({skipped} skipped - no O marker), cleaning user headers..."
            else:
                status = f"Applied theory markers: {success_count} files processed, cleaning user headers..."

        except Exception as e:
            status = f"Error running taup setsac: {str(e)}"

        self.show_status_message(status, timeout_ms=8000)

    def _on_clean_user_headers_finished(self, success_count, error_count):
        # Reload waveforms - markers are now in SAC t headers and will be displayed automatically
        self.show_raw_waveforms()

        status = f"Cleaned user headers: {success_count} succeeded"
        if error_count > 0:
            status += f", {error_count} failed"
        self.show_status_message(status, timeout_ms=5000)

    def clear_theory_markers(self):
        """Clear theory phase markers from SAC files (unset t headers)"""
        wavepath = self.mpl.wavefig.wavepath
        sac_files = sorted([f for f in os.listdir(wavepath) if f.lower().endswith(self.suffix.lower())])

        if not sac_files:
            self.show_status_message('No SAC files found', timeout_ms=5000)
            return

        # Show "Clearing..." message
        self.show_status_message('Clearing theory markers...', timeout_ms=0)

        # Run in background thread to avoid freezing UI
        self.clear_thread = ClearTheoryMarkersThread(
            wavepath,
            self.suffix,
            self.mpl.wavefig.markers
        )
        self.clear_thread.finished.connect(self._on_clear_theory_finished)
        self.clear_thread.start()

    def _on_clear_theory_finished(self, success_count, error_count):
        # Reload waveforms
        self.show_raw_waveforms()

        status = f"Cleared theory markers: {success_count} succeeded"
        if error_count > 0:
            status += f", {error_count} failed"
        self.show_status_message(status, timeout_ms=5000)

    def add_current_bp_preset(self):
        profile = self._current_bp_preset()
        if profile is None:
            self.show_status_message('Invalid BP parameters; preset not added', timeout_ms=5000)
            return
        target_key = self._bp_preset_key(profile)
        for existing in self.bp_presets:
            if self._bp_preset_key(existing) == target_key:
                self._refresh_bp_preset_combo(select_key=target_key)
                self.show_status_message(f"BP preset already exists: {self._bp_preset_label(profile)}")
                return
        self.bp_presets.append(profile)
        self._save_bp_presets()
        self._refresh_bp_preset_combo(select_key=target_key)
        self.show_status_message(f"Added {self._bp_preset_label(profile)}")

    def add_current_phase_preset(self):
        current_tokens = self._normalize_phase_preset(self.mpl.wavefig.standard_export_phase_tokens)
        if current_tokens is None:
            self.show_status_message('Current phase combination is empty or invalid', timeout_ms=5000)
            return
        if current_tokens in self.phase_presets:
            self._refresh_phase_preset_combo(select_text=current_tokens)
            self.show_status_message(f"Phase preset already exists: {current_tokens}")
            return
        self.phase_presets.append(current_tokens)
        self._save_phase_presets()
        self._refresh_phase_preset_combo(select_text=current_tokens)
        self.show_status_message(f"Added phase preset: {current_tokens}")

    def remove_selected_bp_preset(self):
        if not hasattr(self, 'bp_preset_combo'):
            return
        selected_index = self.bp_preset_combo.currentIndex()
        if selected_index <= 0:
            self.show_status_message('Select a BP preset to remove', timeout_ms=4000)
            return
        profile = self.bp_preset_combo.itemData(selected_index)
        normalized = self._normalize_bp_preset(profile)
        if normalized is None:
            self.show_status_message('Selected BP preset is invalid', timeout_ms=4000)
            return
        self.bp_presets = [
            existing for existing in self.bp_presets
            if self._bp_preset_key(existing) != self._bp_preset_key(normalized)
        ]
        self._save_bp_presets()
        self._refresh_bp_preset_combo()
        self.show_status_message(f"Removed {self._bp_preset_label(normalized)}")

    def remove_selected_phase_preset(self):
        if not hasattr(self, 'phase_preset_combo'):
            return
        selected_index = self.phase_preset_combo.currentIndex()
        if selected_index <= 0:
            self.show_status_message('Select a phase preset to remove', timeout_ms=4000)
            return
        preset_text = self.phase_preset_combo.itemData(selected_index)
        normalized = self._normalize_phase_preset(preset_text)
        if normalized is None:
            self.show_status_message('Selected phase preset is invalid', timeout_ms=4000)
            return
        self.phase_presets = [
            existing for existing in self.phase_presets
            if existing != normalized
        ]
        self._save_phase_presets()
        self._refresh_phase_preset_combo()
        self.show_status_message(f"Removed phase preset: {normalized}")

    def sync_phase_preset_from_wavefig(self, phase_tokens):
        normalized = self._normalize_phase_preset(phase_tokens)
        self._refresh_phase_preset_combo(select_text=normalized)

    def _default_xlim_for_marker(self, marker):
        if marker in ('t0', 't7'):
            return [-10, 70]
        if marker in ('t2', 't6'):
            return [-40, 30]
        if marker in ('t3', 't5'):
            return [-50, 20]
        return [self.x1, self.x2]

    def change_alignment_selection(self, marker):
        self.pending_tmarker = marker
        x1, x2 = self._default_xlim_for_marker(marker)
        self.x1_input.setText(str(x1))
        self.x2_input.setText(str(x2))
        self.x1 = x1
        self.x2 = x2
        self.C_time_window()

    def change_mode_selection(self, mode):
        self.pending_axis_mode = mode
        self.C_time_window()

    def finish(self):
        self.mpl.wavefig.finish()
        self.mpl.wavefig.close_preview_window()
        self.close()

    def set_pick_mode(self, key):
        pick_key = str(key).lower()
        self.mpl.wavefig.key = pick_key
        self.mpl.wavefig.pick_mode_armed = pick_key in self.mpl.wavefig.markers or pick_key in ('d', 's')

    def flip_current_wave_polarity(self):
        wave_name = self.mpl.wavefig.current_pick_wave_name
        if not wave_name:
            self.show_status_message('No current waveform to flip', timeout_ms=4000)
            return
        changed = self.mpl.wavefig._toggle_user4_marker(wave_name)
        if not changed:
            state_label = 'flipped' if self.mpl.wavefig._is_user4_wave(wave_name) else 'normal'
            self.show_status_message(f'{wave_name} already {state_label}', timeout_ms=4000)
            return
        self.mpl.wavefig.refresh_current_page()
        if self.mpl.wavefig.plotfig is not None and plt.fignum_exists(self.mpl.wavefig.plotfig.number):
            preview_index = getattr(self.mpl.wavefig.plotfig, '_preview_controls', {}).get('preview_index', 0)
            self.mpl.wavefig._refresh_preview_figure(self.mpl.wavefig.plotfig, preview_index)
        self.mpl.draw()
        state_label = 'flipped' if self.mpl.wavefig._is_user4_wave(wave_name) else 'restored'
        self.show_status_message(f'Polarity {state_label} for {wave_name}', timeout_ms=4000)

    def plot_ui(self):
        self.open_preview(0)

    def open_preview(self, preview_index):
        if self.mpl.wavefig.plotfig is not None:
            self.mpl.wavefig._close_preview_control_dock()
            plt.close(self.mpl.wavefig.plotfig)
        opened = self.mpl.wavefig.plot_preview(preview_index)
        if opened is False:
            phase_label = ''
            if 0 <= preview_index < len(self.preview_phases):
                phase_label = self.preview_phases[preview_index]
            if phase_label:
                self.show_status_message(f'Preview not opened: no visible waveforms for {phase_label}', timeout_ms=5000)
            else:
                self.show_status_message('Preview not opened', timeout_ms=5000)

    def plot_save(self):
        default_name = self.mpl.wavefig.evtname
        fileName_choose, filetype = QFileDialog.getSaveFileName(self,
                                    "Save the figure",
                                    os.path.join(os.getcwd(), default_name),
                                    "PDF Files (*.pdf);;Images (*.png);;All Files (*)")

        if fileName_choose == "":
            return
        if not hasattr(self.mpl.wavefig, 'plotfig'):
            self.mpl.wavefig.plot()
        try:
            self.mpl.wavefig.plotfig.savefig(fileName_choose, dpi=500, bbox_inches='tight')
            # self.mpl.wavefig.logger.info('Figure saved to {}'.format(fileName_choose))
        except Exception as e:
            print(f"An error occurred: {e}")
        #     self.mpl.wavefig.logger.error('{}'.format(e))

    def _set_geom_center(self, height=1, width=1):
        # Maximize the main pick window (original behavior). On WSLg/XWayland
        # the WindowMaximized hint alone can be ignored (window stays at default
        # size in the top-left); fall back to an explicit near-full-size centered
        # geometry when showMaximized doesn't actually maximize.
        self.showMaximized()
        try:
            geo = self.geometry()
            workarea_geo = None
            wa = screen_workarea_rect(widget=self)
            if wa is not None and wa.isValid():
                workarea_geo = (wa.x(), wa.y(), wa.width(), wa.height())
        except Exception:
            workarea_geo = None
        # If the window didn't actually expand to (near) the workarea, the WM
        # ignored showMaximized — force a centered near-full-size geometry.
        if workarea_geo is not None:
            try:
                wx, wy, ww, wh = workarea_geo
                if geo.width() < int(ww * 0.9) or geo.height() < int(wh * 0.9):
                    center_widget_on_workarea(self, frac=0.98)
                    self.show()
            except Exception:
                pass

    def _define_global_shortcuts(self):
        self.key_n = QShortcut(QKeySequence('n'), self)
        self.key_n.activated.connect(self.next_connect)
        self.key_b = QShortcut(QKeySequence('b'), self)
        self.key_b.activated.connect(self.previous_connect)
        self.key_g = QShortcut(QKeySequence('g'), self)
        self.key_g.activated.connect(lambda: self.jump_to_missing_alignment_wave(1))
        self.key_left = QShortcut(QKeySequence('Left'), self)
        self.key_left.activated.connect(lambda: self.jump_to_marked_wave(-1))
        self.key_right = QShortcut(QKeySequence('Right'), self)
        self.key_right.activated.connect(lambda: self.jump_to_marked_wave(1))
        self.key_space = QShortcut(QKeySequence('Space'), self)
        self.key_space.activated.connect(self.close_topmost_window)
        self.key_search = QShortcut(QKeySequence('Ctrl+F'), self)
        self.key_search.activated.connect(self.focus_station_search)
        self.key_search_slash = QShortcut(QKeySequence('/'), self)
        self.key_search_slash.activated.connect(self.focus_station_search)
        self.pick_shortcuts = []
        for key in self.PICK_SHORTCUTS:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda pick_key=key: self.set_pick_mode(pick_key))
            self.pick_shortcuts.append(shortcut)
        self.preview_shortcuts = []
        for preview_index, phase in enumerate(self.preview_phases):
            key = self.PREVIEW_SHORTCUTS.get(phase)
            if key is None:
                continue
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda idx=preview_index: self.open_preview(idx))
            self.preview_shortcuts.append(shortcut)

    def add_btn(self):
        pre_btn = QPushButton("Back (b)")
        pre_btn.clicked.connect(self.previous_connect)
        next_btn = QPushButton("Next (n)")
        next_btn.clicked.connect(self.next_connect)
        jump_missing_btn = QPushButton("Jump Missing (g)")
        jump_missing_btn.clicked.connect(lambda: self.jump_to_missing_alignment_wave(1))
        jump_u1_btn = QPushButton("Jump U1")
        jump_u1_btn.clicked.connect(lambda: self.set_jump_target_mode('user1'))
        jump_u2_btn = QPushButton("Jump U2")
        jump_u2_btn.clicked.connect(lambda: self.set_jump_target_mode('user2'))
        jump_u5_btn = QPushButton("Jump U5")
        jump_u5_btn.clicked.connect(lambda: self.set_jump_target_mode('user5'))
        jump_flip_btn = QPushButton("Jump Flip")
        jump_flip_btn.clicked.connect(lambda: self.set_jump_target_mode('user4'))
        flip_current_btn = QPushButton("Flip Current")
        flip_current_btn.clicked.connect(self.flip_current_wave_polarity)
        finish_btn = QPushButton("Finish")
        finish_btn.clicked.connect(self.finish)

        btnbox_top = QHBoxLayout()
        btnbox_top.setSpacing(8)
        btnbox_top.addStretch(1)
        btnbox_top.addWidget(pre_btn)
        btnbox_top.addWidget(next_btn)
        btnbox_top.addWidget(jump_missing_btn)
        btnbox_top.addWidget(jump_u1_btn)
        btnbox_top.addWidget(jump_u2_btn)
        btnbox_top.addWidget(jump_u5_btn)
        btnbox_top.addWidget(jump_flip_btn)
        btnbox_top.addWidget(flip_current_btn)
        btnbox_top.addWidget(finish_btn)

        btnbox_bottom = QHBoxLayout()
        btnbox_bottom.setSpacing(8)
        btnbox_bottom.addStretch(1)
        for preview_index, phase in enumerate(self.preview_phases):
            key = self.PREVIEW_SHORTCUTS.get(phase, '?').upper()
            plot_btn = QPushButton(f"Preview {phase} ({key})")
            plot_btn.clicked.connect(lambda _checked=False, idx=preview_index: self.open_preview(idx))
            btnbox_bottom.addWidget(plot_btn)


        x1_input = QLineEdit(self)
        self.x1_input = x1_input
        x1_input.setValidator(QIntValidator())
        x1_input.setText(str(self.x1))
        x1_input.textChanged.connect(self.getx1)
        x1_input.setFixedWidth(72)
        x1_input.clearFocus()

        x2_input = QLineEdit(self)
        self.x2_input = x2_input
        x2_input.setValidator(QIntValidator())
        x2_input.setText(str(self.x2))
        x2_input.textChanged.connect(self.getx2)
        x2_input.setFixedWidth(72)
        x2_input.clearFocus()

        station_search_input = QLineEdit(self)
        self.station_search_input = station_search_input
        station_search_input.setPlaceholderText('Ctrl+F or /: search station, Enter: jump')
        station_search_input.setMinimumWidth(180)
        station_search_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        station_search_input.returnPressed.connect(self.jump_to_station_search)
        station_search_input.hide()

        backup_restore_input = QLineEdit(self)
        self.backup_restore_input = backup_restore_input
        backup_restore_input.setPlaceholderText('Backup path, Enter: restore')
        backup_restore_input.setMinimumWidth(140)
        backup_restore_input.setMaximumWidth(260)
        backup_restore_input.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        backup_restore_input.returnPressed.connect(self.restore_from_backup_path)

        # Same-directory event switcher: prev/next button + dropdown of sibling
        # event directories, so you can move between events without typing an
        # absolute path each time (dpk <abs_path> still works too).
        prev_event_btn = QPushButton("◀")
        prev_event_btn.setFixedWidth(28)
        prev_event_btn.setToolTip('Previous event in this directory')
        prev_event_btn.clicked.connect(lambda: self._switch_event_by_offset(-1))
        self.prev_event_btn = prev_event_btn

        event_combo = QComboBox(self)
        self.event_combo = event_combo
        event_combo.setMinimumWidth(150)
        event_combo.setMaximumWidth(260)
        event_combo.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        event_combo.setToolTip('Switch to another event in the same directory')
        self._event_combo_updating = False
        event_combo.activated.connect(lambda index: self._switch_to_event_combo_index(index))

        next_event_btn = QPushButton("▶")
        next_event_btn.setFixedWidth(28)
        next_event_btn.setToolTip('Next event in this directory')
        next_event_btn.clicked.connect(lambda: self._switch_event_by_offset(1))
        self.next_event_btn = next_event_btn

        bp_freqmin_input = QLineEdit(self)
        self.bp_freqmin_input = bp_freqmin_input
        bp_freqmin_input.setValidator(QDoubleValidator(0.0, 100.0, 6, self))
        bp_freqmin_input.setText(str(self.bp_freqmin))
        bp_freqmin_input.textChanged.connect(self.get_bp_freqmin)
        bp_freqmin_input.setFixedWidth(64)
        bp_freqmin_input.clearFocus()

        bp_freqmax_input = QLineEdit(self)
        self.bp_freqmax_input = bp_freqmax_input
        bp_freqmax_input.setValidator(QDoubleValidator(0.0, 100.0, 6, self))
        bp_freqmax_input.setText(str(self.bp_freqmax))
        bp_freqmax_input.textChanged.connect(self.get_bp_freqmax)
        bp_freqmax_input.setFixedWidth(64)
        bp_freqmax_input.clearFocus()

        bp_corners_input = QLineEdit(self)
        self.bp_corners_input = bp_corners_input
        bp_corners_input.setValidator(QIntValidator(1, 20, self))
        bp_corners_input.setText(str(self.bp_corners))
        bp_corners_input.textChanged.connect(self.get_bp_corners)
        bp_corners_input.setFixedWidth(40)
        bp_corners_input.clearFocus()

        bp_passes_input = QLineEdit(self)
        self.bp_passes_input = bp_passes_input
        bp_passes_input.setValidator(QIntValidator(1, 20, self))
        bp_passes_input.setText(str(self.bp_passes))
        bp_passes_input.textChanged.connect(self.get_bp_passes)
        bp_passes_input.setFixedWidth(40)
        bp_passes_input.clearFocus()

        align_label = QLabel("Align")
        align_combo = QComboBox(self)
        self.align_combo = align_combo
        align_combo.addItems(self.ALIGN_OPTIONS)
        align_combo.setCurrentText(self.tmaker or 't0')
        align_combo.setFixedWidth(88)
        align_combo.currentTextChanged.connect(self.change_alignment_selection)

        mode_label = QLabel("Mode")
        mode_combo = QComboBox(self)
        self.mode_combo = mode_combo
        for mode_value, mode_label_text in self.MODE_OPTIONS:
            mode_combo.addItem(mode_label_text, mode_value)
        mode_index = mode_combo.findData(self.axis_mode)
        mode_combo.setCurrentIndex(mode_index if mode_index >= 0 else 1)
        mode_combo.setFixedWidth(116)
        mode_combo.currentIndexChanged.connect(lambda index: self.change_mode_selection(self.mode_combo.itemData(index)))

        raw_btn = QPushButton("Reload")
        raw_btn.clicked.connect(self.show_raw_waveforms)
        raw_btn.setFixedWidth(76)

        apply_bp_btn = QPushButton("Apply BP")
        self.apply_bp_btn = apply_bp_btn
        apply_bp_btn.clicked.connect(self.apply_bandpass_settings)
        apply_bp_btn.setFixedWidth(88)

        pathbox_primary = QHBoxLayout()
        pathbox_primary.setSpacing(8)

        alignbox = QHBoxLayout()
        alignbox.setSpacing(4)
        alignbox.addWidget(align_label)
        alignbox.addWidget(align_combo)

        modebox = QHBoxLayout()
        modebox.setSpacing(4)
        modebox.addWidget(mode_label)
        modebox.addWidget(mode_combo)

        bp_label = QLabel("BP")
        bp_preset_combo = QComboBox(self)
        self.bp_preset_combo = bp_preset_combo
        bp_preset_combo.setMinimumWidth(130)
        bp_preset_combo.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        bp_preset_combo.currentIndexChanged.connect(self.apply_bp_preset_selection)

        bp_add_btn = QPushButton("+")
        self.bp_add_btn = bp_add_btn
        bp_add_btn.setFixedWidth(30)
        bp_add_btn.clicked.connect(self.add_current_bp_preset)

        bp_remove_btn = QPushButton("-")
        self.bp_remove_btn = bp_remove_btn
        bp_remove_btn.setFixedWidth(30)
        bp_remove_btn.clicked.connect(self.remove_selected_bp_preset)

        phase_label = QLabel("Ph")
        phase_preset_combo = QComboBox(self)
        self.phase_preset_combo = phase_preset_combo
        phase_preset_combo.setMinimumWidth(130)
        phase_preset_combo.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        phase_preset_combo.currentIndexChanged.connect(self.apply_phase_preset_selection)

        phase_add_btn = QPushButton("+")
        self.phase_add_btn = phase_add_btn
        phase_add_btn.setFixedWidth(30)
        phase_add_btn.clicked.connect(self.add_current_phase_preset)

        phase_remove_btn = QPushButton("-")
        self.phase_remove_btn = phase_remove_btn
        phase_remove_btn.setFixedWidth(30)
        phase_remove_btn.clicked.connect(self.remove_selected_phase_preset)

        bpbox = QHBoxLayout()
        bpbox.setSpacing(4)
        bpbox.addWidget(bp_label)
        bpbox.addWidget(bp_preset_combo)
        bpbox.addWidget(bp_add_btn)
        bpbox.addWidget(bp_remove_btn)
        bpbox.addWidget(QLabel("c"))
        bpbox.addWidget(bp_freqmin_input)
        bpbox.addWidget(bp_freqmax_input)
        bpbox.addWidget(QLabel("n"))
        bpbox.addWidget(bp_corners_input)
        bpbox.addWidget(QLabel("p"))
        bpbox.addWidget(bp_passes_input)
        bpbox.addWidget(apply_bp_btn)
        bpbox.addWidget(raw_btn)

        phbox = QHBoxLayout()
        phbox.setSpacing(4)
        phbox.addWidget(phase_label)
        phbox.addWidget(phase_preset_combo)
        phbox.addWidget(phase_add_btn)
        phbox.addWidget(phase_remove_btn)

        pathbox_primary.addLayout(alignbox)
        pathbox_primary.addWidget(x1_input)
        pathbox_primary.addWidget(x2_input)
        pathbox_primary.addWidget(station_search_input, 1)
        pathbox_primary.addWidget(backup_restore_input, 1)
        pathbox_primary.addWidget(prev_event_btn)
        pathbox_primary.addWidget(event_combo, 1)
        pathbox_primary.addWidget(next_event_btn)
        pathbox_primary.addLayout(modebox)
        pathbox_primary.addStretch(1)

        pathbox_secondary = QHBoxLayout()
        pathbox_secondary.setSpacing(12)
        pathbox_secondary.addLayout(bpbox)
        pathbox_secondary.addLayout(phbox)
        pathbox_secondary.addStretch(1)

        ctrl_layout_top = QHBoxLayout()
        ctrl_layout_top.setSpacing(12)
        ctrl_layout_top.addLayout(pathbox_primary)

        ctrl_layout_filters = QHBoxLayout()
        ctrl_layout_filters.setSpacing(12)
        ctrl_layout_filters.addLayout(pathbox_secondary)

        ctrl_layout_actions = QHBoxLayout()
        ctrl_layout_actions.setSpacing(16)
        ctrl_layout_actions.addLayout(btnbox_top)

        ctrl_layout_bottom = QHBoxLayout()
        ctrl_layout_bottom.setSpacing(16)
        ctrl_layout_bottom.addStretch(1)
        ctrl_layout_bottom.addLayout(btnbox_bottom)

        # Theory phase marker controls (new row)
        theory_label = QLabel("Theory Phases")
        theory_label.setStyleSheet("font-weight: bold;")

        theory_model_label = QLabel("Model:")
        theory_model_combo = QComboBox(self)
        self.theory_model_combo = theory_model_combo
        theory_model_combo.addItems(['iasp91', 'prem', 'ak135', 'iasp91'])
        theory_model_combo.setCurrentText(self.theory_model)
        theory_model_combo.setFixedWidth(90)
        theory_model_combo.currentTextChanged.connect(self.on_theory_model_changed)

        theory_phase_label = QLabel("Phases:")
        theory_phase_input = QLineEdit(self)
        self.theory_phase_input = theory_phase_input
        theory_phase_input.setText(self.theory_phases)
        theory_phase_input.setPlaceholderText('e.g., pP-2,sP-3,P-0')
        theory_phase_input.setFixedWidth(150)

        theory_apply_btn = QPushButton("Apply Theory Markers")
        theory_apply_btn.clicked.connect(self.apply_theory_markers)

        theory_clear_btn = QPushButton("Clear Theory")
        theory_clear_btn.clicked.connect(self.clear_theory_markers)

        theorybox = QHBoxLayout()
        theorybox.setSpacing(8)
        theorybox.addWidget(theory_label)
        theorybox.addWidget(theory_model_label)
        theorybox.addWidget(theory_model_combo)
        theorybox.addWidget(theory_phase_label)
        theorybox.addWidget(theory_phase_input)
        theorybox.addWidget(theory_apply_btn)
        theorybox.addWidget(theory_clear_btn)
        theorybox.addStretch(1)

        ctrl_layout_theory = QHBoxLayout()
        ctrl_layout_theory.setSpacing(16)
        ctrl_layout_theory.addLayout(theorybox)

        self.layout.addLayout(ctrl_layout_top)
        self.layout.addLayout(ctrl_layout_filters)
        self.layout.addLayout(ctrl_layout_actions)
        self.layout.addLayout(ctrl_layout_bottom)
        self.layout.addLayout(ctrl_layout_theory)
        self._refresh_bp_preset_combo()
        self._refresh_phase_preset_combo()
        x1_input.returnPressed.connect(self.C_time_window)
        x2_input.returnPressed.connect(self.C_time_window)

def main():
    parser = argparse.ArgumentParser(description="User interface for picking waveforms")

    parser.add_argument('wave_path', type=str, help='Path to waveforms')
    parser.add_argument('-a', help="Arrangement of waveforms, defaults to 'gcarc'", dest='order',
                        default='gcarc', type=str, metavar='baz|gcarc|az')
    parser.add_argument('-x', help="Set x limits of the current axes; if omitted, use defaults for the selected alignment marker",
                        dest='xlim', default=None, nargs=2, type=float, metavar=('xmin', 'xmax'))
    parser.add_argument('-t', help="Set tmarker for alignment, defaults t0", dest='tmarker', type=str, default='t0')
    parser.add_argument('-s', help="Set sacfile suffix, defaults .sac", dest='suffix', type=str, default='.sac')
    parser.add_argument('-p', help="Preview align phases, defaults t7,t6,t5,t0,t2,t3",
                        dest='ta_tb', type=str, default='t7,t6,t5,t0,t2,t3')
    parser.add_argument('-x2', help="Set x limits for previews; if omitted, use the same default window rules as the main alignment view",
                        dest='xlim_preview', default=None, nargs='+', type=float)
    arg = parser.parse_args()

    wavepath = arg.wave_path
    if not exists(wavepath):
        raise FileNotFoundError('No such directory: {}'.format(wavepath))
    runtime_event_dir = str(source_event_dir_for_runtime(wavepath))


    if arg.xlim:
        xlim = arg.xlim
    elif is_stack_event_dir(wavepath):
        xlim = stack_sac_time_window(wavepath, suffix=arg.suffix) or [-10, 10]
    elif arg.tmarker in ('t0', 't7'):
        xlim = [-10, 70]
    elif arg.tmarker in ('t2', 't6'):
        xlim = [-40, 30]
    elif arg.tmarker in ('t3', 't5'):
        xlim = [-50, 20]
    else:
        xlim = [-10, 10]


    preview_phases = [item.strip() for item in arg.ta_tb.split(',') if item.strip()]
    if len(preview_phases) == 0:
        raise ValueError('At least one preview phase must be provided')
    if arg.xlim_preview is not None and len(arg.xlim_preview) != len(preview_phases) * 2:
        raise ValueError('Preview window count must match preview phase count')

    if not is_stack_event_dir(wavepath):
        ensure_event_pierce_files(
            event_dir=runtime_event_dir,
            phases=('pP', 'sP'),
            models=('prem',),
            pierce_depth_km=24.4,
        )
        ensure_event_pierce_files(
            event_dir=runtime_event_dir,
            phases=('pP', 'sP'),
            models=('iasp91',),
            pierce_depth_km=35.0,
        )
        ensure_event_theory_summary(
            event_dir=runtime_event_dir,
            model='iasp91',
            suffix=arg.suffix,
        )

    app = QApplication(sys.argv)
    ui = MatplotlibWidget(wavepath, xlim=xlim, order=arg.order, tmarker=arg.tmarker, suffix=arg.suffix,ta_tb=arg.ta_tb,xlim_preview=arg.xlim_preview)

    sys.exit(app.exec())


if __name__ == '__main__':
    main()



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     ui = MatplotlibWidget()
#     ui.show()
#     sys.exit(app.exec_())
