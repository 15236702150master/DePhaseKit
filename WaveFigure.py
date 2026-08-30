import glob
import hashlib
import json
import os
import re
import sys
import obspy
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from matplotlib import colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib import transforms
from matplotlib.figure import Figure
from os.path import join, basename
import shutil
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, TextBox, RectangleSelector, EllipseSelector
import math
import matplotlib.ticker as ticker
from collections import Counter
from scipy.signal import hilbert
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QComboBox,
    QVBoxLayout,
    QWidget,
)
from pierce_point_cache import (
    DEFAULT_OUTPUT_ROOT as DEFAULT_PIERCE_OUTPUT_ROOT,
    DEFAULT_PIERCE_DEPTH_KM,
    DEFAULT_TAUP_BIN,
    PiercePointRecord,
    PROJECT_ROOT,
    ensure_pierce_file,
    ensure_event_pierce_files,
    load_pierce_points,
    pierce_file_path,
    relative_event_path,
)
from forward.constants import DEFAULT_CRUST_VP, DEFAULT_CRUST_VS
from stack_crustal_thickness import (
    DEFAULT_TAUP_BIN as DEFAULT_STACK_TAUP_BIN,
    calculate_pp_pmp_thickness,
    calculate_sp_smp_thickness,
    fetch_taup_ray_parameter,
    fetch_obspy_ray_parameter,
    calculate_single_trace_thickness,
    reverse_station_coord,
)
from stack_system import (
    delete_stack_config,
    delete_stack_group_configs,
    inspect_stack_event_health,
    iter_stack_sac_paths,
    is_stack_event_dir,
    load_stack_event_marker,
    load_stack_sidecar_map,
    repair_stack_event_metadata,
    source_event_dir_for_runtime,
    stack_event_dir_for_source,
    stack_output_dir_for_runtime,
    stack_output_dir_for_source,
    stack_wave_name_from_path,
    stack_wave_summary_from_sidecar,
    _natural_sort_key,
    update_stack_sidecar_markers,
    write_stack_sidecar_payload,
    write_stack_workspace_index,
    write_stack_event_marker,
)
from window_geometry import screen_workarea_rect, center_widget_on_workarea, center_widget_keep_size

PREVIEW_AMPLITUDE_PRESET_PATH = os.path.join(os.path.dirname(__file__), 'preview_amplitude_presets.json')
SAC_KM_PER_DEG = 111.19492
STACK_TRACE_COLOR = '#8b1a1a'
STACK_TRACE_LINEWIDTH = 1.5
MEMBER_TRACE_COLOR = 'black'
MEMBER_TRACE_LINEWIDTH = 0.2
EVENT_DIR_NAME_RE = re.compile(r'^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$')


def _remove_matplotlib_keymap_bindings(keymap_name, blocked_keys):
    configured_keys = list(plt.rcParams.get(keymap_name, []))
    plt.rcParams[keymap_name] = [
        key for key in configured_keys
        if key not in set(blocked_keys)
    ]


_remove_matplotlib_keymap_bindings('keymap.yscale', {'l'})
_remove_matplotlib_keymap_bindings('keymap.xscale', {'L'})


def _force_qt_arrow_cursor_for_figure(fig):
    canvas = getattr(fig, 'canvas', fig)
    if canvas is None:
        return
    try:
        canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
    except Exception:
        return
    if getattr(canvas, '_dephasekit_arrow_cursor_forced', False):
        return

    def _keep_arrow_cursor(_cursor=None):
        try:
            canvas.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        except Exception:
            pass

    canvas.set_cursor = _keep_arrow_cursor
    canvas._dephasekit_arrow_cursor_forced = True


def _sac_attr(trace, attr, default=math.nan):
    sac = getattr(trace.stats, 'sac', None)
    if sac is None:
        return default
    try:
        value = getattr(sac, attr)
    except AttributeError:
        return default
    if value is None:
        return default
    return value


def _finite_sac_number(value):
    numeric = _safe_float(value)
    if math.isnan(numeric) or not math.isfinite(numeric):
        return math.nan
    if numeric in (-12345.0, 12345.0):
        return math.nan
    return numeric


def _sac_gcarc(trace, default=math.nan):
    # Some SAC tools compute/display gcarc from lcalda/dist/coordinates even
    # when ObsPy does not expose a stored stats.sac.gcarc field.
    gcarc = _finite_sac_number(_sac_attr(trace, 'gcarc', math.nan))
    if not math.isnan(gcarc):
        return gcarc
    dist_km = _finite_sac_number(_sac_attr(trace, 'dist', math.nan))
    if not math.isnan(dist_km):
        return dist_km / SAC_KM_PER_DEG
    evla = _finite_sac_number(_sac_attr(trace, 'evla', math.nan))
    evlo = _finite_sac_number(_sac_attr(trace, 'evlo', math.nan))
    stla = _finite_sac_number(_sac_attr(trace, 'stla', math.nan))
    stlo = _finite_sac_number(_sac_attr(trace, 'stlo', math.nan))
    if not any(math.isnan(value) for value in (evla, evlo, stla, stlo)):
        try:
            from obspy.geodetics import locations2degrees
            return float(locations2degrees(evla, evlo, stla, stlo))
        except Exception:
            return default
    return default


def _validate_sac_file(path):
    try:
        traces = obspy.read(str(path))
    except Exception as exc:
        return False, str(exc)
    if len(traces) == 0:
        return False, "Empty SAC stream"
    return True, ""


def _event_name_from_dsm_path(path):
    if not path:
        return ''
    try:
        resolved_path = Path(path).expanduser().resolve()
        relative_parts = resolved_path.relative_to((PROJECT_ROOT / 'data' / 'dsm').resolve()).parts
    except (OSError, ValueError):
        return ''
    for part in relative_parts:
        if EVENT_DIR_NAME_RE.match(part):
            return part
    return ''


def _sac_float(trace, attr, default=math.nan):
    if str(attr).lower() == 'gcarc':
        return _sac_gcarc(trace, default=default)
    return _safe_float(_sac_attr(trace, attr, default))


def _attrib_dict_from_trace(trace):
    sac = getattr(trace.stats, 'sac', None)
    if sac is None:
        trace.stats.sac = obspy.core.AttribDict()
    elif not isinstance(sac, obspy.core.AttribDict):
        trace.stats.sac = obspy.core.AttribDict(dict(sac))
    return trace.stats.sac


class QtLineEditAdapter:
    def __init__(self, widget):
        self.widget = widget

    @property
    def text(self):
        return self.widget.text()

    def set_val(self, value):
        previous = self.widget.blockSignals(True)
        try:
            self.widget.setText(str(value))
        finally:
            self.widget.blockSignals(previous)


class QtLabelAdapter:
    def __init__(self, widget):
        self.widget = widget

    def set_text(self, value):
        self.widget.setText(str(value))

    def set_color(self, color):
        self.widget.setStyleSheet(f'color: {color};')


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan

def init_figure_forplt():
    h = plt.figure(figsize=(10.8, 10))
    _force_qt_arrow_cursor_for_figure(h)
    h.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.83)
    gs = GridSpec(1, 4)
    gs.update(wspace=0.22)
    axr = plt.subplot(gs[0, 0:2])
    axr.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axb = plt.subplot(gs[0, 2])
    axb.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axp = plt.subplot(gs[0, 3])
    axp.grid(color='gray', linestyle='--', linewidth=0.3, axis='both')
    return h, axr, axb, axp


def init_tall_preview_figure():
    h = plt.figure(figsize=(9.2, 14.2))
    _force_qt_arrow_cursor_for_figure(h)
    h.subplots_adjust(left=0.09, right=0.975, bottom=0.07, top=0.915)
    gs = GridSpec(1, 4)
    gs.update(wspace=0.18)
    axr = plt.subplot(gs[0, 0:2])
    axr.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axb = plt.subplot(gs[0, 2])
    axb.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axp = plt.subplot(gs[0, 3])
    axp.grid(color='gray', linestyle='--', linewidth=0.3, axis='both')
    position = axr.get_position()
    axr.set_position([position.x0, position.y0, position.width * 0.92, position.height])
    return h, axr, axb, axp


def init_standard_wave_figure():
    # Keep standard exports tall and compact in x.
    fig = plt.figure(figsize=(8.6, 14.2))
    _force_qt_arrow_cursor_for_figure(fig)
    fig.subplots_adjust(left=0.15, right=0.975, bottom=0.08, top=0.965)
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def _event_title_prefix(is_stack_mode=False):
    return "StackEvt" if is_stack_mode else "Evt"


def plot_waves(axr, axb, evtdata, enf, y_values=None):
    # bound = np.zeros(stadata.rflength)
    lines = []
    if y_values is None:
        y_values = evtdata.gcarc
    for i in range(evtdata.sta_num):
        max_amp = np.max(np.abs(evtdata.data[i]))
        if max_amp == 0:
            normalized = np.zeros_like(evtdata.data[i])
        else:
            normalized = evtdata.data[i] / max_amp
        datar = normalized * enf + y_values[i]
        trace = evtdata.wave_ori[i] if hasattr(evtdata, 'wave_ori') and i < len(evtdata.wave_ori) else None
        color, linewidth, alpha = _preview_trace_style(trace)
        line, = axr.plot(evtdata.time_axis, datar, linewidth=linewidth, color=color, alpha=alpha)
        lines.append(line)

    scatter = axb.scatter(evtdata.az, y_values, s=7)
    return lines, scatter


def plot_waves_only(axr, evtdata, enf, y_values=None):
    lines = []
    if y_values is None:
        y_values = evtdata.gcarc
    for i in range(evtdata.sta_num):
        max_amp = np.max(np.abs(evtdata.data[i]))
        if max_amp == 0:
            normalized = np.zeros_like(evtdata.data[i])
        else:
            normalized = evtdata.data[i] / max_amp
        datar = normalized * enf + y_values[i]
        trace = evtdata.wave_ori[i] if hasattr(evtdata, 'wave_ori') and i < len(evtdata.wave_ori) else None
        color, linewidth, alpha = _preview_trace_style(trace)
        line, = axr.plot(evtdata.time_axis, datar, linewidth=linewidth, color=color, alpha=alpha)
        lines.append(line)
    return lines


def _stack_member_visible_mask(evtdata):
    return np.asarray([
        getattr(tr.stats, 'dephasekit_stack_preview_role', '') != 'stack'
        for tr in evtdata.wave_ori
    ], dtype=bool)


def _preview_trace_style(trace):
    if trace is not None and getattr(trace.stats, 'dephasekit_stack_preview_role', '') == 'stack':
        return STACK_TRACE_COLOR, STACK_TRACE_LINEWIDTH, 0.95
    return MEMBER_TRACE_COLOR, MEMBER_TRACE_LINEWIDTH, 1.0


def plot_waves_with_masked_azimuth(axr, axb, evtdata, enf, y_values=None, azimuth_mask=None, azimuth_y_values=None):
    lines = []
    if y_values is None:
        y_values = evtdata.gcarc
    if azimuth_y_values is None:
        azimuth_y_values = y_values
    azimuth_y_values = np.asarray(azimuth_y_values, dtype=float)
    for i in range(evtdata.sta_num):
        max_amp = np.max(np.abs(evtdata.data[i]))
        if max_amp == 0:
            normalized = np.zeros_like(evtdata.data[i])
        else:
            normalized = evtdata.data[i] / max_amp
        datar = normalized * enf + y_values[i]
        trace = evtdata.wave_ori[i] if hasattr(evtdata, 'wave_ori') and i < len(evtdata.wave_ori) else None
        color, linewidth, alpha = _preview_trace_style(trace)
        line, = axr.plot(evtdata.time_axis, datar, linewidth=linewidth, color=color, alpha=alpha)
        lines.append(line)

    if azimuth_mask is None:
        scatter = axb.scatter(evtdata.az, azimuth_y_values, s=7)
    else:
        azimuth_mask = np.asarray(azimuth_mask, dtype=bool)
        scatter = axb.scatter(evtdata.az[azimuth_mask], azimuth_y_values[azimuth_mask], s=7)
        scatter._dephasekit_preview_full_indices = np.flatnonzero(azimuth_mask)
    return lines, scatter


def plot_standard_waves(axr, evtdata, y_values, colors, enf=1, linewidths=None):
    lines = []
    for i in range(evtdata.sta_num):
        max_amp = np.max(np.abs(evtdata.data[i]))
        if max_amp == 0:
            normalized = np.zeros_like(evtdata.data[i])
        else:
            normalized = evtdata.data[i] / max_amp
        datar = normalized * enf + y_values[i]
        linewidth = 0.35
        if linewidths is not None and i < len(linewidths):
            linewidth = linewidths[i]
        line, = axr.plot(evtdata.time_axis, datar, linewidth=linewidth, color=colors[i])
        lines.append(line)
    return lines


def set_wave_axis_only(axr, evtdata, t, title=None, show_ylabel=True, y_values=None, y_ticks=None, y_ticklabels=None,
                       ylabel=None):
    xmin = evtdata.x1
    xmax = evtdata.x2
    diff_x = xmax - xmin
    if diff_x <= 15:
        interval_x = 1
    elif 15 < diff_x <= 30:
        interval_x = 2
    elif 30 < diff_x <= 50:
        interval_x = 5
    else:
        interval_x = 10
    if y_values is None:
        y_values = np.asarray(evtdata.gcarc, dtype=float)
    else:
        y_values = np.asarray(y_values, dtype=float)
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    if np.isclose(ymin, ymax):
        ymin -= 2
        ymax += 2
    else:
        padding = max(0.8, 0.06 * (ymax - ymin))
        ymin -= padding
        ymax += padding
    if y_ticks is None:
        y_ticks = np.linspace(ymin, ymax, min(7, max(2, evtdata.sta_num)))
    if y_ticklabels is None:
        y_ticklabels = [f"{tick:g}" for tick in y_ticks]
    if ylabel is None:
        ylabel = r'Epicenter distance($^\circ$)'
    x_range = np.arange(xmin, xmax + interval_x, interval_x)
    axr.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axr.set_xlim(xmin, xmax)
    axr.set_xticks(x_range)
    axr.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axr.set_ylim(ymin, ymax)
    axr.set_yticks(y_ticks)
    if show_ylabel:
        axr.set_yticklabels(y_ticklabels, fontsize=8)
        axr.set_ylabel(ylabel, fontsize=13)
    else:
        axr.set_yticklabels([])
    axr.set_xlabel(f'Time after t{t} (s)', fontsize=13, labelpad=2)
    axr.add_line(Line2D([0, 0], axr.get_ylim(), color='black'))
    if title is not None:
        axr.set_title(title, fontsize=11)


def auto_x_tick_interval(diff_x):
    if diff_x <= 15:
        return 1
    if 15 < diff_x <= 30:
        return 2
    if 30 < diff_x <= 50:
        return 5
    return 10


def set_standard_wave_axis(axr, evtdata, axis_values, xlabel, ylabel, y_mode='gcarc', y_ticks=None, y_ticklabels=None):
    xmin = evtdata.x1
    xmax = evtdata.x2
    diff_x = xmax - xmin
    if diff_x <= 15:
        interval_x = 1
    elif 15 < diff_x <= 30:
        interval_x = 2
    elif 30 < diff_x <= 50:
        interval_x = 5
    else:
        interval_x = 10
    x_range = np.arange(xmin, xmax + interval_x, interval_x)
    axr.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
    axr.set_xlim(xmin, xmax)
    axr.set_xticks(x_range)
    axr.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ymin = float(np.min(axis_values))
    ymax = float(np.max(axis_values))
    if np.isclose(ymin, ymax):
        ymin -= 2
        ymax += 2
    else:
        ymin -= 2
        ymax += 2
    if y_ticks is None:
        if y_mode == 'az':
            tick_step = 10
        else:
            tick_step = 2
        ymin = int(np.floor(ymin / tick_step) * tick_step)
        ymax = int(np.ceil(ymax / tick_step) * tick_step)
        y_ticks = np.arange(ymin, ymax + tick_step, tick_step)
    if y_ticklabels is None:
        y_ticklabels = [f"{tick:g}" for tick in y_ticks]
    axr.set_ylim(ymin, ymax)
    axr.set_yticks(y_ticks)
    axr.set_yticklabels(y_ticklabels, fontsize=8)
    axr.set_xlabel(xlabel, fontsize=13, labelpad=2)
    axr.set_ylabel(ylabel, fontsize=13)


def set_page(self):
    axs = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]  # 将所有的子图放在一个列表中
    last_index = (self.ipage + 1) * 5
    first_index = last_index - 5
    for i in range(first_index, last_index, 1):
        if i < self.sta_num:
            ax_index = i % 5
            # axs[ax_index].plot([0, 0], [0, self.axpages * self.maxidx + 1], color="black")
            x1 = round(self.tmarker_t[i] + self.xlim[0])
            x2 = round(self.tmarker_t[i] + self.xlim[1])
            axs[ax_index].set_xlim(x1, x2)
            diff = x2 - x1
            if diff <= 15:
                interval = 1
            elif 15 < diff <= 30:
                interval = 2
            elif 30 < diff <= 50:
                interval = 5
            else:
                interval = 10
            # major
            # axs[ax_index].xaxis.set_major_locator(ticker.MultipleLocator(10))
            # minor
            axs[ax_index].xaxis.set_minor_locator(ticker.MultipleLocator(1))
            #
            axs[ax_index].set_xticks(np.arange(x1, x2, interval))
            #
            axs[ax_index].tick_params(axis='x', which='major', length=4, labelsize=10)  # 主刻度标签的大小
            axs[ax_index].tick_params(axis='x', which='minor', length=1, labelsize=0)  # 隐藏次刻度标签
        else:
            print("last pages, all {} waveforms".format(self.sta_num))



def set_fig(axr, axb, evtdata ,t, interval_x_override=None, y_values=None, y_ticks=None, y_ticklabels=None,
            ylabel=None):
    xmin = evtdata.x1
    xmax = evtdata.x2
    diff_x = xmax - xmin # 12
    # y_range = np.arange(stadata.sta_num) + 1
    if interval_x_override is not None:
        interval_x = interval_x_override
    else:
        interval_x = auto_x_tick_interval(diff_x)
    if y_values is None:
        y_values = np.asarray(evtdata.gcarc, dtype=float)
    else:
        y_values = np.asarray(y_values, dtype=float)
    ymin = float(np.min(y_values))
    ymax = float(np.max(y_values))
    if np.isclose(ymin, ymax):
        ymin -= 2
        ymax += 2
    else:
        padding = max(0.8, 0.06 * (ymax - ymin))
        ymin -= padding
        ymax += padding
    if y_ticks is None:
        y_ticks = np.linspace(ymin, ymax, min(7, max(2, evtdata.sta_num)))
    if y_ticklabels is None:
        y_ticklabels = [f"{tick:g}" for tick in y_ticks]
    if ylabel is None:
        ylabel = r'Epicenter distance($^\circ$)'
    x_range = np.arange(xmin, xmax + interval_x, interval_x)
    # space = 1
	
    # set axr
    axr.set_xlim(xmin, xmax)

    # axr.set_xticks(np.arange(xmin, xmax, interval))
    axr.set_xticks(x_range)
    # axr.set_xticklabels(x_range, fontsize=8)
    axr.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    # axr.set_ylim(0, stadata.sta_num + space)
    axr.set_ylim(ymin, ymax)  # Y轴范围根据震中距的数量设置
    axr.set_yticks(y_ticks)
    # axr.set_yticklabels(stadata.stnames, fontsize=5)
    axr.set_yticklabels(y_ticklabels, fontsize=8)
	
    axr.set_xlabel(f'Time after t{t} (s)', fontsize=13, labelpad=2)
	
    # axr.set_ylabel('Sta', fontsize=13)
    axr.set_ylabel(ylabel, fontsize=13)
    axr.add_line(Line2D([0, 0], axr.get_ylim(), color='black'))
    # axr.set_title('R components ({})'.format(evtdata.staname), fontsize=16)

    # set axb
    axb.set_xlim(0, 360)
    axb.set_xticks(np.linspace(0, 360, 7))
    axb.set_xticklabels(np.linspace(0, 360, 7, dtype='i'), fontsize=8)
    # axb.set_ylim(0, stadata.sta_num + space)
    axb.set_ylim(ymin, ymax)
    axb.set_yticks(y_ticks)
    axb.set_yticklabels(y_ticklabels, fontsize=5)
    axb.set_xlabel(r'Azimuth ($^\circ$)', fontsize=13)
    fig = axr.figure
    fig.suptitle(
        "{}:{}\n Latitude: {:.2f}\N{DEGREE SIGN}, Longitude: {:.2f}\N{DEGREE SIGN}, Depth:{:.1f} km".format(
            _event_title_prefix(getattr(evtdata, 'is_stack_mode', False)),
            evtdata.evtname, evtdata.evla, evtdata.evlo, evtdata.evdp),
        fontsize=16)


def set_stack_overlay_fig(axr, axb, evtdata, t, amplitude_scale=1.0, interval_x_override=None,
                          azimuth_y_values=None):
    xmin = evtdata.x1
    xmax = evtdata.x2
    diff_x = xmax - xmin
    interval_x = interval_x_override if interval_x_override is not None else auto_x_tick_interval(diff_x)
    x_range = np.arange(xmin, xmax + interval_x, interval_x)

    axr.set_xlim(xmin, xmax)
    axr.set_xticks(x_range)
    axr.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    y_limit = max(1.2, float(amplitude_scale) + 0.25)
    axr.set_ylim(-y_limit, y_limit)
    axr.set_yticks([-1.0, 0.0, 1.0])
    axr.set_yticklabels(['-1', '0', '1'], fontsize=8)
    axr.set_xlabel(f'Time after t{t} (s)', fontsize=13, labelpad=2)
    axr.set_ylabel('Normalized amplitude', fontsize=13)
    axr.add_line(Line2D([0, 0], axr.get_ylim(), color='black'))

    if azimuth_y_values is None:
        azimuth_y_values = np.asarray(evtdata.gcarc, dtype=float)
    else:
        azimuth_y_values = np.asarray(azimuth_y_values, dtype=float)
    finite_y = azimuth_y_values[np.isfinite(azimuth_y_values)]
    if finite_y.size == 0:
        ymin, ymax = 0.0, 1.0
    else:
        ymin = float(np.min(finite_y))
        ymax = float(np.max(finite_y))
    if np.isclose(ymin, ymax):
        ymin -= 2.0
        ymax += 2.0
    else:
        padding = max(0.8, 0.06 * (ymax - ymin))
        ymin -= padding
        ymax += padding
    y_ticks = np.linspace(ymin, ymax, min(7, max(2, evtdata.sta_num)))

    axb.set_xlim(0, 360)
    axb.set_xticks(np.linspace(0, 360, 7))
    axb.set_xticklabels(np.linspace(0, 360, 7, dtype='i'), fontsize=8)
    axb.set_ylim(ymin, ymax)
    axb.set_yticks(y_ticks)
    axb.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=5)
    axb.set_xlabel(r'Azimuth ($^\circ$)', fontsize=13)

    fig = axr.figure
    fig.suptitle(
        "{}:{}\n Latitude: {:.2f}\N{DEGREE SIGN}, Longitude: {:.2f}\N{DEGREE SIGN}, Depth:{:.1f} km".format(
            _event_title_prefix(getattr(evtdata, 'is_stack_mode', False)),
            evtdata.evtname, evtdata.evla, evtdata.evlo, evtdata.evdp),
        fontsize=16)

# def getjulday(s):
#     year, month, day, hour, min, sec = s.split("_")
#     date = datetime(int(year), int(month), int(day))
#     julian_day = date.toordinal() - datetime(int(year), 1, 1).toordinal() + 1
#     result = f"{year}.{julian_day:03d}.{hour}.{min}.{sec}"
#     return result


def indexpags(sta_num, maxidx=5):
    full_pages = sta_num // maxidx
    if np.mod(sta_num, maxidx) == 0:
        axpages = full_pages
    else:
        axpages = full_pages + 1
    waveidx = []
    for i in range(axpages - 1):
        waveidx.append(np.arange(maxidx * i, maxidx * (i + 1)))
    waveidx.append(np.arange(maxidx * (axpages - 1), sta_num))
    return axpages, waveidx


class EvtData():
    def __init__(self, waveform_stream, r_tlst, x1, x2, dt, order='gcarc', event_name_override=None):
        self.dt = dt
        self.wave_ori = waveform_stream
        self.is_stack_mode = any(
            str(getattr(tr.stats, 'network', '')).upper() == 'DPK'
            or str(getattr(tr.stats, 'station', '')).upper() == 'STACK'
            for tr in self.wave_ori
        )
        self.reference_t = r_tlst
        self.sta_num = len(r_tlst)
        self.wavelength = max(1, int(round((x2 - x1) / self.dt)))
        self.data = np.empty([self.sta_num, self.wavelength])
        self.baz = np.array([_sac_float(tr, 'baz', 0.0) for tr in self.wave_ori], dtype=float)
        self.az = np.array([_sac_float(tr, 'az', 0.0) for tr in self.wave_ori], dtype=float)
        self.gcarc = np.array([_sac_float(tr, 'gcarc', 0.0) for tr in self.wave_ori], dtype=float)
        self.year = str(int(_safe_float(_sac_attr(self.wave_ori[0], 'nzyear', 0)))).zfill(4)
        self.jday = str(int(_safe_float(_sac_attr(self.wave_ori[0], 'nzjday', 0)))).zfill(3)
        self.hour = str(int(_safe_float(_sac_attr(self.wave_ori[0], 'nzhour', 0)))).zfill(2)
        self.min = str(int(_safe_float(_sac_attr(self.wave_ori[0], 'nzmin', 0)))).zfill(2)
        self.sec = str(int(_safe_float(_sac_attr(self.wave_ori[0], 'nzsec', 0)))).zfill(2)
        default_evtname = f"{self.year}.{self.jday}.{self.hour}.{self.min}.{self.sec}"
        self.evtname = str(event_name_override or default_evtname)
        self.evla = _sac_float(self.wave_ori[0], 'evla', 0.0)
        self.evlo = _sac_float(self.wave_ori[0], 'evlo', 0.0)
        self.evdp = _sac_float(self.wave_ori[0], 'evdp', 0.0)
        # self.network = np.array([tr.stats.network for tr in self.wave_ori])
        # self.station = np.array([tr.stats.station for tr in self.wave_ori])
        # self.stnames = [f"{n}.{s}" for n, s in zip(self.network, self.station)]
        self._sort2(order)
        for i in range(self.sta_num):
            t1_index = int(round((self.reference_t[i] - _sac_float(self.wave_ori[i], 'b', 0.0)) / self.dt))
            start_index = t1_index + int(round(x1 / self.dt))
            end_index = start_index + self.wavelength
            self.data[i] = self._extract_window_with_padding(self.wave_ori[i].data, start_index, end_index)
        # Derive the preview time axis from the same sample count used for the
        # waveform slices so x/y lengths always stay aligned.
        self.time_axis = x1 + np.arange(self.wavelength, dtype=float) * self.dt
        self.x1 = x1
        self.x2 = x2
        self.max_gcarc = np.max(self.gcarc)
        self.min_gcarc = np.min(self.gcarc)
        self.max_baz = np.max(self.baz)
        self.min_baz = np.min(self.baz)
        self.max_az = np.max(self.az)
        self.min_az = np.min(self.az)

    def _extract_window_with_padding(self, trace_data, start_index, end_index):
        window = np.zeros(self.wavelength, dtype=float)
        trace_length = len(trace_data)
        clipped_start = max(0, start_index)
        clipped_end = min(trace_length, end_index)
        if clipped_end <= clipped_start:
            return window
        source = np.asarray(trace_data[clipped_start:clipped_end], dtype=float)
        target_start = clipped_start - start_index
        target_end = target_start + len(source)
        window[target_start:target_end] = source
        return window


    def _sort2(self, order):
        if order == 'baz':
            idx = np.argsort(self.baz)
        elif order == 'az':
            idx = np.argsort(self.az)
        elif order == 'gcarc':
            idx = np.argsort(self.gcarc)
        # elif order == 'date':
            # idx = pd.to_datet ime(self.filenames, format='%Y.%j.%H.%M.%S').argsort()
        else:
            pass
        self.baz = self.baz[idx]
        self.az = self.az[idx]
        self.gcarc = self.gcarc[idx]
        self.reference_t = self.reference_t[idx]
        self.wave_ori = obspy.Stream([self.wave_ori[i] for i in idx])
        # self.stnames = [self.stnames[i] for i in idx]


class WaveFigure(Figure):
    def __init__(self, wavepath, xlim , tmarker, suffix,ta_tb,xlim_preview ,width=21, height=11, dpi=100, axis_mode='absolute', member_filter=None ):
        super(WaveFigure, self).__init__()
        self.width = width
        self.height = height
        self.dpi = dpi
        # 可选：只读入指定 wave_name 集合的 SAC（用于从厚度审阅直跳到某 group
        # 的拾取窗时，主图只显示该 group 成员）。None=读全部。
        self.member_filter = set(member_filter) if member_filter else None
        self.wavepath = wavepath
        self.runtime_event_dir = str(source_event_dir_for_runtime(wavepath))
        self.stack_mode = is_stack_event_dir(wavepath)
        self.stack_repair_report = None
        self.stack_health_report = None
        if self.stack_mode:
            self.stack_repair_report = repair_stack_event_metadata(wavepath, persist=True)
            self.stack_health_report = inspect_stack_event_health(wavepath)
        self.stack_event_marker = load_stack_event_marker(wavepath) if self.stack_mode else {}
        self.stack_sidecars = load_stack_sidecar_map(wavepath) if self.stack_mode else {}
        self.tmarker = tmarker
        self.enf = 1
        self.dt = None
        self.ipage = 0
        self.maxidx = 5
        self.xlim = xlim
        self.axis_mode = axis_mode
        self.plotfig = None
        self.marker_styles = {
            '0': ('t0', '#d62728'),
            '1': ('t1', '#00ffff'),
            '2': ('t2', '#0000ff'),
            '3': ('t3', '#008000'),
            '4': ('t4', '#ff7f0e'),
            '5': ('t5', '#a52a2a'),
            '6': ('t6', '#800080'),
            '7': ('t7', '#8b0000'),
            '8': ('t8', '#00ffff'),
            '9': ('t9', '#ff7f0e'),
        }
        self.phase_display_labels = {
            '0': 'P',
            '1': 'PcP',
            '2': 'pP',
            '3': 'sP',
            '5': 'sP',
            '6': 'pP',
            '7': 'P',
            '8': 'pmP',
            '9': 'smP',
        }
        self.phase_label_prefixes = {
            '0': 'Theory',
            '1': 'Theory',
            '2': 'Theory',
            '3': 'Theory',
            '5': 'Actual',
            '6': 'Actual',
            '7': 'Actual',
            '8': 'Theory',
            '9': 'Theory',
        }
        self.markers = {key: {} for key in self.marker_styles}
        self.user_markers = {
            'user1': {},
            'user2': {},
            'user3': {},
            'user4': {},
            'user5': {},
        }
        self.key = None
        self.suffix = suffix
        preview_phases = [item.strip() for item in ta_tb.split(',') if item.strip()]
        self.preview_modes = []
        for i, phase in enumerate(preview_phases):
            if not phase.startswith('t') or len(phase) < 2:
                raise ValueError(f'Invalid preview phase: {phase}')
            marker_key = phase[1:]
            if marker_key not in self.marker_styles or marker_key == 'm':
                raise ValueError(f'Unsupported preview phase: {phase}')
            if xlim_preview is None:
                window_start, window_end = self._default_xlim_for_marker(phase)
            else:
                window_start = xlim_preview[i * 2]
                window_end = xlim_preview[i * 2 + 1]
            self.preview_modes.append([marker_key, window_start, window_end])
        self.standard_export_phase_tokens = 't0,t1,t2,t3,t5,t6,t7,t8,t9'
        self.preview_mark_color = '#9a1fff'
        self.preview_selected_mark_color = '#d2691e'
        self.user5_mark_color = '#00bcd4'
        self.user5_selected_color = '#aef7ff'
        self.user1_mark_color = '#008f5a'
        self.user1_selected_color = '#ffd400'
        self.user4_mark_color = '#ff9f1c'
        self.user4_selected_color = '#ffd166'
        self.preview_amplitude_scale = 1.0
        self.preview_amplitude_step = 0.1
        self.preview_amplitude_min = 0.0
        self.preview_amplitude_max = 3.0
        self.preview_amplitude_presets = self._load_preview_amplitude_presets()
        self.preview_peak_half_window_default = 1.0
        self.preview_peak_pick_mode = 'pkm'
        self.preview_x_tick_interval_override = None
        self.preview_alignment_nudge_steps = {
            'fine': 1,
            'normal': 5,
            'large': 20,
        }
        self.preview_curve_pick_color = '#ff7a00'
        self.preview_amplitude_restore_scale = 1.0
        self.preview_amplitude_hidden_mode = False
        self.preview_hidden_wave_names = set()
        self.preview_hidden_batches = []
        self.preview_trace_layout_mode = 'real'
        self.preview_view_mode = 'tall'
        self.preview_even_spacing_step = 1.0
        self.preview_even_spacing_adjust_step = 0.2
        self.preview_even_spacing_min = 0.6
        self.preview_even_spacing_max = 6.0
        self.bandpass_settings = {
            'freqmin': None,
            'freqmax': None,
            'corners': 2,
            'passes': 1,
        }
        self.compare_preset_profiles = []
        self.compare_default_bandpass_profiles = []
        self.compare_bandpass_profiles = []
        self.comparefig = None
        self.preview_control_dock = None
        self.max_compare_columns = 4
        self.current_pick_wave_name = None
        self.current_pick_station_name = None
        self.preview_jump_highlight_wave_name = None
        self.preview_selected_wave_names = set()
        self.preview_pierce_phase = 'pP'
        self.preview_pierce_model = 'iasp91'
        self.preview_pierce_depth_km = DEFAULT_PIERCE_DEPTH_KM
        self.preview_pierce_output_root = str(DEFAULT_PIERCE_OUTPUT_ROOT)
        self.preview_pierce_taup_bin = str(DEFAULT_TAUP_BIN)
        self.stack_crustal_taup_bin = str(DEFAULT_STACK_TAUP_BIN)
        self._stack_crustal_summary_cache = {}
        self._preview_warm_cache_keys = set()
        self._preview_backend_warmed = False
        self.preview_pierce_cache = {}
        self.preview_pierce_generation_attempted = False
        self.preview_pierce_missing_notice_shown = False
        self.preview_deferred_panel_threshold = 220
        self.preview_pierce_autogen_threshold = 120
        self.preview_pierce_async_threshold = 140
        self.preview_pierce_selection_mode = 'point'
        self.preview_keep_selection_mode = 'selected'
        self.preview_pierce_range_locked = False
        self.preview_pierce_fixed_bounds = None
        self.preview_group_overlay_enabled = False
        self.preview_ungrouped_only_enabled = False
        self.preview_stack_show_member_pierce = True
        self.preview_stack_display_mode = 'top'
        self.theory_time_model = 'iasp91'
        self.theory_time_cache = {}
        self.theory_delta_summary_cache = None
        self.standard_export_options = {
            'event_name': True,
            'depth': True,
            'bandpass': True,
            'theory_model': True,
            'gcarc_mean': True,
            'pp_minus_p': True,
            'sp_minus_p': True,
            'phase_legend': False,
            'phase_keys': [str(i) for i in range(10)],
            'export_gcarc': True,
            'export_az': False,
            'export_pierce': False,
            'export_pierce_group': False,
        }
        self.pick_mode_armed = False
        self.jump_status_callback = None
        self.compare_status_callback = None
        self.compare_defaults_update_callback = None
        self.phase_tokens_change_callback = None
        self.stack_review_refresh_callback = None
        self.jump_target_mode = 'user2'
        self.stack_manual_marker_keys = set()
        self.stack_manual_user_keys = set()
        self.dirty_marker_wave_names = set()
        self._pending_source_marker_writes = {}
        self._source_marker_flush_timer = None
        self._source_marker_flush_scheduled = False
        self._source_marker_flush_delay_ms = 600

    def _stack_sidecar_for_wave(self, wave_name):
        if not getattr(self, 'stack_mode', False):
            return {}
        return getattr(self, 'stack_sidecars', {}).get(str(wave_name), {}) or {}

    def _is_stack_trace_align_marker(self, wave_name, marker_key):
        """True if wave_name is a stack trace and marker_key is its structural
        align marker. The align marker is read-only on stack traces: it sits at
        -x1 by construction (the stacked data is aligned there), so editing it
        in the pick window can't be persisted meaningfully and would only cause
        the 'changed but reverts on reload' confusion. Re-align by re-stacking."""
        if not getattr(self, 'stack_mode', False):
            return False
        try:
            is_stack_trace = str(wave_name) in self.ori_sacnames
        except Exception:
            is_stack_trace = False
        if not is_stack_trace:
            return False
        align_key = self._stack_align_marker_key(str(wave_name))
        if align_key is None:
            return False
        return str(marker_key) == str(align_key)

    def _semantic_event_dir(self):
        return str(getattr(self, 'runtime_event_dir', getattr(self, 'wavepath', '')))

    def _semantic_event_name(self):
        marker = getattr(self, 'stack_event_marker', {}) or {}
        marker_name = str(marker.get('source_event_name', '')).strip()
        if marker_name:
            return marker_name
        dsm_event_name = _event_name_from_dsm_path(getattr(self, 'wavepath', ''))
        if dsm_event_name:
            return dsm_event_name
        event_dir = self._semantic_event_dir()
        if event_dir:
            return os.path.basename(os.path.abspath(event_dir))
        return ''

    def _stack_meta_float(self, wave_name, key, default=math.nan):
        sidecar = self._stack_sidecar_for_wave(wave_name)
        value = sidecar.get(key, default)
        return _safe_float(value)

    def _stack_meta_text(self, wave_name, key, default=''):
        sidecar = self._stack_sidecar_for_wave(wave_name)
        value = sidecar.get(key, default)
        if value is None:
            return default
        return str(value)

    def _stack_meta_int(self, wave_name, key, default=0):
        sidecar = self._stack_sidecar_for_wave(wave_name)
        value = sidecar.get(key, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    def _stack_wave_summary(self, wave_name):
        if not self.stack_mode:
            return ''
        sidecar = self._stack_sidecar_for_wave(wave_name)
        if not sidecar:
            return ''
        return stack_wave_summary_from_sidecar(sidecar)

    def _stack_marker_value(self, wave_name, marker_key):
        bucket = self.markers.get(str(marker_key), {})
        key = str(wave_name)
        if key in bucket:
            # Present in memory (even NaN = explicitly deleted/absent): this is
            # authoritative. Do NOT fall back to the sidecar, otherwise a deleted
            # align marker (e.g. t5) keeps resurfacing via the stale sidecar
            # value and labels like "sP-smP" never clear.
            return _safe_float(bucket[key])
        sidecar = self._stack_sidecar_for_wave(wave_name)
        markers = dict(sidecar.get('markers', {}) or {})
        return _safe_float(markers.get(f't{self._normalize_marker_key(marker_key)}', math.nan))

    def _stack_crustal_ray_parameter(self, evdp, gcarc, phase, model='prem'):
        if not hasattr(self, '_stack_crustal_ray_param_cache'):
            self._stack_crustal_ray_param_cache = {}
        cache_key = (
            str(self.stack_crustal_taup_bin),
            _safe_float(evdp),
            _safe_float(gcarc),
            str(phase),
            str(model),
        )
        cached = self._stack_crustal_ray_param_cache.get(cache_key)
        if cached is not None:
            return cached
        ray_param = fetch_taup_ray_parameter(
            self.stack_crustal_taup_bin,
            evdp_km=evdp,
            gcarc_deg=gcarc,
            phase=phase,
            model=model,
        )
        self._stack_crustal_ray_param_cache[cache_key] = ray_param
        return ray_param

    def _stack_crustal_summary(self, wave_name):
        wave_name = str(wave_name or '')
        if not getattr(self, 'stack_mode', False) or not wave_name:
            return {}
        sidecar = self._stack_sidecar_for_wave(wave_name)
        if not sidecar:
            return {}
        marker_signature = tuple(
            self._stack_marker_value(wave_name, marker_key)
            for marker_key in ('5', '6', '8', '9')
        )
        geometry = sidecar.get('geometry', {}) or {}
        event_info = sidecar.get('event', {}) or {}
        cache_key = (
            wave_name,
            marker_signature,
            _safe_float(geometry.get('pierce_lon_mean', math.nan)),
            _safe_float(geometry.get('pierce_lat_mean', math.nan)),
            _safe_float(geometry.get('gcarc_mean', math.nan)),
            _safe_float(event_info.get('evdp', math.nan)),
        )
        if not hasattr(self, '_stack_crustal_summary_cache'):
            self._stack_crustal_summary_cache = {}
        cached = self._stack_crustal_summary_cache.get(cache_key)
        if cached is not None:
            return cached

        summary = {}
        gcarc = _safe_float(geometry.get('gcarc_mean', math.nan))
        evdp = _safe_float(event_info.get('evdp', math.nan))
        if math.isnan(gcarc) or math.isnan(evdp):
            self._stack_crustal_summary_cache[cache_key] = summary
            return summary

        t5 = self._stack_marker_value(wave_name, '5')
        t6 = self._stack_marker_value(wave_name, '6')
        t8 = self._stack_marker_value(wave_name, '8')
        t9 = self._stack_marker_value(wave_name, '9')

        # 与厚度审阅列表复用同一条 stack-group 计算链：
        # gcarc_mean + evdp -> TauP ray parameter，再用统一厚度公式。
        if not math.isnan(t6) and not math.isnan(t8):
            try:
                ray_param_pp = self._stack_crustal_ray_parameter(evdp, gcarc, phase='pP', model='prem')
                thickness_pp = calculate_pp_pmp_thickness(t6 - t8, DEFAULT_CRUST_VP, ray_param_pp)
                if not math.isnan(thickness_pp) and thickness_pp > 0.0:
                    summary['pp_pmp_km'] = float(thickness_pp)
            except Exception:
                pass

        if not math.isnan(t5) and not math.isnan(t9):
            try:
                ray_param_sp = self._stack_crustal_ray_parameter(evdp, gcarc, phase='sP', model='prem')
                thickness_sp = calculate_sp_smp_thickness(t5 - t9, DEFAULT_CRUST_VP, DEFAULT_CRUST_VS, ray_param_sp)
                if not math.isnan(thickness_sp) and thickness_sp > 0.0:
                    summary['sp_smp_km'] = float(thickness_sp)
            except Exception:
                pass

        self._stack_crustal_summary_cache[cache_key] = summary
        return summary

    def _stack_crustal_summary_text(self, wave_name):
        summary = self._stack_crustal_summary(wave_name)
        if not summary:
            return ''
        parts = []
        if 'pp_pmp_km' in summary:
            parts.append(f"pP-pmP:{summary['pp_pmp_km']:.2f} km")
        if 'sp_smp_km' in summary:
            parts.append(f"sP-smP:{summary['sp_smp_km']:.2f} km")
        return '    '.join(parts)

    def _read_trace_coords_from_disk(self, wave_name):
        """从磁盘读 SAC 头段，返回 (evla, evlo, evdp, stla, stlo)，读不到返回 None。"""
        wave_name = str(wave_name or '')
        if not wave_name:
            return None
        wavepath = getattr(self, 'wavepath', None)
        if not wavepath:
            return None
        sac_path = os.path.join(str(wavepath), wave_name)
        if not os.path.isfile(sac_path):
            return None
        try:
            tr = obspy.read(sac_path, headonly=True)[0]
        except Exception:
            return None
        h = tr.stats.sac
        return (
            _sac_float(tr, 'evla', math.nan),
            _sac_float(tr, 'evlo', math.nan),
            _sac_float(tr, 'evdp', math.nan),
            _sac_float(tr, 'stla', math.nan),
            _sac_float(tr, 'stlo', math.nan),
        )

    def _single_trace_ray_parameter(self, coords, phase, model='prem'):
        if not hasattr(self, '_single_trace_ray_param_cache'):
            self._single_trace_ray_param_cache = {}
        cache_key = tuple(_safe_float(value) for value in coords) + (str(phase), str(model))
        cached = self._single_trace_ray_param_cache.get(cache_key)
        if cached is not None:
            return cached
        evla, evlo, evdp, stla, stlo = coords
        ray_param = fetch_obspy_ray_parameter(
            evdp, evla, evlo, stla, stlo,
            phase=phase,
            model=model,
        )
        self._single_trace_ray_param_cache[cache_key] = ray_param
        return ray_param

    def _single_trace_crustal_summary(self, wave_name):
        """单条波形厚度：完全照搬 calucate_xmP.py。

        用 obspy get_ray_paths_geo 取路径首点 ray param，Vp/Vs 写死 5.8/3.2。
        仅在非 stack 模式下使用（stack 模式由 _stack_crustal_summary 负责）。
        """
        if getattr(self, 'stack_mode', False):
            return {}
        wave_name = str(wave_name or '')
        if not wave_name:
            return {}
        if not hasattr(self, '_single_trace_crustal_cache'):
            self._single_trace_crustal_cache = {}

        t5 = self._stack_marker_value(wave_name, '5')
        t6 = self._stack_marker_value(wave_name, '6')
        t8 = self._stack_marker_value(wave_name, '8')
        t9 = self._stack_marker_value(wave_name, '9')
        marker_sig = (t5, t6, t8, t9)
        cache_key = (wave_name, marker_sig)
        cached = self._single_trace_crustal_cache.get(cache_key)
        if cached is not None:
            return cached

        summary = {}
        # 直接从磁盘读 SAC 头段取事件/台站坐标，不依赖 self 的波形状态
        # （单条拾取窗在不同初始化阶段 self.evla/wave_ori 可能不可用）。
        if not hasattr(self, '_single_trace_coord_cache'):
            self._single_trace_coord_cache = {}
        coords = self._single_trace_coord_cache.get(wave_name)
        if coords is None:
            coords = self._read_trace_coords_from_disk(wave_name)
            self._single_trace_coord_cache[wave_name] = coords
        if coords is None:
            self._single_trace_crustal_cache[cache_key] = summary
            return summary
        evla, evlo, evdp, stla, stlo = coords
        if (math.isnan(evla) or math.isnan(evlo) or math.isnan(evdp)
                or math.isnan(stla) or math.isnan(stlo)):
            self._single_trace_crustal_cache[cache_key] = summary
            return summary

        if not math.isnan(t6) and not math.isnan(t8):
            try:
                ray_param = self._single_trace_ray_parameter(coords, phase='pP', model='prem')
                thickness = calculate_single_trace_thickness(t6 - t8, ray_param, 'p')
                if not math.isnan(thickness) and thickness > 0.0:
                    summary['pp_pmp_km'] = float(thickness)
            except Exception:
                pass

        if not math.isnan(t5) and not math.isnan(t9):
            try:
                ray_param = self._single_trace_ray_parameter(coords, phase='sP', model='prem')
                thickness = calculate_single_trace_thickness(t5 - t9, ray_param, 's')
                if not math.isnan(thickness) and thickness > 0.0:
                    summary['sp_smp_km'] = float(thickness)
            except Exception:
                pass

        self._single_trace_crustal_cache[cache_key] = summary
        return summary

    def _single_trace_crustal_text(self, wave_name):
        summary = self._single_trace_crustal_summary(wave_name)
        if not summary:
            return ''
        parts = []
        if 'pp_pmp_km' in summary:
            parts.append(f"pP-pmP:{summary['pp_pmp_km']:.2f} km")
        if 'sp_smp_km' in summary:
            parts.append(f"sP-smP:{summary['sp_smp_km']:.2f} km")
        return '    '.join(parts)

    def _stack_preview_stack_wave_names(self):
        if not getattr(self, 'stack_mode', False):
            return []
        return [
            str(wave_name)
            for wave_name in self.ori_sacnames
            if self._stack_sidecar_for_wave(wave_name)
        ]

    def _current_stack_preview_wave_name(self):
        stack_names = self._stack_preview_stack_wave_names()
        if not stack_names:
            return None
        current_name = str(getattr(self, 'current_pick_wave_name', '') or '')
        if current_name in stack_names:
            return current_name
        try:
            visible_index = self._visible_wave_index_for_page_slot(0)
        except Exception:
            visible_index = None
        if visible_index is not None and 0 <= visible_index < len(self.ori_sacnames):
            visible_name = str(self.ori_sacnames[visible_index])
            if visible_name in stack_names:
                return visible_name
        return stack_names[0]

    def _warm_preview_backend_resources(self):
        if getattr(self, '_preview_backend_warmed', False):
            return False
        self._preview_backend_warmed = True
        fig = None
        was_interactive = plt.isinteractive()
        try:
            plt.ioff()
            fig = plt.figure(figsize=(0.8, 0.6))
            _force_qt_arrow_cursor_for_figure(fig)
            ax = fig.add_subplot(1, 1, 1)
            ax.plot([0.0, 1.0], [0.0, 1.0], linewidth=0.5)
            ax.text(0.5, 0.5, 'DePhaseKit', ha='center', va='center', fontsize=6)
            ax.set_axis_off()
            fig.canvas.draw()
            return True
        except Exception:
            return False
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass
            try:
                if was_interactive:
                    plt.ion()
                else:
                    plt.ioff()
            except Exception:
                pass

    def _preview_warm_cache_key(self, preview_index):
        if preview_index < 0 or preview_index >= len(getattr(self, 'preview_modes', [])):
            return None
        tmarker, _x1, _x2 = self.preview_modes[preview_index]
        stack_name = None
        if getattr(self, 'stack_mode', False):
            stack_name = self._current_stack_preview_wave_name()
        bandpass_profile = self._current_bandpass_profile()
        return (
            int(preview_index),
            str(tmarker),
            stack_name,
            self._bandpass_profile_key(bandpass_profile) if bandpass_profile is not None else None,
        )

    def warm_preview_resources(self, preview_index=0):
        """Preload the resources used by the first preview without showing it."""
        cache_key = self._preview_warm_cache_key(preview_index)
        if cache_key is None or cache_key in self._preview_warm_cache_keys:
            return False
        self._preview_warm_cache_keys.add(cache_key)
        try:
            self._warm_preview_backend_resources()
            tmarker = self.preview_modes[preview_index][0]
            waves, _t_lst, _reference_times = self._collect_preview_display_stream(tmarker)
            if not getattr(self, 'stack_mode', False) and len(waves) > 0:
                metadata = [
                    {'wave_name': getattr(tr.stats, 'dephasekit_wave_name', '')}
                    for tr in waves
                ]
                self._maybe_generate_current_preview_pierce_cache(metadata)
                self._load_pierce_points_for_current_event(auto_generate=False)
            return True
        except Exception:
            return False

    def _stack_preview_member_wave_names(self, stack_wave_name):
        sidecar = self._stack_sidecar_for_wave(stack_wave_name)
        # 旧格式：成员名在 wave_names_used / wave_names_aligned / wave_names_requested
        for key in ('wave_names_used', 'wave_names_aligned', 'wave_names_requested'):
            names = [
                str(value)
                for value in (sidecar.get(key, []) or [])
                if str(value)
            ]
            if names:
                return names
        # 新格式：outputs.members 指向 result_package_dir/members.txt
        import csv as _csv
        try:
            result_pkg = Path(sidecar.get('result_package_dir', ''))
            if not result_pkg.exists():
                return []
            members_file = result_pkg / "members.txt"
            if not members_file.exists():
                return []
            with open(members_file, encoding="utf-8") as fh:
                reader = _csv.DictReader(fh, delimiter="\t")
                return [
                    str(row["wave_name"]).strip()
                    for row in reader
                    if str(row.get("status", "")).strip().lower() == "used"
                ]
        except Exception:
            return []

    def _seed_member_marker_cache(self, member_name, member_trace):
        # Seed self.markers with a member waveform's t0-t9 from the on-disk SAC
        # header each time the member is loaded for stack preview. Preview reads
        # prefer self.markers over the SAC header, so this cache must reflect the
        # latest disk state: the original-waveform preview (a separate
        # WaveFigure instance) writes marker edits to the member SAC, and without
        # re-seeding this stack preview would keep showing the stale first-load
        # values. Manual picks made in THIS stack preview are tracked in
        # stack_manual_marker_keys and preserved (never overwritten by disk).
        if not hasattr(self, 'markers'):
            return
        member_name = str(member_name)
        manual_keys = getattr(self, 'stack_manual_marker_keys', set())
        for idx in range(10):
            marker_key = str(idx)
            bucket = self.markers.get(marker_key)
            if bucket is None:
                continue
            if (member_name, marker_key) in manual_keys:
                # User edited this marker in this stack preview; keep their pick.
                continue
            bucket[member_name] = _sac_float(member_trace, f't{marker_key}', math.nan)
        # Seed user markers (user1/2/4/5) too, so member flip (user4) and
        # quality coloring applied in _prepare_stack_preview_trace reflect the
        # source SAC. Without this, members never appear flipped in the stack
        # preview because self.user_markers is only populated for stack traces.
        manual_user_keys = getattr(self, 'stack_manual_user_keys', set())
        for user_key in ('user1', 'user2', 'user4', 'user5'):
            bucket = self.user_markers.get(user_key) if hasattr(self, 'user_markers') else None
            if bucket is None:
                continue
            if (member_name, user_key) in manual_user_keys:
                continue
            bucket[member_name] = _sac_float(member_trace, user_key, math.nan)

    def _stack_preview_reference_time(self, trace, tmarker, wave_name=None):
        marker_key = self._normalize_marker_key(tmarker)
        # Prefer the in-memory marker (reflects in-session edits without a
        # restart); fall back to the on-disk SAC header. Mirrors the
        # self.markers-first pattern in _stack_member_marker_mean.
        marker_time = math.nan
        if wave_name is not None and hasattr(self, 'markers'):
            marker_time = self._preview_marker_reference_time(marker_key, wave_name)
        if math.isnan(marker_time):
            marker_time = _sac_float(trace, f't{marker_key}', math.nan)
        if math.isnan(marker_time):
            return math.nan
        return float(marker_time)

    def _prepare_stack_preview_trace(self, trace, wave_name, role, stack_wave_name):
        trace.stats.dephasekit_wave_name = str(wave_name)
        trace.stats.dephasekit_stack_preview_role = str(role)
        trace.stats.dephasekit_stack_wave_name = str(stack_wave_name)
        target_fs = 1.0 / self.dt
        if abs(trace.stats.sampling_rate - target_fs) > 1e-3:
            trace.resample(target_fs, window="hann")
        self._apply_bandpass_to_trace(trace, self.bandpass_settings)
        trace.data = np.asarray(trace.data, dtype=float) * self._wave_polarity_factor(wave_name)
        return trace

    def _stack_preview_display_mode(self):
        mode = str(getattr(self, 'preview_stack_display_mode', 'overlay') or 'overlay').lower()
        if mode == 'top':
            return 'top'
        return 'overlay'

    def _stack_preview_display_button_label(self):
        return 'TopDist' if self._stack_preview_display_mode() == 'top' else 'Overlay'

    def _stack_align_marker_key(self, stack_wave_name):
        sidecar = self._stack_sidecar_for_wave(stack_wave_name)
        marker_value = sidecar.get('align_marker', '')
        marker_key = self._normalize_marker_key(marker_value)
        if marker_key is not None and str(marker_key).isdigit() and 0 <= int(marker_key) <= 9:
            return marker_key
        markers = sidecar.get('markers', {}) or {}
        candidate_marker_keys = []
        for key, value in markers.items():
            marker_key = self._normalize_marker_key(key)
            if marker_key is None or not str(marker_key).isdigit() or not (0 <= int(marker_key) <= 9):
                continue
            marker_value = _safe_float(value)
            if not math.isnan(marker_value):
                candidate_marker_keys.append(marker_key)
        for marker_key in candidate_marker_keys:
            if marker_key != '0':
                return marker_key
        if candidate_marker_keys:
            return candidate_marker_keys[0]
        return None

    def _sanitize_stack_trace_markers(self, trace, stack_wave_name):
        align_marker_key = self._stack_align_marker_key(stack_wave_name)
        if align_marker_key is None:
            return trace
        sac = _attrib_dict_from_trace(trace)
        align_value = _sac_float(trace, f't{align_marker_key}', math.nan)
        if math.isnan(align_value):
            align_value = 0.0
        for marker_index in range(10):
            setattr(sac, f't{marker_index}', math.nan)
        setattr(sac, f't{align_marker_key}', float(align_value))
        return trace

    def _stack_sidecar_relative_window(self, stack_wave_name):
        sidecar = self._stack_sidecar_for_wave(stack_wave_name)
        window = sidecar.get('window', None)
        if isinstance(window, (list, tuple)) and len(window) >= 2:
            x1 = _safe_float(window[0])
            x2 = _safe_float(window[1])
            if not math.isnan(x1) and not math.isnan(x2) and x2 > x1:
                return float(x1), float(x2)
        return None

    def _stack_trace_uses_legacy_relative_frame(self, trace, stack_wave_name, align_marker_key):
        window = self._stack_sidecar_relative_window(stack_wave_name)
        if window is None:
            return False
        align_value = _sac_float(trace, f't{align_marker_key}', math.nan)
        if math.isnan(align_value):
            return False
        tolerance = max(float(getattr(self, 'dt', 0.02) or 0.02) * 2.0, 1e-3)
        if abs(float(align_value)) > tolerance:
            return False
        b_value = _sac_float(trace, 'b', math.nan)
        e_value = _sac_float(trace, 'e', math.nan)
        if math.isnan(b_value) or math.isnan(e_value):
            return False
        if (
            abs(float(b_value) - window[0]) <= tolerance
            and abs(float(e_value) - window[1]) <= tolerance
        ):
            return True
        duration = float(e_value) - float(b_value)
        window_duration = window[1] - window[0]
        return abs(duration - window_duration) <= max(tolerance, 0.1)

    def _stack_trace_needs_sidecar_window_reanchor(self, trace, stack_wave_name, stack_reference):
        window = self._stack_sidecar_relative_window(stack_wave_name)
        if window is None:
            return False
        stack_reference = _safe_float(stack_reference)
        if math.isnan(stack_reference):
            return False
        b_value = _sac_float(trace, 'b', math.nan)
        e_value = _sac_float(trace, 'e', math.nan)
        if math.isnan(b_value) or math.isnan(e_value) or e_value <= b_value:
            return False
        tolerance = max(float(getattr(self, 'dt', 0.02) or 0.02) * 2.0, 1e-3)
        if (float(b_value) - tolerance) <= stack_reference <= (float(e_value) + tolerance):
            return False
        window_duration = float(window[1]) - float(window[0])
        trace_duration = float(e_value) - float(b_value)
        display_align_time = -float(window[0])
        duration_matches = abs(trace_duration - window_duration) <= max(tolerance, 0.25)
        display_align_inside = (
            (float(b_value) - tolerance) <= display_align_time <= (float(e_value) + tolerance)
        )
        return duration_matches or display_align_inside

    def _set_stack_trace_align_reference(self, trace, stack_wave_name, align_marker_key, align_time):
        align_time = _safe_float(align_time)
        if math.isnan(align_time):
            return trace
        window = self._stack_sidecar_relative_window(stack_wave_name)
        if window is None:
            old_align = _sac_float(trace, f't{align_marker_key}', math.nan)
            b_value = _sac_float(trace, 'b', math.nan)
            e_value = _sac_float(trace, 'e', math.nan)
            if not math.isnan(old_align) and not math.isnan(b_value) and not math.isnan(e_value):
                window = (float(b_value - old_align), float(e_value - old_align))
        sac = _attrib_dict_from_trace(trace)
        if window is not None:
            delta = _safe_float(getattr(trace.stats, 'delta', getattr(self, 'dt', math.nan)))
            npts = int(getattr(trace.stats, 'npts', len(trace.data)))
            sac.b = float(align_time + window[0])
            if not math.isnan(delta) and npts > 0:
                sac.e = float(align_time + window[0] + (npts - 1) * delta)
            else:
                sac.e = float(align_time + window[1])
        for marker_index in range(10):
            setattr(sac, f't{marker_index}', math.nan)
        setattr(sac, f't{align_marker_key}', float(align_time))
        return trace

    def _preview_stack_reference_value(self, evtdata, row_indices=None):
        reference_values = np.asarray(getattr(evtdata, 'reference_t', []), dtype=float)
        if row_indices is not None:
            valid_indices = [
                int(row_index)
                for row_index in row_indices
                if 0 <= int(row_index) < len(reference_values)
            ]
            reference_values = reference_values[valid_indices] if valid_indices else np.asarray([], dtype=float)
        finite_values = reference_values[np.isfinite(reference_values)]
        if finite_values.size == 0:
            return 0.0
        return float(np.mean(finite_values))

    def _stack_preview_window_for_wave(self, stack_wave_name):
        if not stack_wave_name:
            return None
        window = self._stack_sidecar_relative_window(stack_wave_name)
        if window is not None:
            return window
        try:
            trace = self._trace_from_runtime_dir(stack_wave_name)
        except Exception:
            return None
        x1 = _sac_float(trace, 'b', math.nan)
        x2 = _sac_float(trace, 'e', math.nan)
        if not math.isnan(x1) and not math.isnan(x2) and x2 > x1:
            align_marker_key = self._stack_align_marker_key(stack_wave_name)
            if align_marker_key is not None:
                align_value = _sac_float(trace, f't{align_marker_key}', math.nan)
                if not math.isnan(align_value):
                    x1 -= align_value
                    x2 -= align_value
            return float(x1), float(x2)
        return None

    def _apply_stack_preview_window(self, preview_index, stack_wave_name, fig=None):
        window = self._stack_preview_window_for_wave(stack_wave_name)
        if window is None:
            return None
        x1, x2 = window
        if 0 <= preview_index < len(self.preview_modes):
            self.preview_modes[preview_index][1] = x1
            self.preview_modes[preview_index][2] = x2
        if fig is not None:
            controls = getattr(fig, '_preview_controls', {})
            x1_control = controls.get('x1')
            x2_control = controls.get('x2')
            if x1_control is not None:
                x1_control.set_val(f'{x1:g}')
            if x2_control is not None:
                x2_control.set_val(f'{x2:g}')
        return window

    def _collect_stack_preview_stream(self, tmarker, stack_wave_name=None):
        stack_wave_name = stack_wave_name or self._current_stack_preview_wave_name()
        if not stack_wave_name:
            return obspy.Stream(), np.array([]), {}

        member_names = self._stack_preview_member_wave_names(stack_wave_name)
        member_traces = []
        member_references = []
        active_reference_times = {}
        for member_name in member_names:
            try:
                member_trace = self._trace_from_source_event_dir(member_name)
            except Exception as exc:
                print(f"Stack preview skipped source waveform {member_name}: {exc}")
                continue
            self._seed_member_marker_cache(member_name, member_trace)
            reference_time = self._stack_preview_reference_time(member_trace, tmarker, wave_name=member_name)
            if math.isnan(reference_time):
                continue
            member_trace = self._prepare_stack_preview_trace(
                member_trace,
                member_name,
                'member',
                stack_wave_name,
            )
            member_traces.append(member_trace)
            member_references.append(reference_time)
            active_reference_times[member_name] = float(reference_time)

        if not member_traces:
            return obspy.Stream(), np.array([]), {}

        try:
            stack_trace = self._trace_from_runtime_dir(stack_wave_name)
        except Exception as exc:
            print(f"Stack preview skipped stack waveform {stack_wave_name}: {exc}")
            return obspy.Stream(), np.array([]), {}

        align_marker_key = self._normalize_marker_key(tmarker)
        stack_reference = self._stack_preview_reference_time(stack_trace, tmarker, wave_name=stack_wave_name)
        finite_member_references = np.asarray(member_references, dtype=float)
        finite_member_references = finite_member_references[np.isfinite(finite_member_references)]
        member_reference_center = (
            float(np.mean(finite_member_references))
            if finite_member_references.size
            else math.nan
        )
        if align_marker_key is not None and self._stack_trace_uses_legacy_relative_frame(
            stack_trace,
            stack_wave_name,
            align_marker_key,
        ) and not math.isnan(member_reference_center):
            stack_trace = self._set_stack_trace_align_reference(
                stack_trace,
                stack_wave_name,
                align_marker_key,
                member_reference_center,
            )
            stack_reference = member_reference_center
        elif align_marker_key is not None:
            # Modern window-relative frame (b=0): the stacked data is aligned at
            # -x1 by construction, so the stack trace's align reference is -x1
            # regardless of any drifted value cached in self.markers or read from
            # the on-disk SAC header. Forcing this here breaks the
            # self-propagating marker offset (the same class of bug noted at the
            # _write_preview_stack_sac b/e comment) and makes a drifted stack SAC
            # display correctly. Only applies when members gave a finite align
            # center (otherwise we cannot confirm the window-relative frame).
            sidecar_window = self._stack_sidecar_relative_window(stack_wave_name)
            if sidecar_window is not None and not math.isnan(member_reference_center):
                stack_reference = -float(sidecar_window[0])
            if math.isnan(stack_reference) and not math.isnan(member_reference_center):
                stack_reference = member_reference_center
            if (
                not math.isnan(stack_reference)
                and self._stack_trace_needs_sidecar_window_reanchor(
                    stack_trace,
                    stack_wave_name,
                    stack_reference,
                )
            ):
                stack_trace = self._set_stack_trace_align_reference(
                    stack_trace,
                    stack_wave_name,
                    align_marker_key,
                    stack_reference,
                )
        if math.isnan(stack_reference):
            stack_reference = 0.0
        stack_trace = self._prepare_stack_preview_trace(
            stack_trace,
            stack_wave_name,
            'stack',
            stack_wave_name,
        )
        stack_trace = self._sanitize_stack_trace_markers(stack_trace, stack_wave_name)
        stack_trace.stats.dephasekit_stack_member_count = len(member_traces)

        member_distances = [
            _sac_float(member_trace, 'gcarc', math.nan)
            for member_trace in member_traces
        ]
        member_distances = [
            float(distance)
            for distance in member_distances
            if not math.isnan(distance)
        ]
        if member_distances and self._stack_preview_display_mode() == 'top':
            sac = _attrib_dict_from_trace(stack_trace)
            max_distance = max(member_distances)
            padding = max(1.0, abs(max_distance) * 0.03)
            sac.gcarc = max_distance + padding

        waves = obspy.Stream()
        waves += stack_trace
        t_lst = [float(stack_reference)]
        active_reference_times[str(stack_wave_name)] = float(stack_reference)
        for member_trace, reference_time in zip(member_traces, member_references):
            waves += member_trace
            t_lst.append(float(reference_time))

        self.stack_preview_active_wave_name = str(stack_wave_name)
        return waves, np.asarray(t_lst, dtype=float), active_reference_times

    def _collect_preview_display_stream(self, tmarker, fig=None, reference_times=None):
        if getattr(self, 'stack_mode', False):
            stack_wave_name = getattr(fig, '_stack_preview_wave_name', None) if fig is not None else None
            return self._collect_stack_preview_stream(tmarker, stack_wave_name=stack_wave_name)
        return self._collect_preview_stream(tmarker, reference_times=reference_times)

    def _wave_display_name(self, wave_name, default_name=''):
        wave_name = str(wave_name or '')
        default_name = str(default_name or wave_name)
        if not getattr(self, 'stack_mode', False) or not wave_name:
            return default_name
        stack_summary = self._stack_wave_summary(wave_name)
        if not stack_summary:
            return default_name
        return f"{default_name} [{stack_summary}]"

    def _set_preview_group_input_value(self, fig, group_name):
        controls = getattr(fig, '_preview_controls', {})
        widget = controls.get('group_save_widget')
        if widget is None:
            return
        normalized_name = self._normalize_preview_group_name(group_name)
        if normalized_name is None:
            return
        widget.setText(normalized_name)

    def _preview_exact_group_match(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if not isinstance(preview_state, dict):
            return None
        visible_wave_names = {
            str(meta.get('wave_name'))
            for meta in (preview_state.get('metadata', []) or [])
            if meta.get('wave_name')
        }
        if not visible_wave_names:
            return None
        for group_name in self._list_preview_group_names_for_stack():
            _normalized_name, group_wave_names, error_message = self._preview_stack_group_wave_names(group_name)
            if error_message is not None:
                continue
            if visible_wave_names == {str(wave_name) for wave_name in group_wave_names}:
                return group_name
        return None

    def _preview_stack_default_scope(self, preview_index):
        fig = getattr(self, 'plotfig', None)
        exact_group_name = self._preview_exact_group_match(fig)
        if exact_group_name is not None:
            return f'group:{exact_group_name}'
        controls = getattr(fig, '_preview_controls', {}) if fig is not None else {}
        group_widget = controls.get('group_save_widget')
        combo_widget = controls.get('group_combo_widget')
        candidates = []
        if group_widget is not None:
            try:
                candidates.append(group_widget.text())
            except Exception:
                pass
        if combo_widget is not None:
            try:
                candidates.append(combo_widget.currentData())
            except Exception:
                pass
            try:
                candidates.append(combo_widget.currentText())
            except Exception:
                pass
        available_groups = set(self._list_preview_group_names_for_stack())
        for candidate in candidates:
            normalized_name = self._normalize_preview_group_name(candidate)
            if normalized_name is None:
                continue
            if normalized_name in available_groups:
                return f'group:{normalized_name}'
        preview_state = getattr(fig, '_preview_state', {}) if fig is not None else {}
        selected_indices = preview_state.get('selected_indices', set()) if isinstance(preview_state, dict) else set()
        if selected_indices:
            return 'selected'
        return 'visible'

    def _compare_stack_summary_text(self, compare_metadata, active_wave_name=''):
        if not getattr(self, 'stack_mode', False):
            return ''
        active_wave_name = str(active_wave_name or '')
        for meta in compare_metadata or []:
            if str(meta.get('wave_name', '')) != active_wave_name:
                continue
            stack_summary = str(meta.get('stack_summary', '') or '').strip()
            if stack_summary:
                return f"Stack: {stack_summary}"
        for meta in compare_metadata or []:
            stack_summary = str(meta.get('stack_summary', '') or '').strip()
            if stack_summary:
                return f"Stack: {stack_summary}"
        return ''

    def _wave_files_for_suffix(self):
        suffix = str(self.suffix or '')
        if suffix == '':
            return []
        suffix_lower = suffix.lower()
        if getattr(self, 'stack_mode', False):
            return [str(path) for path in iter_stack_sac_paths(self.wavepath, suffix=suffix_lower)]
        matched_files = []
        for entry in os.listdir(self.wavepath):
            full_path = join(self.wavepath, entry)
            if not os.path.isfile(full_path):
                continue
            if entry.lower().endswith(suffix_lower):
                matched_files.append(full_path)
        return sorted(matched_files)

    def _runtime_wave_path(self, wave_name):
        return join(self.wavepath, wave_name)

    def _source_wave_path(self, wave_name):
        return join(self.runtime_event_dir, wave_name)

    def _mark_wave_markers_dirty(self, wave_name):
        if not wave_name:
            return
        dirty = getattr(self, 'dirty_marker_wave_names', None)
        if not isinstance(dirty, set):
            dirty = set(dirty or [])
            self.dirty_marker_wave_names = dirty
        dirty.add(str(wave_name))

    def _queue_source_marker_write(self, wave_name, header_key):
        if not getattr(self, 'stack_mode', False) or not wave_name:
            return False
        try:
            source_path = self._source_wave_path(wave_name)
        except Exception:
            return False
        if not source_path or not os.path.exists(source_path):
            return False
        header_key = str(header_key or '').strip().lower()
        if not header_key:
            return False
        pending = getattr(self, '_pending_source_marker_writes', None)
        if not isinstance(pending, dict):
            pending = {}
            self._pending_source_marker_writes = pending
        pending.setdefault(str(wave_name), set()).add(header_key)
        self._schedule_source_marker_flush()
        return True

    def _schedule_source_marker_flush(self):
        app = QApplication.instance()
        if app is None:
            return
        delay_ms = int(getattr(self, '_source_marker_flush_delay_ms', 600))
        timer = getattr(self, '_source_marker_flush_timer', None)
        if timer is None:
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(self._flush_pending_source_marker_writes)
            self._source_marker_flush_timer = timer
        self._source_marker_flush_scheduled = True
        timer.start(max(0, delay_ms))

    def _flush_pending_source_marker_writes(self, notify_review=True):
        timer = getattr(self, '_source_marker_flush_timer', None)
        if timer is not None:
            try:
                timer.stop()
            except Exception:
                pass
        self._source_marker_flush_scheduled = False
        pending = getattr(self, '_pending_source_marker_writes', {}) or {}
        if not pending:
            return []
        pending = {
            str(wave_name): set(header_keys)
            for wave_name, header_keys in dict(pending).items()
        }
        self._pending_source_marker_writes = {}
        written = []
        for wave_name, header_keys in sorted(pending.items()):
            try:
                if self._write_member_marker_snapshot_to_source(wave_name, header_keys):
                    written.append(wave_name)
            except Exception as exc:
                print(f'warn: delayed member marker write-back failed for {wave_name}: {exc}')
        if written and notify_review and callable(getattr(self, 'stack_review_refresh_callback', None)):
            try:
                self.stack_review_refresh_callback()
            except Exception:
                pass
        return written

    def _write_member_marker_snapshot_to_source(self, wave_name, header_keys=None):
        if not getattr(self, 'stack_mode', False):
            return False
        source_path = self._source_wave_path(wave_name)
        if not source_path or not os.path.exists(source_path):
            return False
        st = obspy.read(source_path)
        sac = _attrib_dict_from_trace(st[0])
        should_write = False
        header_keys = {str(key).strip().lower() for key in (header_keys or set()) if str(key).strip()}
        for marker_index in range(10):
            marker_key = str(marker_index)
            header_key = f't{marker_key}'
            if header_keys and header_key not in header_keys:
                continue
            bucket = getattr(self, 'markers', {}).get(marker_key, {})
            if wave_name not in bucket:
                continue
            value = _safe_float(bucket.get(wave_name))
            sac[header_key] = float(value) if not math.isnan(value) else math.nan
            should_write = True
        for user_key in ('user1', 'user2', 'user3', 'user4', 'user5'):
            if header_keys and user_key not in header_keys:
                continue
            bucket = getattr(self, 'user_markers', {}).get(user_key, {})
            if wave_name not in bucket:
                continue
            value = _safe_float(bucket.get(wave_name))
            sac[user_key] = float(value) if not math.isnan(value) else math.nan
            should_write = True
        if should_write:
            st.write(source_path, format='SAC')
        return should_write

    def _trace_from_runtime_dir(self, wave_name):
        tr = obspy.read(self._runtime_wave_path(wave_name))[0]
        tr.stats.dephasekit_wave_name = wave_name
        if self.stack_mode:
            self._apply_stack_sidecar_to_trace(tr, wave_name)
        return tr

    def _trace_from_loaded_wave(self, wave_name):
        wave_index = self._wave_index_by_name(wave_name)
        if wave_index is None:
            return None
        if wave_index < 0 or wave_index >= len(getattr(self, 'wave', [])):
            return None
        tr = self.wave[wave_index].copy()
        tr.stats.dephasekit_wave_name = wave_name
        return tr

    def _trace_from_source_event_dir(self, wave_name):
        tr = obspy.read(self._source_wave_path(wave_name))[0]
        tr.stats.dephasekit_wave_name = wave_name
        return tr

    def _apply_stack_sidecar_to_trace(self, trace, wave_name):
        if not self.stack_mode:
            return trace
        sac = _attrib_dict_from_trace(trace)
        sidecar = self._stack_sidecar_for_wave(wave_name)
        if not sidecar:
            return trace

        geometry = sidecar.get('geometry', {}) or {}
        event_info = sidecar.get('event', {}) or {}
        markers = sidecar.get('markers', {}) or {}

        defaults = {
            'gcarc': geometry.get('gcarc_mean', geometry.get('gcarc', 0.0)),
            'az': geometry.get('az_mean', geometry.get('az', 0.0)),
            'baz': geometry.get('baz_mean', geometry.get('baz', 0.0)),
            'evla': event_info.get('evla', 0.0),
            'evlo': event_info.get('evlo', 0.0),
            'evdp': event_info.get('evdp', 0.0),
            'nzyear': event_info.get('nzyear', 0),
            'nzjday': event_info.get('nzjday', 0),
            'nzhour': event_info.get('nzhour', 0),
            'nzmin': event_info.get('nzmin', 0),
            'nzsec': event_info.get('nzsec', 0),
        }
        for attr, value in defaults.items():
            if math.isnan(_sac_float(trace, attr, math.nan)):
                setattr(sac, attr, value)

        for marker_attr in [f't{i}' for i in range(10)]:
            marker_value = markers.get(marker_attr, math.nan)
            if marker_value is None:
                marker_value = math.nan
            setattr(sac, marker_attr, _safe_float(marker_value))
        return trace

    def _sync_stack_workspace_sac_headers_from_sidecars(self):
        if not getattr(self, 'stack_mode', False):
            return 0
        updated_count = 0
        for wave_name, sidecar in (getattr(self, 'stack_sidecars', {}) or {}).items():
            sac_path = Path(self.wavepath) / str(wave_name)
            if not sac_path.exists():
                continue
            markers = dict(sidecar.get('markers', {}) or {})
            if not markers:
                continue
            try:
                st = obspy.read(str(sac_path))
            except Exception:
                continue
            sac_headers = st[0].stats.sac
            should_write = False
            for idx in range(10):
                marker_key = f't{idx}'
                marker_value = _safe_float(markers.get(marker_key, math.nan))
                current_value = _sac_float(st[0], marker_key, math.nan)
                if math.isnan(marker_value) and math.isnan(current_value):
                    continue
                if (
                    not math.isnan(marker_value)
                    and not math.isnan(current_value)
                    and abs(float(marker_value) - float(current_value)) <= 1e-6
                ):
                    continue
                setattr(sac_headers, marker_key, marker_value)
                should_write = True
            if not should_write:
                continue
            st.write(str(sac_path), format='SAC')
            updated_count += 1
        return updated_count

    def _default_xlim_for_marker(self, marker):
        if marker in ('t0', 't7'):
            return [-10, 70]
        if marker in ('t2', 't6'):
            return [-40, 30]
        if marker in ('t3', 't5'):
            return [-50, 20]
        return [-10, 10]

    def sync_preview_window_for_marker(self, marker, xlim):
        marker_key = marker[1:] if isinstance(marker, str) and marker.startswith('t') else str(marker)
        if xlim is None or len(xlim) != 2:
            return
        synced_preview_index = None
        for preview_index, preview_mode in enumerate(self.preview_modes):
            if preview_mode[0] != marker_key:
                continue
            preview_mode[1] = float(xlim[0])
            preview_mode[2] = float(xlim[1])
            if synced_preview_index is None:
                synced_preview_index = preview_index

        if synced_preview_index is None:
            return

        if (self.plotfig is not None
                and hasattr(self.plotfig, '_preview_state')
                and self.plotfig._preview_state is not None
                and self.plotfig._preview_state.get('tmarker') == marker_key):
            self._refresh_preview_figure(self.plotfig, synced_preview_index)

        self._refresh_compare_for_preview_index(synced_preview_index)

    def _refresh_compare_for_preview_index(self, preview_index):
        if (self.comparefig is None
                or not plt.fignum_exists(self.comparefig.number)
                or not hasattr(self.comparefig, '_compare_state')
                or self.comparefig._compare_state is None):
            return
        if self.comparefig._compare_state.get('preview_index') != preview_index:
            return
        self.plot_compare_preview(preview_index)

    def init_canvas(self, order='gcarc'):
        self.init_figure(width=self.width, height=self.height, dpi=self.dpi)
        self.order = order
        self.read_sac(order=order)
        self.set_figure()
        self.set_page()
        self.init_variables()
        self.plotwave()

    def _move_file_to_bucket(self, sac_path, bucket_name):
        target_directory = self._target_directory_for_bucket(bucket_name)
        os.makedirs(target_directory, exist_ok=True)
        target_path = os.path.join(target_directory, basename(sac_path))
        shutil.move(sac_path, target_path)
        print(f"Moved file: {sac_path} -> {target_path}")

    def _sync_stack_sidecar_from_markers(self, sac_file):
        if not getattr(self, 'stack_mode', False):
            return {}
        wave_path = os.path.abspath(self.wavepath)
        stack_sac_path = join(wave_path, sac_file)
        if not os.path.exists(stack_sac_path):
            return {}
        # Enforce the window-relative frame invariant (align marker = -x1) so a
        # drifted cached align value can't propagate into the sidecar and offset
        # every auto marker on reload. The stacked data is aligned at -x1 by
        # construction, so the align marker is structural, not a free value.
        # Falls back to raw markers only when there is no sidecar window to
        # anchor against.
        relative = self._stack_window_relative_markers_for_wave(sac_file)
        if relative is not None:
            marker_payload = dict(relative[0])
        else:
            marker_payload = {
                f't{idx}': self.markers.get(str(idx), {}).get(sac_file, math.nan)
                for idx in range(10)
            }
        user_marker_payload = {
            key: self.user_markers.get(key, {}).get(sac_file, math.nan)
            for key in ('user1', 'user2', 'user3', 'user4', 'user5')
        }
        payload = update_stack_sidecar_markers(
            stack_sac_path,
            markers=marker_payload,
            user_markers=user_marker_payload,
        )
        self.stack_sidecars = load_stack_sidecar_map(wave_path)
        return payload

    def _sync_stack_package_sac_from_markers(self, sac_file):
        if not getattr(self, 'stack_mode', False):
            return False
        sidecar = self._stack_sidecar_for_wave(sac_file)
        package_dir = str(sidecar.get('result_package_dir') or '').strip()
        if not package_dir:
            return False
        package_sac_path = Path(package_dir) / 'stack.sac'
        if not package_sac_path.exists():
            return False
        try:
            st = obspy.read(str(package_sac_path))
        except Exception:
            return False
        sac_headers = st[0].stats.sac
        should_write = False
        stack_relative_markers = self._stack_window_relative_markers_for_wave(sac_file)
        if stack_relative_markers is not None:
            relative_markers, window_length = stack_relative_markers
            sac_headers['b'] = 0.0
            sac_headers['e'] = float(window_length)
            should_write = True
            for marker_key, arrival_time in relative_markers.items():
                sac_headers[marker_key] = arrival_time
        else:
            for t_idx in range(10):
                marker_key = str(t_idx)
                header_key = f't{marker_key}'
                arrival_time = self.markers.get(marker_key, {}).get(sac_file, math.nan)
                if math.isnan(arrival_time):
                    if hasattr(sac_headers, header_key):
                        sac_headers[header_key] = arrival_time
                        should_write = True
                else:
                    sac_headers[header_key] = arrival_time
                    should_write = True
        if not should_write:
            return False
        # obspy derives SAC b/e on write from stats.starttime/delta/npts
        # (b = starttime - nztime), NOT from sac.b/sac.e. Setting sac.b=0 alone
        # is silently overwritten, leaving a nonzero b that offsets every marker
        # on reload (same class of bug noted at _write_preview_stack_sac). Anchor
        # starttime to the event origin (nztime) to force b=0.
        trace = st[0]
        try:
            nztime = obspy.UTCDateTime(
                int(_safe_float(_sac_attr(trace, 'nzyear', 0))),
                1, 1, 0, 0, 0,
            ) + (int(_safe_float(_sac_attr(trace, 'nzjday', 1))) - 1) * 86400 \
              + int(_safe_float(_sac_attr(trace, 'nzhour', 0))) * 3600 \
              + int(_safe_float(_sac_attr(trace, 'nzmin', 0))) * 60 \
              + int(_safe_float(_sac_attr(trace, 'nzsec', 0)))
            trace.stats.starttime = nztime
        except Exception:
            pass
        st.write(str(package_sac_path), format='SAC')
        return True

    def _stack_window_relative_markers_for_wave(self, sac_file):
        sidecar = self._stack_sidecar_for_wave(sac_file)
        if not sidecar and getattr(self, 'stack_mode', False):
            try:
                self.stack_sidecars = load_stack_sidecar_map(os.path.abspath(self.wavepath))
            except Exception:
                self.stack_sidecars = getattr(self, 'stack_sidecars', {}) or {}
            sidecar = self._stack_sidecar_for_wave(sac_file)
        window = sidecar.get('window', None)
        if not isinstance(window, (list, tuple)) or len(window) < 2:
            return None
        x1 = _safe_float(window[0])
        x2 = _safe_float(window[1])
        align_marker_key = self._normalize_marker_key(sidecar.get('align_marker'))
        if align_marker_key is None or math.isnan(x1) or math.isnan(x2):
            return None
        align_time = _safe_float(self.markers.get(str(align_marker_key), {}).get(sac_file, math.nan))
        if math.isnan(align_time):
            return None
        sidecar_markers = dict(sidecar.get('markers', {}) or {})
        marker_origin = -float(x1)
        relative_markers = {f't{idx}': math.nan for idx in range(10)}
        relative_markers[f't{align_marker_key}'] = marker_origin
        for idx in range(10):
            marker_key = str(idx)
            if marker_key == str(align_marker_key):
                continue
            marker_time = _safe_float(self.markers.get(marker_key, {}).get(sac_file, math.nan))
            sidecar_marker_time = _safe_float(sidecar_markers.get(f't{marker_key}', math.nan))
            if math.isnan(marker_time):
                continue
            if (str(sac_file), str(marker_key)) in getattr(self, 'stack_manual_marker_keys', set()):
                if 0.0 <= float(marker_time) <= float(x2 - x1):
                    relative_markers[f't{marker_key}'] = float(marker_time)
                continue
            if (
                marker_key not in ('5', '6', '7')
                and marker_key != str(align_marker_key)
                and
                not math.isnan(sidecar_marker_time)
                and abs(float(marker_time) - float(sidecar_marker_time)) <= 1e-6
                and 0.0 <= float(sidecar_marker_time) <= float(x2 - x1)
            ):
                relative_markers[f't{marker_key}'] = float(sidecar_marker_time)
                continue
            relative_time = marker_origin + (float(marker_time) - float(align_time))
            if 0.0 <= relative_time <= float(x2 - x1):
                relative_markers[f't{marker_key}'] = relative_time
        return relative_markers, float(x2 - x1)

    def _current_preview_tick_interval(self, x1, x2):
        if self.preview_x_tick_interval_override is not None:
            return float(self.preview_x_tick_interval_override), 'manual'
        return float(auto_x_tick_interval(x2 - x1)), 'auto'

    def _target_directory_for_bucket(self, bucket_name):
        wavepath_abs = os.path.abspath(self.wavepath)
        event_name = os.path.basename(wavepath_abs)
        dataset_name = os.path.basename(os.path.dirname(wavepath_abs))
        data_root = str(PROJECT_ROOT / "data")
        if bucket_name in ("T1_sac", "LowQ_sac") and os.path.dirname(wavepath_abs).startswith(data_root):
            return os.path.join(str(PROJECT_ROOT / "data" / "process" / "low_waveform"), dataset_name, event_name)
        return os.path.join(os.getcwd(), bucket_name, event_name)

    def _analysis_output_directory(self):
        if getattr(self, 'stack_mode', False):
            return str(stack_output_dir_for_runtime(self.wavepath))
        wavepath_abs = os.path.abspath(self.wavepath)
        data_root = str(PROJECT_ROOT / "data")
        output_root = os.path.join(data_root, "output", "phases")
        if wavepath_abs.startswith(data_root + os.sep):
            relative_event_path = os.path.relpath(wavepath_abs, data_root)
            return os.path.join(output_root, relative_event_path)
        event_name = os.path.basename(wavepath_abs)
        dataset_name = os.path.basename(os.path.dirname(wavepath_abs))
        return os.path.join(output_root, dataset_name, event_name)

    def _preview_group_directory(self):
        output_root = PROJECT_ROOT / "data" / "output" / "process" / "group"
        return os.path.join(str(output_root), str(relative_event_path(self._semantic_event_dir())))

    def _normalize_preview_group_name(self, raw_name):
        text = str(raw_name or '').strip().lower()
        if text == '':
            return None
        if text.startswith('group'):
            text = text[5:]
        if text == '' or not text.isdigit():
            return None
        return f'group{int(text)}'

    def _preview_group_paths(self, group_name):
        normalized_name = self._normalize_preview_group_name(group_name)
        if normalized_name is None:
            return None, None, None
        group_dir = self._preview_group_directory()
        txt_path = os.path.join(group_dir, f'{normalized_name}.txt')
        png_path = os.path.join(group_dir, f'{normalized_name}.png')
        return normalized_name, txt_path, png_path

    def _list_preview_groups(self):
        group_dir = self._preview_group_directory()
        if not os.path.isdir(group_dir):
            return []
        group_names = []
        for entry in os.listdir(group_dir):
            group_stem, extension = os.path.splitext(entry)
            if extension and extension.lower() != '.txt':
                continue
            normalized_name = self._normalize_preview_group_name(group_stem)
            if normalized_name is None:
                continue
            txt_path = os.path.join(group_dir, f'{normalized_name}.txt')
            if os.path.isfile(txt_path):
                group_names.append(normalized_name)
        return sorted(set(group_names), key=lambda item: int(item[5:]))

    def _read_preview_group_wave_names(self, txt_path):
        wave_names = []
        if not os.path.isfile(txt_path):
            return wave_names
        with open(txt_path, 'r', encoding='utf-8') as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line == '' or line.startswith('#'):
                    continue
                wave_name = line.split('\t', 1)[0].strip()
                if wave_name:
                    wave_names.append(wave_name)
        return wave_names

    def _preview_group_wave_map(self):
        group_map = {}
        for group_name in self._list_preview_groups():
            _normalized_name, txt_path, _png_path = self._preview_group_paths(group_name)
            for wave_name in self._read_preview_group_wave_names(txt_path):
                group_map.setdefault(str(wave_name), group_name)
        return group_map

    def _preview_group_color(self, group_name):
        normalized_name = self._normalize_preview_group_name(group_name)
        if normalized_name is None:
            return '#444444'
        palette = [
            '#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd',
            '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
            '#b22222', '#006400', '#8b008b', '#ff1493', '#008b8b',
            '#b8860b', '#4169e1', '#228b22',
        ]
        try:
            index = (int(normalized_name[5:]) - 1) % len(palette)
        except ValueError:
            index = 0
        return palette[index]

    def _collect_selected_preview_entries(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return []
        metadata = preview_state.get('metadata', [])
        entries = []
        for idx in sorted(preview_state.get('selected_indices', set())):
            if idx < 0 or idx >= len(metadata):
                continue
            meta = metadata[idx]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            entries.append({
                'wave_name': wave_name,
                'label': str(meta.get('name') or wave_name),
            })
        return entries

    def _refresh_preview_group_combo(self, fig, selected_group=None):
        controls = getattr(fig, '_preview_controls', {})
        combo = controls.get('group_combo_widget')
        if combo is None:
            return
        current_text = selected_group or combo.currentText()
        group_names = self._list_preview_groups()
        previous = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem('', '')
            for group_name in group_names:
                combo.addItem(group_name, group_name)
            target_index = combo.findText(str(current_text))
            if target_index >= 0:
                combo.setCurrentIndex(target_index)
            elif combo.count() > 0:
                combo.setCurrentIndex(0)
        finally:
            combo.blockSignals(previous)

    def _save_preview_group(self, fig, preview_index, raw_group_name):
        normalized_name, txt_path, png_path = self._preview_group_paths(raw_group_name)
        if normalized_name is None:
            return False, 'Group name must be like 1 or group1'
        selected_entries = self._collect_selected_preview_entries(fig)
        if not selected_entries:
            return False, 'No selected waveforms to save'
        skipped_user1_count = 0
        group_entries = []
        for entry in selected_entries:
            if self._is_user1_wave(entry['wave_name']):
                skipped_user1_count += 1
                continue
            group_entries.append(entry)
        if not group_entries:
            return False, 'No selected non-user1 waveforms to save'
        os.makedirs(self._preview_group_directory(), exist_ok=True)
        tmarker = self.preview_modes[preview_index][0] if preview_index < len(self.preview_modes) else ''
        with open(txt_path, 'w', encoding='utf-8') as handle:
            handle.write(f'# event\t{os.path.basename(os.path.abspath(self._semantic_event_dir()))}\n')
            handle.write(f'# phase\tt{tmarker}\n')
            handle.write(f'# saved_at\t{obspy.UTCDateTime().strftime("%Y-%m-%dT%H:%M:%SZ")}\n')
            for entry in group_entries:
                handle.write(f'{entry["wave_name"]}\t{entry["label"]}\n')
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        self._refresh_preview_group_combo(fig, selected_group=normalized_name)
        skipped_summary = f'; skipped {skipped_user1_count} user1' if skipped_user1_count else ''
        return True, f'Saved {normalized_name} ({len(group_entries)} waveforms{skipped_summary})'

    def _delete_preview_group(self, fig, raw_group_name):
        normalized_name, txt_path, png_path = self._preview_group_paths(raw_group_name)
        if normalized_name is None:
            return False, 'Choose a saved group first'
        removed_paths = []
        for path in (txt_path, png_path):
            if path and os.path.isfile(path):
                os.remove(path)
                removed_paths.append(path)
        if not removed_paths:
            self._refresh_preview_group_combo(fig)
            return False, f'{normalized_name} is already missing'
        # 连带删除该 group 的 stack 文件/sidecar/结果包，避免删 group 后叠加文件
        # 残留导致厚度审阅等下游读到不一致状态（只有 stack 无 groupN.txt）。
        try:
            stack_report = delete_stack_group_configs(
                self._semantic_event_dir(), normalized_name, refresh_index=True)
            n_stack = len(stack_report.get('deleted', []))
        except Exception as exc:
            n_stack = 0
            print(f'warn: delete stack configs for {normalized_name} failed: {exc}')
        self._refresh_preview_group_combo(fig)
        msg = f'Deleted {normalized_name}'
        if n_stack:
            msg += f' (+{n_stack} stack files)'
        return True, msg

    def _restore_preview_group(self, fig, raw_group_name):
        normalized_name, txt_path, _png_path = self._preview_group_paths(raw_group_name)
        if normalized_name is None:
            return False, 'Choose a saved group first'
        wave_names = self._read_preview_group_wave_names(txt_path)
        if not wave_names:
            return False, f'{normalized_name} is empty or missing'
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return False, 'Preview state is not available'
        visible_wave_names = {
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name')
        }
        matched_wave_names = [wave_name for wave_name in wave_names if wave_name in visible_wave_names]
        if not matched_wave_names:
            return False, f'No visible waveforms matched {normalized_name}'
        self._update_preview_selection_by_wave_names(fig, matched_wave_names)
        self._apply_preview_selection(fig)
        self._set_preview_group_input_value(fig, normalized_name)
        summary = self._preview_group_availability_summary(fig, normalized_name, wave_names=wave_names)
        if summary is None:
            return True, f'Restored {normalized_name} ({len(matched_wave_names)} waveforms)'
        return True, f'Restored {normalized_name} ({len(matched_wave_names)} waveforms) | {summary}'

    def _preview_stack_output_directory(self):
        runtime_event_dir = getattr(self, 'runtime_event_dir', getattr(self, 'wavepath', ''))
        return str(stack_output_dir_for_source(runtime_event_dir))

    def _sanitize_preview_stack_label(self, raw_label):
        text = str(raw_label or '').strip().lower()
        if not text:
            return ''
        safe_chars = []
        previous_was_separator = False
        for char in text:
            if char.isalnum():
                safe_chars.append(char)
                previous_was_separator = False
            elif not previous_was_separator:
                safe_chars.append('_')
                previous_was_separator = True
        return ''.join(safe_chars).strip('_')

    def _list_preview_group_names_for_stack(self):
        return self._list_preview_groups()

    def _normalize_stack_option_signature(self, options):
        if not isinstance(options, dict):
            return None
        align_marker = self._normalize_marker_key(options.get('align_marker'))
        if align_marker is None:
            return None
        group_name = self._stack_group_name_from_scope(options.get('scope'))
        try:
            x1 = float(options.get('x1'))
            x2 = float(options.get('x2'))
        except (TypeError, ValueError):
            return None
        normalize = str(options.get('normalize', 'rms')).strip().lower()
        stack_type = str(options.get('stack_type', 'linear')).strip().lower()
        polarity = str(options.get('polarity', 'apply_user4')).strip().lower()
        moveout_mode = str(options.get('moveout_mode', 'off')).strip().lower()
        moveout_phase = self._normalize_marker_key(options.get('moveout_phase', ''))
        smatstack_max_shift_s = ''
        if stack_type == 'smatstack':
            try:
                smatstack_max_shift_s = round(float(options.get('smatstack_max_shift_s', 5.0)), 6)
            except (TypeError, ValueError):
                return None
        return (
            group_name,
            f't{align_marker}',
            round(float(x1), 6),
            round(float(x2), 6),
            polarity,
            normalize,
            stack_type,
            moveout_mode,
            '' if moveout_phase is None else f't{moveout_phase}',
            smatstack_max_shift_s,
        )

    def _stack_config_number_token(self, value):
        number = round(float(value), 6)
        sign = 'm' if number < 0 else 'p'
        text = f'{abs(number):g}'.replace('.', 'p')
        return f'{sign}{text}'

    def _stack_config_filename_tag(self, stack_inputs):
        signature = self._normalize_stack_option_signature(stack_inputs)
        if signature is None:
            return ''
        (
            group_name,
            align_marker,
            x1,
            x2,
            polarity,
            normalize,
            stack_type,
            moveout_mode,
            moveout_phase,
            smatstack_max_shift_s,
        ) = signature
        parts = []
        if group_name:
            parts.append(self._sanitize_preview_stack_label(group_name))
        else:
            scope_text = str(stack_inputs.get('scope') or 'visible').replace(':', '_')
            parts.append(self._sanitize_preview_stack_label(scope_text) or 'visible')
        parts.append(str(align_marker))
        parts.append(f'x{self._stack_config_number_token(x1)}_{self._stack_config_number_token(x2)}')
        if polarity != 'apply_user4':
            parts.append(self._sanitize_preview_stack_label(polarity) or polarity)
        if stack_type != 'linear':
            parts.append(self._sanitize_preview_stack_label(stack_type) or stack_type)
        if normalize != 'rms':
            parts.append(self._sanitize_preview_stack_label(normalize) or normalize)
        if moveout_mode != 'off':
            moveout_part = self._sanitize_preview_stack_label(moveout_mode) or moveout_mode
            if moveout_phase:
                moveout_part = f'{moveout_part}_{moveout_phase}'
            parts.append(moveout_part)
        if stack_type == 'smatstack' and smatstack_max_shift_s != '':
            parts.append(f'smax{self._stack_config_number_token(smatstack_max_shift_s)}')
        return '_'.join(part for part in parts if part)

    def _stack_saved_option_payload(self, sidecar):
        if not isinstance(sidecar, dict):
            return None
        return {
            'scope': sidecar.get('scope', 'visible'),
            'align_marker': self._normalize_marker_key(sidecar.get('align_marker')),
            'x1': sidecar.get('window', [None, None])[0] if isinstance(sidecar.get('window'), (list, tuple)) else None,
            'x2': sidecar.get('window', [None, None])[1] if isinstance(sidecar.get('window'), (list, tuple)) else None,
            'polarity': sidecar.get('polarity', 'apply_user4'),
            'normalize': sidecar.get('normalize', 'rms'),
            'stack_type': sidecar.get('stack_type', 'linear'),
            'moveout_mode': sidecar.get('moveout_mode', 'off'),
            'moveout_phase': self._normalize_marker_key(sidecar.get('moveout_phase', '')),
            'smatstack_max_shift_s': sidecar.get('smatstack_max_shift_s') if sidecar.get('smatstack_max_shift_s') is not None else 5.0,
            'label': sidecar.get('label', ''),
            'group_name': sidecar.get('group_name', ''),
            'result_package_dir': sidecar.get('result_package_dir', ''),
            'stack_wave_name': sidecar.get('stack_wave_name', ''),
        }

    def _saved_stack_options_for_group(self, raw_group_name):
        normalized_group = self._normalize_preview_group_name(raw_group_name)
        if normalized_group is None:
            return []
        if getattr(self, 'stack_mode', False):
            sidecars = getattr(self, 'stack_sidecars', {}) or {}
        else:
            sidecars = load_stack_sidecar_map(self._stack_data_event_directory())
        options_by_signature = {}
        for sidecar in sidecars.values():
            sidecar_group = str(
                sidecar.get('group_name')
                or self._stack_group_name_from_scope(sidecar.get('scope'))
                or ''
            ).strip()
            if sidecar_group.lower() != normalized_group.lower():
                continue
            payload = self._stack_saved_option_payload(sidecar)
            signature = self._normalize_stack_option_signature(payload)
            if signature is None:
                continue
            options_by_signature[signature] = payload
        return list(options_by_signature.values())

    def _saved_stack_config_default_combo_index(self, saved_options):
        return 1 if saved_options else 0

    def _preview_stack_group_wave_names(self, raw_group_name):
        normalized_name, txt_path, _png_path = self._preview_group_paths(raw_group_name)
        if normalized_name is None:
            return None, [], 'Group name must be like 1 or group1'
        wave_names = self._read_preview_group_wave_names(txt_path)
        if not wave_names:
            return normalized_name, [], f'{normalized_name} is empty or missing'
        return normalized_name, wave_names, None

    def _preview_group_availability_summary(self, fig, raw_group_name, wave_names=None):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return None
        normalized_name = self._normalize_preview_group_name(raw_group_name)
        if normalized_name is None:
            return None
        if wave_names is None:
            _normalized_name, wave_names, error_message = self._preview_stack_group_wave_names(normalized_name)
            if error_message is not None:
                return error_message
        total_group_count = len(wave_names)
        if total_group_count == 0:
            return f'{normalized_name}: empty'
        visible_wave_names = {
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name')
        }
        matched_wave_names = [wave_name for wave_name in wave_names if wave_name in visible_wave_names]
        missing_wave_names = [wave_name for wave_name in wave_names if wave_name not in visible_wave_names]
        tmarker = self._normalize_marker_key(preview_state.get('tmarker'))
        summary = f'{normalized_name}: {len(matched_wave_names)}/{total_group_count} visible'
        if not missing_wave_names:
            return summary
        marker_label = f't{tmarker}' if tmarker is not None else 'current marker'
        preview_names = ', '.join(missing_wave_names[:3])
        if len(missing_wave_names) > 3:
            preview_names += ', ...'
        return (
            f'{summary}; missing {len(missing_wave_names)} for {marker_label} '
            f'[{preview_names}]'
        )

    def _preview_stack_scope_wave_names(self, fig, scope):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return [], 'Preview state unavailable'
        metadata = preview_state.get('metadata', [])
        scope_key = str(scope or 'visible').strip().lower()
        if scope_key.startswith('group:'):
            raw_group_name = scope_key.split(':', 1)[1]
            normalized_name, group_wave_names, error_message = self._preview_stack_group_wave_names(raw_group_name)
            if error_message is not None:
                return [], error_message
            visible_wave_names = {
                meta.get('wave_name')
                for meta in metadata
                if meta.get('wave_name')
            }
            matched_wave_names = [
                wave_name for wave_name in group_wave_names
                if wave_name in visible_wave_names
            ]
            if not matched_wave_names:
                summary = self._preview_group_availability_summary(
                    fig,
                    normalized_name,
                    wave_names=group_wave_names,
                )
                return [], f'No visible waveforms matched {normalized_name}; {summary}'
            return matched_wave_names, None
        if scope_key == 'selected':
            selected_indices = sorted(preview_state.get('selected_indices', set()))
            wave_names = [
                metadata[idx].get('wave_name')
                for idx in selected_indices
                if 0 <= idx < len(metadata) and metadata[idx].get('wave_name')
            ]
            if not wave_names:
                return [], 'No selected waveforms to stack'
            return wave_names, None
        wave_names = [
            meta.get('wave_name')
            for meta in metadata
            if meta.get('wave_name')
        ]
        if not wave_names:
            return [], 'No visible waveforms to stack'
        return wave_names, None

    def _preview_stack_normalize_rows(self, data_rows, normalize_mode):
        rows = np.asarray(data_rows, dtype=float)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return rows, np.asarray([], dtype=bool), []
        normalized_rows = rows.copy()
        valid_mask = np.all(np.isfinite(normalized_rows), axis=1)
        skipped_reasons = []
        mode = str(normalize_mode or 'rms').strip().lower()
        if mode == 'off':
            return normalized_rows[valid_mask], valid_mask, skipped_reasons
        for row_index, row in enumerate(normalized_rows):
            if not valid_mask[row_index]:
                skipped_reasons.append((row_index, 'non-finite data'))
                continue
            if mode == 'peak':
                scale = float(np.max(np.abs(row)))
            else:
                scale = float(np.sqrt(np.mean(np.square(row))))
            if not np.isfinite(scale) or scale <= 0.0:
                valid_mask[row_index] = False
                skipped_reasons.append((row_index, f'zero {mode} scale'))
                continue
            normalized_rows[row_index] = row / scale
        return normalized_rows[valid_mask], valid_mask, skipped_reasons

    def _preview_stack_phase_weighting(self, normalized_rows, order=2):
        rows = np.asarray(normalized_rows, dtype=float)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return None
        analytic = hilbert(rows, axis=1)
        amplitudes = np.abs(analytic)
        unit = np.divide(
            analytic,
            amplitudes,
            out=np.zeros_like(analytic, dtype=np.complex128),
            where=amplitudes > 0,
        )
        coherence = np.abs(np.mean(unit, axis=0))
        zero_columns = np.all(amplitudes <= 1e-12, axis=0)
        coherence[zero_columns] = 1.0
        coherence = np.clip(coherence, 0.0, 1.0) ** max(1, int(order))
        return coherence

    def _preview_stack_moveout_seconds(self, wave_name, moveout_mode, align_marker, moveout_phase):
        mode = str(moveout_mode or 'off').strip().lower()
        if mode == 'off':
            return 0.0, None
        phase_key = str(moveout_phase or '').strip()
        if phase_key.startswith('t'):
            phase_key = phase_key[1:]
        if phase_key not in ('2', '3'):
            return math.nan, f'Unsupported moveout phase: t{phase_key}'
        delta_info = self._current_wave_theory_delta(
            wave_name=wave_name,
            model=self.theory_time_model,
        )
        if delta_info is None:
            return math.nan, f'No theory delta for {wave_name}'
        field_name = 'pP-P' if phase_key == '2' else 'sP-P'
        delta_value = delta_info.get(field_name, math.nan)
        if math.isnan(delta_value):
            return math.nan, f'Missing theory {field_name} for {wave_name}'
        summary = self._event_theory_delta_summary(model=self.theory_time_model)
        if summary is None:
            return math.nan, 'Theory summary unavailable'
        summary_field = f'{field_name}_mean'
        summary_value = summary.get(summary_field, math.nan)
        if math.isnan(summary_value):
            return math.nan, f'Missing mean theory {field_name}'
        return float(delta_value - summary_value), None

    def _build_preview_stack_stream(self, wave_names, apply_user4_flips=True):
        waves = obspy.Stream()
        target_fs = 1.0 / self.dt
        for wave_name in wave_names:
            tr = self._trace_from_runtime_dir(wave_name)
            if abs(tr.stats.sampling_rate - target_fs) > 1e-3:
                tr.resample(target_fs, window="hann")
            self._apply_bandpass_to_trace(tr, self.bandpass_settings)
            data = np.asarray(tr.data, dtype=float)
            if apply_user4_flips:
                data = data * self._wave_polarity_factor(wave_name)
            tr.data = data
            waves += tr
        return waves

    def _preview_stack_inputs(self, fig, preview_index, options):
        if preview_index >= len(self.preview_modes):
            return None, 'Invalid preview index'
        tmarker, default_x1, default_x2 = self.preview_modes[preview_index]
        scope = str(options.get('scope', 'visible')).lower()
        wave_names, error_message = self._preview_stack_scope_wave_names(fig, scope)
        if error_message is not None:
            return None, error_message
        align_marker = self._normalize_marker_key(options.get('align_marker', tmarker))
        if align_marker not in self.marker_styles:
            return None, f'Invalid align marker: t{align_marker}'
        try:
            x1 = float(options.get('x1', default_x1))
            x2 = float(options.get('x2', default_x2))
        except (TypeError, ValueError):
            return None, 'Stack window x1/x2 must be numeric'
        if x2 <= x1:
            return None, 'Stack x2 must be greater than x1'
        stack_type = str(options.get('stack_type', 'linear')).lower()
        smatstack_max_shift_s = 5.0
        if stack_type == 'smatstack':
            try:
                smatstack_max_shift_s = abs(float(options.get('smatstack_max_shift_s', 5.0)))
            except (TypeError, ValueError):
                return None, 'SMatStack max shift must be numeric seconds'

        polarity = str(options.get('polarity', 'apply_user4')).lower()
        if polarity == 'reject_mixed':
            has_flipped = [self._is_user4_wave(wave_name) for wave_name in wave_names]
            if any(has_flipped) and not all(has_flipped):
                return None, 'Mixed flipped/unflipped waveforms; use Apply user4 flips or split the group first'
        apply_user4_flips = polarity in ('apply_user4', 'reject_mixed')

        reference_times = None
        if align_marker == self._normalize_marker_key(tmarker):
            reference_times = self._preview_reference_times_from_figure(fig, expected_tmarker=tmarker)
        active_wave_names = []
        active_reference_times = []
        skipped_missing = []
        moveout_mode = str(options.get('moveout_mode', 'off')).strip().lower()
        moveout_phase = self._normalize_marker_key(options.get('moveout_phase', '2'))
        moveout_applied = []
        moveout_skipped = []
        for wave_name in wave_names:
            reference_time = self._preview_alignment_reference_time(
                align_marker,
                wave_name,
                reference_times=reference_times,
            )
            if math.isnan(reference_time):
                skipped_missing.append(wave_name)
                continue
            moveout_seconds, moveout_error = self._preview_stack_moveout_seconds(
                wave_name,
                moveout_mode,
                align_marker,
                moveout_phase,
            )
            if moveout_error is not None:
                if moveout_mode == 'off':
                    moveout_seconds = 0.0
                else:
                    moveout_skipped.append({'wave_name': wave_name, 'reason': moveout_error})
                    continue
            active_wave_names.append(wave_name)
            active_reference_times.append(float(reference_time) + float(moveout_seconds))
            moveout_applied.append({'wave_name': wave_name, 'seconds': float(moveout_seconds)})
        if not active_wave_names:
            return None, f'No waveforms have t{align_marker} reference times'
        waves = self._build_preview_stack_stream(active_wave_names, apply_user4_flips=apply_user4_flips)
        if len(waves) == 0:
            return None, 'No readable waveforms to stack'
        evtdata = EvtData(
            waves,
            np.asarray(active_reference_times, dtype=float),
            x1=x1,
            x2=x2,
            dt=self.dt,
            order='gcarc',
            event_name_override=self._semantic_event_name(),
        )
        return {
            'evtdata': evtdata,
            'align_marker': align_marker,
            'x1': x1,
            'x2': x2,
            'scope': scope,
            'requested_wave_names': list(wave_names),
            'active_wave_names': list(active_wave_names),
            'polarity': polarity,
            'normalize': str(options.get('normalize', 'rms')).lower(),
            'stack_type': stack_type,
            'label': self._sanitize_preview_stack_label(options.get('label', '')),
            'skipped_missing': skipped_missing,
            'apply_user4_flips': apply_user4_flips,
            'moveout_mode': moveout_mode,
            'moveout_phase': moveout_phase,
            'moveout_applied': moveout_applied,
            'moveout_skipped': moveout_skipped,
            'smatstack_max_shift_s': smatstack_max_shift_s,
        }, None

    def _compute_preview_linear_stack(self, evtdata, normalize_mode):
        normalized_rows, valid_mask, skipped_reasons = self._preview_stack_normalize_rows(
            evtdata.data,
            normalize_mode,
        )
        if normalized_rows.size == 0:
            return None, None, valid_mask, skipped_reasons
        stack_data = np.mean(normalized_rows, axis=0)
        return stack_data, normalized_rows, valid_mask, skipped_reasons

    def _compute_preview_pws_stack(self, evtdata, normalize_mode, order=2):
        normalized_rows, valid_mask, skipped_reasons = self._preview_stack_normalize_rows(
            evtdata.data,
            normalize_mode,
        )
        if normalized_rows.size == 0:
            return None, None, None, valid_mask, skipped_reasons
        linear_stack = np.mean(normalized_rows, axis=0)
        weights = self._preview_stack_phase_weighting(normalized_rows, order=order)
        if weights is None:
            return None, None, None, valid_mask, skipped_reasons
        stack_data = linear_stack * weights
        return stack_data, linear_stack, weights, valid_mask, skipped_reasons

    def _preview_stack_zero_fill_shift(self, row, shift_samples):
        data = np.asarray(row, dtype=float)
        shifted = np.zeros_like(data)
        shift = int(shift_samples)
        if shift == 0:
            return data.copy()
        if abs(shift) >= len(data):
            return shifted
        if shift > 0:
            shifted[shift:] = data[:-shift]
        else:
            shifted[:shift] = data[-shift:]
        return shifted

    def _compute_preview_smatstack_stack(self, evtdata, normalize_mode, max_shift_seconds=5.0):
        normalized_rows, valid_mask, skipped_reasons = self._preview_stack_normalize_rows(
            evtdata.data,
            normalize_mode,
        )
        if normalized_rows.size == 0:
            return None, None, None, valid_mask, skipped_reasons, {}

        dt = float(getattr(evtdata, 'dt', getattr(self, 'dt', 1.0)) or 1.0)
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0
        max_shift_seconds = abs(float(max_shift_seconds))
        max_shift_samples = int(round(max_shift_seconds / dt))
        shifts_to_try = np.arange(-max_shift_samples, max_shift_samples + 1, dtype=int)

        row_count = normalized_rows.shape[0]
        shifted_rows = np.zeros_like(normalized_rows)
        shift_samples_valid = np.zeros(row_count, dtype=int)
        match_scores_valid = np.zeros(row_count, dtype=float)

        shifted_rows[0] = normalized_rows[0]
        energy = shifted_rows[0].copy()
        match_scores_valid[0] = float(np.inner(energy, shifted_rows[0]))

        for row_index in range(1, row_count):
            row = normalized_rows[row_index]
            best_shift = 0
            best_score = -math.inf
            best_row = row.copy()
            for shift in shifts_to_try:
                candidate = self._preview_stack_zero_fill_shift(row, shift)
                score = float(np.inner(energy, candidate))
                is_better_score = score > best_score + 1e-12
                is_nearer_zero_tie = abs(score - best_score) <= 1e-12 and abs(int(shift)) < abs(best_shift)
                if is_better_score or is_nearer_zero_tie:
                    best_shift = int(shift)
                    best_score = score
                    best_row = candidate
            shifted_rows[row_index] = best_row
            shift_samples_valid[row_index] = best_shift
            match_scores_valid[row_index] = best_score
            energy = energy + best_row

        linear_stack = np.mean(normalized_rows, axis=0)
        stack_data = np.mean(shifted_rows, axis=0)
        valid_indices = np.flatnonzero(valid_mask)
        shift_samples_by_input_row = [None] * len(valid_mask)
        shift_seconds_by_input_row = [None] * len(valid_mask)
        match_scores_by_input_row = [None] * len(valid_mask)
        for local_index, input_row in enumerate(valid_indices):
            shift_samples = int(shift_samples_valid[local_index])
            shift_samples_by_input_row[int(input_row)] = shift_samples
            shift_seconds_by_input_row[int(input_row)] = float(shift_samples * dt)
            match_scores_by_input_row[int(input_row)] = float(match_scores_valid[local_index])

        smatstack_info = {
            'algorithm': 'sequential inner-product shift-match-stack with zero-fill shifts',
            'max_shift_seconds': float(max_shift_seconds),
            'max_shift_samples': int(max_shift_samples),
            'dt': float(dt),
            'reference_input_row': int(valid_indices[0]) if len(valid_indices) else None,
            'shift_samples_by_input_row': shift_samples_by_input_row,
            'shift_seconds_by_input_row': shift_seconds_by_input_row,
            'match_scores_by_input_row': match_scores_by_input_row,
        }
        return stack_data, shifted_rows, linear_stack, valid_mask, skipped_reasons, smatstack_info

    def _preview_stack_output_basename(self, stack_inputs, timestamp_tag):
        config_tag = self._stack_config_filename_tag(stack_inputs)
        if config_tag:
            return f"stack_{config_tag}_{timestamp_tag}"
        scope_text = str(stack_inputs['scope']).replace(':', '_')
        parts = [
            f"stack_t{stack_inputs['align_marker']}",
            scope_text,
            stack_inputs['stack_type'],
            stack_inputs['normalize'],
        ]
        if stack_inputs.get('label'):
            parts.append(stack_inputs['label'])
        parts.append(timestamp_tag)
        return '_'.join(parts)

    def _preview_stack_output_paths(self, stack_inputs, timestamp_tag):
        output_root = self._preview_stack_output_directory()
        basename_tag = self._preview_stack_output_basename(stack_inputs, timestamp_tag)
        package_dir = os.path.join(output_root, basename_tag)
        return {
            'output_root': output_root,
            'package_dir': package_dir,
            'basename_tag': basename_tag,
            'png': os.path.join(package_dir, 'preview.png'),
            'txt': os.path.join(package_dir, 'stack.txt'),
            'sac': os.path.join(package_dir, 'stack.sac'),
            'json': os.path.join(package_dir, 'meta.json'),
            'members': os.path.join(package_dir, 'members.txt'),
        }

    def _stack_data_event_directory(self):
        runtime_event_dir = getattr(self, 'runtime_event_dir', getattr(self, 'wavepath', ''))
        return str(stack_event_dir_for_source(runtime_event_dir))

    def _stack_wave_filename(self, basename_tag):
        if isinstance(getattr(self, '_last_stack_inputs_for_filename', None), dict):
            stack_inputs = self._last_stack_inputs_for_filename
            config_tag = self._stack_config_filename_tag(stack_inputs)
            if config_tag:
                return f'stack_{config_tag}.sac'
        label = str((basename_tag or '')).strip()
        if label and not label.startswith('stack_t'):
            return f'stack_{self._sanitize_preview_stack_label(label)}.sac'
        return 'stack.sac'

    def _stack_group_name_from_scope(self, scope_value):
        scope_text = str(scope_value or '').strip()
        if scope_text.lower().startswith('group:'):
            group_name = scope_text.split(':', 1)[1].strip()
            return group_name or ''
        return ''

    def _stack_markers_in_window_frame(self, metadata):
        x1 = _safe_float(metadata.get('x1', math.nan))
        x2 = _safe_float(metadata.get('x2', math.nan))
        marker_origin = -x1 if not math.isnan(x1) else math.nan
        stored_markers = dict(metadata.get('stack_markers', {}) or {})
        align_marker_key = self._normalize_marker_key(metadata.get('align_marker'))
        relative_markers = {f't{idx}': math.nan for idx in range(10)}
        if align_marker_key is None or math.isnan(marker_origin):
            return relative_markers
        align_header_key = f't{align_marker_key}'
        align_time = _safe_float(stored_markers.get(align_header_key, math.nan))
        if math.isnan(align_time):
            return relative_markers
        relative_markers[align_header_key] = float(marker_origin)
        for idx in range(10):
            marker_key = f't{idx}'
            if marker_key == align_header_key:
                continue
            marker_time = _safe_float(stored_markers.get(marker_key, math.nan))
            if math.isnan(marker_time):
                continue
            relative_markers[marker_key] = float(marker_origin + (marker_time - align_time))
        if not math.isnan(x2):
            for marker_key, marker_time in list(relative_markers.items()):
                if math.isnan(marker_time):
                    continue
                if marker_time < 0.0 or marker_time > float(x2 - x1):
                    relative_markers[marker_key] = math.nan
        return relative_markers

    def _write_stack_data_directory(self, output_paths, stack_inputs, metadata):
        stack_event_dir = Path(self._stack_data_event_directory())
        stack_event_dir.mkdir(parents=True, exist_ok=True)
        runtime_event_dir = getattr(self, 'runtime_event_dir', getattr(self, 'wavepath', ''))
        self._last_stack_inputs_for_filename = dict(stack_inputs or {})
        write_stack_event_marker(
            stack_event_dir,
            source_event_dir=runtime_event_dir,
            output_dir=output_paths['output_root'],
            source_event_name=self._semantic_event_name(),
        )

        stack_wave_name = self._stack_wave_filename(output_paths['basename_tag'])
        stack_wave_path = stack_event_dir / stack_wave_name
        is_valid, error_message = _validate_sac_file(output_paths['sac'])
        if not is_valid:
            raise RuntimeError(f'Generated package SAC is invalid: {error_message}')

        temp_stack_wave = stack_event_dir / f'.{stack_wave_name}.tmp'
        try:
            shutil.copy2(output_paths['sac'], temp_stack_wave)
            copied_valid, copied_error = _validate_sac_file(temp_stack_wave)
            if not copied_valid:
                raise RuntimeError(f'Copied stack SAC is invalid: {copied_error}')
            stack_markers = self._stack_markers_in_window_frame(metadata)
            if stack_markers:
                st = obspy.read(str(temp_stack_wave))
                trace = st[0]
                sac = _attrib_dict_from_trace(trace)
                x1 = _safe_float(metadata.get('x1', math.nan))
                x2 = _safe_float(metadata.get('x2', math.nan))
                if not math.isnan(x1):
                    sac.b = 0.0
                if not math.isnan(x1) and not math.isnan(x2):
                    sac.e = float(x2 - x1)
                for marker_index in range(10):
                    setattr(sac, f't{marker_index}', math.nan)
                for marker_name, marker_value in stack_markers.items():
                    normalized_marker = self._normalize_marker_key(marker_name)
                    if normalized_marker is None:
                        continue
                    marker_value = _safe_float(marker_value)
                    if math.isnan(marker_value):
                        continue
                    setattr(sac, f't{normalized_marker}', float(marker_value))
                st.write(str(temp_stack_wave), format='SAC')
                copied_valid, copied_error = _validate_sac_file(temp_stack_wave)
                if not copied_valid:
                    raise RuntimeError(f'Patched stack SAC is invalid: {copied_error}')
            group_name = self._stack_group_name_from_scope(metadata.get('scope'))
            if group_name:
                delete_stack_group_configs(
                    stack_event_dir,
                    group_name,
                    refresh_index=False,
                )
            delete_stack_config(stack_event_dir, stack_wave_name, refresh_index=False)
            os.replace(temp_stack_wave, stack_wave_path)
        finally:
            if temp_stack_wave.exists():
                temp_stack_wave.unlink()

        stack_wave_relative_name = stack_wave_name_from_path(stack_event_dir, stack_wave_path)

        sidecar_payload = {
            'mode': 'stack',
            'stack_wave_name': stack_wave_relative_name,
            'source_event_dir': str(Path(runtime_event_dir).expanduser().resolve()),
            'result_package_dir': metadata.get('result_package_dir'),
            'stack_type': metadata.get('stack_type'),
            'normalize': metadata.get('normalize'),
            'scope': metadata.get('scope'),
            'group_name': self._stack_group_name_from_scope(metadata.get('scope')),
            'align_marker': metadata.get('align_marker'),
            'polarity': metadata.get('polarity'),
            'label': metadata.get('label', ''),
            'window': [metadata.get('x1'), metadata.get('x2')],
            'wave_count_requested': metadata.get('wave_count_requested'),
            'wave_count_input': metadata.get('wave_count_input'),
            'wave_count_used': metadata.get('wave_count_used'),
            'moveout_mode': metadata.get('moveout_mode'),
            'moveout_phase': metadata.get('moveout_phase'),
            'smatstack_max_shift_s': metadata.get('smatstack_max_shift_s'),
            'smatstack': metadata.get('smatstack'),
            'geometry': {
                'gcarc_mean': metadata.get('gcarc_mean', 0.0),
                'az_mean': metadata.get('az_mean', 0.0),
                'baz_mean': metadata.get('baz_mean', 0.0),
                'pierce_lon_mean': metadata.get('pierce_lon_mean', 0.0),
                'pierce_lat_mean': metadata.get('pierce_lat_mean', 0.0),
            },
            'event': metadata.get('event_info', {}),
            'markers': {},
            'user_markers': {key: math.nan for key in ('user1', 'user2', 'user3', 'user4', 'user5')},
        }
        sidecar_payload['markers'] = self._stack_markers_in_window_frame(metadata)
        write_stack_sidecar_payload(stack_wave_path, sidecar_payload, event_dir=stack_event_dir)
        write_stack_workspace_index(stack_event_dir)
        return str(stack_wave_path)

    def _write_preview_stack_sac(self, output_path, evtdata, stack_data, x1, x2=None, align_marker='0', align_time=None):
        stack_trace = evtdata.wave_ori[0].copy()
        stack_trace.data = np.asarray(stack_data, dtype=np.float32)
        stack_trace.stats.npts = len(stack_trace.data)
        stack_trace.stats.delta = float(self.dt)
        stack_trace.stats.sampling_rate = 1.0 / float(self.dt)
        try:
            stack_trace.stats.network = 'DPK'
            stack_trace.stats.station = 'STACK'
        except Exception:
            pass
        if hasattr(stack_trace.stats, 'sac'):
            # Write the stack trace in the window-relative frame
            # (time_after_align): b=0 at the window start, the alignment marker
            # sits at -x1, and the window spans [0, x2 - x1]. This matches
            # _stack_markers_in_window_frame / _write_stack_data_directory so b/e
            # and the t-markers share one frame; otherwise the markers (relative)
            # drift relative to b/e (absolute mean(reference_t)) whenever
            # mean(reference_t) != -x1, shifting every marker by a fixed offset
            # (e.g. group2's 1.5 s).
            #
            # obspy derives SAC b/e on write from stats.starttime/delta/npts
            # (b = starttime - nztime), NOT from sac.b/sac.e. Setting sac.b=0
            # alone is silently overwritten, so we must set stats.starttime to
            # the event origin (nztime) to force b=0.
            window_length = float(x2 - x1) if x2 is not None else float((len(stack_trace.data) - 1) * self.dt)
            sac = stack_trace.stats.sac
            try:
                nztime = obspy.UTCDateTime(
                    int(_safe_float(_sac_attr(stack_trace, 'nzyear', 0))),
                    1, 1, 0, 0, 0,
                ) + (int(_safe_float(_sac_attr(stack_trace, 'nzjday', 1))) - 1) * 86400 \
                  + int(_safe_float(_sac_attr(stack_trace, 'nzhour', 0))) * 3600 \
                  + int(_safe_float(_sac_attr(stack_trace, 'nzmin', 0))) * 60 \
                  + int(_safe_float(_sac_attr(stack_trace, 'nzsec', 0)))
                stack_trace.stats.starttime = nztime
            except Exception:
                pass
            sac.b = 0.0
            sac.e = window_length
            sac.user0 = float(evtdata.sta_num)
            sac.kstnm = 'STACK'
            sac.knetwk = 'DPK'
            for marker_attr in ('t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'):
                setattr(sac, marker_attr, math.nan)
            normalized_align = self._normalize_marker_key(align_marker)
            if normalized_align is not None and normalized_align.isdigit():
                setattr(sac, f't{normalized_align}', float(-x1))
        output_path = Path(output_path)
        temp_path = output_path.with_name(f'.{output_path.name}.tmp')
        try:
            stack_trace.write(str(temp_path), format='SAC')
            is_valid, error_message = _validate_sac_file(temp_path)
            if not is_valid:
                raise RuntimeError(f'Invalid temporary stack SAC: {error_message}')
            os.replace(temp_path, output_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _stack_preview_pierce_mean(self, wave_names):
        return self._stack_preview_pierce_mean_for_phase(wave_names, self.preview_pierce_phase)

    def _stack_preview_pierce_mean_for_phase(self, wave_names, phase):
        if not wave_names:
            return math.nan, math.nan
        if not hasattr(self, 'preview_pierce_cache'):
            self.preview_pierce_cache = {}
        records = self._load_pierce_points_for_current_event(phase=phase, auto_generate=False)
        if not records:
            return math.nan, math.nan
        longitudes = []
        latitudes = []
        for wave_name in wave_names:
            record = records.get(wave_name)
            if record is None:
                continue
            longitudes.append(float(record.longitude))
            latitudes.append(float(record.latitude))
        if not longitudes or not latitudes:
            return math.nan, math.nan
        return float(np.mean(longitudes)), float(np.mean(latitudes))

    def _stack_preview_pierce_phase_for_align_marker(self, align_marker):
        normalized_align = self._normalize_marker_key(align_marker)
        if normalized_align == '5':
            return 'sP'
        if normalized_align == '6':
            return 'pP'
        return self.preview_pierce_phase

    def _save_preview_stack_outputs(
        self,
        stack_inputs,
        stack_data,
        normalized_rows,
        valid_mask,
        skipped_reasons,
        linear_stack=None,
        phase_weights=None,
        stack_extra=None,
    ):
        evtdata = stack_inputs['evtdata']
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_paths = self._preview_stack_output_paths(stack_inputs, timestamp_tag)
        os.makedirs(output_paths['package_dir'], exist_ok=True)
        png_path = output_paths['png']
        txt_path = output_paths['txt']
        sac_path = output_paths['sac']
        json_path = output_paths['json']
        members_path = output_paths['members']

        try:
            time_axis = np.asarray(evtdata.time_axis, dtype=float)
            valid_indices = np.flatnonzero(valid_mask)
            stack_align_time = self._preview_stack_reference_value(evtdata, valid_indices)
            np.savetxt(
                txt_path,
                np.column_stack([time_axis, np.asarray(stack_data, dtype=float)]),
                fmt='%.8g',
                header='time_after_align_s stack_amplitude',
            )
            self._write_preview_stack_sac(
                sac_path,
                evtdata,
                stack_data,
                stack_inputs['x1'],
                x2=stack_inputs['x2'],
                align_marker=stack_inputs['align_marker'],
                align_time=stack_align_time,
            )
            valid_wave_names = [
                getattr(evtdata.wave_ori[idx].stats, 'dephasekit_wave_name', '')
                for idx in valid_indices
                if idx < len(evtdata.wave_ori)
            ]
            skipped_zero = [
                {
                    'wave_name': getattr(evtdata.wave_ori[row_index].stats, 'dephasekit_wave_name', ''),
                    'reason': reason,
                }
                for row_index, reason in skipped_reasons
                if row_index < len(evtdata.wave_ori)
            ]
            extra_metadata = dict(stack_extra or {})
            smatstack_info = extra_metadata.get('smatstack') if isinstance(extra_metadata, dict) else None
            smatstack_shift_lookup = {}
            if isinstance(smatstack_info, dict):
                row_shifts = smatstack_info.get('shift_samples_by_input_row', []) or []
                row_shift_seconds = smatstack_info.get('shift_seconds_by_input_row', []) or []
                row_scores = smatstack_info.get('match_scores_by_input_row', []) or []
                shift_records = []
                for row_index, trace in enumerate(evtdata.wave_ori):
                    if row_index >= len(row_shifts) or row_shifts[row_index] is None:
                        continue
                    wave_name = getattr(trace.stats, 'dephasekit_wave_name', '')
                    record = {
                        'wave_name': wave_name,
                        'input_row': int(row_index),
                        'shift_samples': int(row_shifts[row_index]),
                        'shift_seconds': float(row_shift_seconds[row_index]),
                        'match_score': float(row_scores[row_index]) if row_index < len(row_scores) and row_scores[row_index] is not None else None,
                    }
                    shift_records.append(record)
                    smatstack_shift_lookup[wave_name] = record
                smatstack_info['shift_records'] = shift_records
            stack_pierce_phase = self._stack_preview_pierce_phase_for_align_marker(stack_inputs['align_marker'])
            pierce_lon_mean, pierce_lat_mean = self._stack_preview_pierce_mean_for_phase(valid_wave_names, stack_pierce_phase)
            gcarc_mean = float(np.mean(evtdata.gcarc)) if len(evtdata.gcarc) else 0.0
            az_mean = float(np.mean(evtdata.az)) if len(evtdata.az) else 0.0
            baz_mean = float(np.mean(evtdata.baz)) if len(evtdata.baz) else 0.0
            first_trace = evtdata.wave_ori[0] if len(evtdata.wave_ori) else None
            event_info = {
                'nzyear': int(_safe_float(_sac_attr(first_trace, 'nzyear', 0))) if first_trace is not None else 0,
                'nzjday': int(_safe_float(_sac_attr(first_trace, 'nzjday', 0))) if first_trace is not None else 0,
                'nzhour': int(_safe_float(_sac_attr(first_trace, 'nzhour', 0))) if first_trace is not None else 0,
                'nzmin': int(_safe_float(_sac_attr(first_trace, 'nzmin', 0))) if first_trace is not None else 0,
                'nzsec': int(_safe_float(_sac_attr(first_trace, 'nzsec', 0))) if first_trace is not None else 0,
                'evla': _sac_float(first_trace, 'evla', 0.0) if first_trace is not None else 0.0,
                'evlo': _sac_float(first_trace, 'evlo', 0.0) if first_trace is not None else 0.0,
                'evdp': _sac_float(first_trace, 'evdp', 0.0) if first_trace is not None else 0.0,
            }
            stack_markers = {f't{i}': math.nan for i in range(10)}
            stack_markers[f"t{stack_inputs['align_marker']}"] = stack_align_time
            for marker_key in self._stack_auxiliary_marker_keys(stack_inputs['align_marker']):
                stack_markers[f't{marker_key}'] = self._stack_member_marker_mean(
                    valid_wave_names,
                    marker_key,
                    evtdata=evtdata,
                )
            member_status_lines = ['wave_name\tstatus\tdetail']
            for wave_name in stack_inputs.get('requested_wave_names', []):
                status = 'used'
                detail = ''
                if wave_name in stack_inputs.get('skipped_missing', []):
                    status = 'skipped_missing_reference'
                    detail = f"t{stack_inputs['align_marker']}"
                else:
                    moveout_skip = next(
                        (
                            item for item in stack_inputs.get('moveout_skipped', [])
                            if item.get('wave_name') == wave_name
                        ),
                        None,
                    )
                    if moveout_skip is not None:
                        status = 'skipped_moveout'
                        detail = moveout_skip.get('reason', '')
                    else:
                        normalization_skip = next(
                            (
                                item for item in skipped_zero
                                if item.get('wave_name') == wave_name
                            ),
                            None,
                        )
                        if normalization_skip is not None:
                            status = 'skipped_normalization'
                            detail = normalization_skip.get('reason', '')
                        elif wave_name not in valid_wave_names:
                            status = 'aligned_not_used'
                if status == 'used' and wave_name in smatstack_shift_lookup:
                    shift_record = smatstack_shift_lookup[wave_name]
                    detail = (
                        f"shift={shift_record['shift_seconds']:.6g}s "
                        f"({shift_record['shift_samples']} samples)"
                    )
                member_status_lines.append(f'{wave_name}\t{status}\t{detail}')
            with open(members_path, 'w', encoding='utf-8') as handle:
                handle.write('\n'.join(member_status_lines) + '\n')

            metadata = {
                'event': os.path.basename(os.path.abspath(self._semantic_event_dir())),
                'event_path': os.path.abspath(self._semantic_event_dir()),
                'created_at': obspy.UTCDateTime().strftime("%Y-%m-%dT%H:%M:%SZ"),
                'scope': stack_inputs['scope'],
                'align_marker': f"t{stack_inputs['align_marker']}",
                'x1': stack_inputs['x1'],
                'x2': stack_inputs['x2'],
                'normalize': stack_inputs['normalize'],
                'stack_type': stack_inputs['stack_type'],
                'polarity': stack_inputs['polarity'],
                'apply_user4_flips': stack_inputs['apply_user4_flips'],
                'wave_count_requested': int(len(stack_inputs.get('requested_wave_names', []))),
                'wave_count_input': int(evtdata.sta_num),
                'wave_count_used': int(len(valid_wave_names)),
                'wave_names_requested': stack_inputs.get('requested_wave_names', []),
                'wave_names_aligned': stack_inputs.get('active_wave_names', []),
                'wave_names_used': valid_wave_names,
                'skipped_missing_reference': stack_inputs.get('skipped_missing', []),
                'skipped_normalization': skipped_zero,
                'moveout_mode': stack_inputs.get('moveout_mode', 'off'),
                'moveout_phase': f"t{stack_inputs.get('moveout_phase', '')}" if stack_inputs.get('moveout_mode', 'off') != 'off' else None,
                'moveout_applied': stack_inputs.get('moveout_applied', []),
                'moveout_skipped': stack_inputs.get('moveout_skipped', []),
                'smatstack_max_shift_s': stack_inputs.get('smatstack_max_shift_s'),
                'bandpass': self._current_bandpass_profile(),
                'preview_phase': f"t{stack_inputs['align_marker']}",
                'preview_pierce_phase': stack_pierce_phase,
                'preview_pierce_model': self.preview_pierce_model,
                'gcarc_mean': gcarc_mean,
                'az_mean': az_mean,
                'baz_mean': baz_mean,
                'pierce_lon_mean': pierce_lon_mean,
                'pierce_lat_mean': pierce_lat_mean,
                'event_info': event_info,
                'stack_markers': stack_markers,
                'result_package_dir': output_paths['package_dir'],
                'outputs': {
                    'png': png_path,
                    'txt': txt_path,
                    'sac': sac_path,
                    'json': json_path,
                    'members': members_path,
                },
            }
            metadata.update(extra_metadata)
            with open(json_path, 'w', encoding='utf-8') as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
            metadata['stack_data_sac'] = self._write_stack_data_directory(output_paths, stack_inputs, metadata)

            fig, ax = plt.subplots(figsize=(9.5, 5.4))
            if normalized_rows is not None and len(normalized_rows):
                alpha = max(0.08, min(0.28, 8.0 / max(len(normalized_rows), 1)))
                for row in normalized_rows:
                    ax.plot(time_axis, row, color='#9aa0a6', linewidth=0.45, alpha=alpha)
            if linear_stack is not None:
                ax.plot(time_axis, linear_stack, color='#5b7c99', linewidth=1.0, alpha=0.85, label='Linear base')
            stack_label = {
                'linear': 'Linear stack',
                'pws': 'PWS stack',
                'smatstack': 'SMatStack',
            }.get(str(stack_inputs.get('stack_type', 'linear')).lower(), 'Stack')
            ax.plot(time_axis, stack_data, color='#c62828', linewidth=1.8, label=stack_label)
            ax.axvline(0.0, color='black', linewidth=0.8)
            ax.grid(color='#cccccc', linestyle='--', linewidth=0.45, alpha=0.7)
            ax.set_xlim(stack_inputs['x1'], stack_inputs['x2'])
            ax.set_xlabel(f"Time after t{stack_inputs['align_marker']} (s)")
            ax.set_ylabel('Normalized amplitude')
            title = (
                f"{metadata['event']} | t{stack_inputs['align_marker']} | "
                f"{stack_inputs['scope']} | {stack_inputs['stack_type']} | "
                f"{stack_inputs['normalize']} | N={len(valid_wave_names)}"
            )
            ax.set_title(title, fontsize=11)
            ax.legend(loc='upper right')
            if phase_weights is not None:
                ax2 = ax.twinx()
                ax2.plot(time_axis, phase_weights, color='#2e7d32', linewidth=0.9, alpha=0.75, label='PWS weight')
                ax2.set_ylabel('Phase coherence')
                ax2.set_ylim(0.0, 1.05)
            fig.tight_layout()
            fig.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            return metadata
        except Exception:
            shutil.rmtree(output_paths['package_dir'], ignore_errors=True)
            raise

    def _stack_member_marker_mean(self, wave_names, marker_key, evtdata=None):
        normalized_marker = self._normalize_marker_key(marker_key)
        if normalized_marker is None:
            return math.nan
        trace_by_wave_name = {}
        for trace in getattr(evtdata, 'wave_ori', []) if evtdata is not None else []:
            wave_name = getattr(trace.stats, 'dephasekit_wave_name', '')
            if wave_name:
                trace_by_wave_name[str(wave_name)] = trace
        values = []
        for wave_name in wave_names or []:
            marker_time = math.nan
            if hasattr(self, 'markers'):
                marker_time = self._preview_marker_reference_time(normalized_marker, wave_name)
            if math.isnan(marker_time):
                trace = trace_by_wave_name.get(str(wave_name))
                if trace is not None:
                    marker_time = _sac_float(trace, f't{normalized_marker}', math.nan)
            if math.isnan(marker_time):
                continue
            values.append(float(marker_time))
        if not values:
            return math.nan
        return float(np.mean(values))

    def _stack_auxiliary_marker_keys(self, align_marker_key):
        normalized_align = self._normalize_marker_key(align_marker_key)
        marker_keys = []
        if normalized_align != '7':
            marker_keys.append('7')
        if normalized_align == '5':
            marker_keys.append('6')
        elif normalized_align == '6':
            marker_keys.append('5')
        return marker_keys

    def _run_preview_stack(self, fig, preview_index, options):
        stack_inputs, error_message = self._preview_stack_inputs(fig, preview_index, options)
        if error_message is not None:
            return False, error_message, None
        linear_stack = None
        phase_weights = None
        stack_extra = None
        if stack_inputs['stack_type'] == 'pws':
            stack_data, linear_stack, phase_weights, valid_mask, skipped_reasons = self._compute_preview_pws_stack(
                stack_inputs['evtdata'],
                stack_inputs['normalize'],
            )
            if stack_data is not None:
                normalized_rows, _valid_mask2, _skipped2 = self._preview_stack_normalize_rows(
                    stack_inputs['evtdata'].data,
                    stack_inputs['normalize'],
                )
        elif stack_inputs['stack_type'] == 'smatstack':
            stack_data, normalized_rows, linear_stack, valid_mask, skipped_reasons, smatstack_info = self._compute_preview_smatstack_stack(
                stack_inputs['evtdata'],
                stack_inputs['normalize'],
                max_shift_seconds=stack_inputs.get('smatstack_max_shift_s', 5.0),
            )
            stack_extra = {'smatstack': smatstack_info}
        else:
            stack_data, normalized_rows, valid_mask, skipped_reasons = self._compute_preview_linear_stack(
                stack_inputs['evtdata'],
                stack_inputs['normalize'],
            )
        if stack_data is None:
            return False, 'No valid waveforms remained after normalization', None
        metadata = self._save_preview_stack_outputs(
            stack_inputs,
            stack_data,
            normalized_rows,
            valid_mask,
            skipped_reasons,
            linear_stack=linear_stack,
            phase_weights=phase_weights,
            stack_extra=stack_extra,
        )
        if not getattr(self, 'stack_mode', False):
            self.stack_sidecars = load_stack_sidecar_map(self._stack_data_event_directory())
        saved_stack_name = os.path.basename(str(metadata.get('stack_data_sac', '') or ''))
        message = (
            f"Saved Stack {saved_stack_name or 'stack.sac'} "
            f"N={metadata['wave_count_used']} "
            f"({metadata['normalize']}/{metadata['stack_type']})"
        )
        return True, message, metadata

    def _theory_summary_output_dir(self):
        # Theory-time summaries are computed from the source event's SAC files and
        # cached under data/output/phases/<source_event> (see ensure_event_theory_summary
        # in ppk.py). In stack mode the generic analysis output dir points at the stack
        # workspace, which never holds this summary, so derive the path from the source
        # event dir (runtime_event_dir) using the same logic as the non-stack branch of
        # _analysis_output_directory, staying consistent with where ppk writes it.
        if getattr(self, 'stack_mode', False):
            event_dir_abs = os.path.abspath(str(self.runtime_event_dir))
            data_root = str(PROJECT_ROOT / "data")
            output_root = os.path.join(data_root, "output", "phases")
            if event_dir_abs.startswith(data_root + os.sep):
                relative_event_path = os.path.relpath(event_dir_abs, data_root)
                return os.path.join(output_root, relative_event_path)
            event_name = os.path.basename(event_dir_abs)
            dataset_name = os.path.basename(os.path.dirname(event_dir_abs))
            return os.path.join(output_root, dataset_name, event_name)
        return self._analysis_output_directory()

    def _theory_summary_cache_path(self, model=None):
        model_key = str(model or self.theory_time_model).lower()
        return os.path.join(
            self._theory_summary_output_dir(),
            f'theory_time_summary_{model_key}.json',
        )

    def _pierce_output_directory(self):
        return os.path.dirname(
            str(
                pierce_file_path(
                    event_dir=self.runtime_event_dir,
                    phase=self.preview_pierce_phase,
                    model=self.preview_pierce_model,
                    pierce_depth_km=self.preview_pierce_depth_km,
                    output_root=self.preview_pierce_output_root,
                )
            )
        )

    def _pierce_depth_for_model(self, model=None):
        model_key = str(model or self.preview_pierce_model).strip().lower()
        if model_key == 'iasp91':
            return 35.0
        return DEFAULT_PIERCE_DEPTH_KM

    def _ensure_event_pierce_cache(self):
        if self.preview_pierce_generation_attempted:
            return
        self.preview_pierce_generation_attempted = True
        try:
            ensure_event_pierce_files(
                event_dir=self.runtime_event_dir,
                phases=('pP', 'sP'),
                models=('prem',),
                pierce_depth_km=self._pierce_depth_for_model('prem'),
                output_root=self.preview_pierce_output_root,
                taup_bin=self.preview_pierce_taup_bin,
            )
            ensure_event_pierce_files(
                event_dir=self.runtime_event_dir,
                phases=('pP', 'sP'),
                models=('iasp91',),
                pierce_depth_km=self._pierce_depth_for_model('iasp91'),
                output_root=self.preview_pierce_output_root,
                taup_bin=self.preview_pierce_taup_bin,
            )
        except Exception as exc:
            print(f"Pierce-point cache generation skipped: {exc}")

    def _ensure_current_preview_pierce_cache(self, phase=None, model=None):
        phase_key, model_key = self._pierce_cache_key(phase=phase, model=model)
        try:
            ensure_pierce_file(
                event_dir=self.runtime_event_dir,
                phase=phase_key,
                model=model_key,
                pierce_depth_km=self._pierce_depth_for_model(model_key),
                output_root=self.preview_pierce_output_root,
                taup_bin=self.preview_pierce_taup_bin,
            )
        except Exception as exc:
            print(f"Current pierce cache generation skipped: {exc}")

    def _pierce_cache_key(self, phase=None, model=None):
        return (
            str(phase or self.preview_pierce_phase),
            str(model or self.preview_pierce_model).lower(),
        )

    def _load_pierce_points_for_current_event(self, phase=None, model=None, auto_generate=False):
        phase_key, model_key = self._pierce_cache_key(phase=phase, model=model)
        cache_key = (phase_key, model_key)
        if cache_key in self.preview_pierce_cache:
            return self.preview_pierce_cache[cache_key]
        file_path = pierce_file_path(
            event_dir=self.runtime_event_dir,
            phase=phase_key,
            model=model_key,
            pierce_depth_km=self._pierce_depth_for_model(model_key),
            output_root=self.preview_pierce_output_root,
        )
        if not os.path.exists(file_path) and auto_generate:
            self._ensure_event_pierce_cache()
        if not os.path.exists(file_path):
            return {}
        records = load_pierce_points(file_path)
        self.preview_pierce_cache[cache_key] = records
        return records

    def _pierce_file_path_for_current_event(self, phase=None, model=None):
        phase_key, model_key = self._pierce_cache_key(phase=phase, model=model)
        return pierce_file_path(
            event_dir=self.runtime_event_dir,
            phase=phase_key,
            model=model_key,
            pierce_depth_km=self._pierce_depth_for_model(model_key),
            output_root=self.preview_pierce_output_root,
        )

    def _stack_sidecar_pierce_record(self, wave_name):
        if not self.stack_mode:
            return None
        sidecar = self._stack_sidecar_for_wave(wave_name)
        geometry = sidecar.get('geometry', {}) or {}
        longitude = _safe_float(geometry.get('pierce_lon_mean', math.nan))
        latitude = _safe_float(geometry.get('pierce_lat_mean', math.nan))
        if math.isnan(longitude) or math.isnan(latitude):
            return None
        return PiercePointRecord(
            wave_name=str(wave_name),
            longitude=float(longitude),
            latitude=float(latitude),
        )

    def _stack_preview_pierce_member_record_name(self, wave_name):
        return f'member::{wave_name}'

    def _is_stack_preview_member_pierce_name(self, wave_name):
        return str(wave_name or '').startswith('member::')

    def _stack_preview_pierce_points(self, active_stack_wave_name=None, include_members=None):
        active_stack_wave_name = active_stack_wave_name or self._current_stack_preview_wave_name()
        if include_members is None:
            include_members = bool(getattr(self, 'preview_stack_show_member_pierce', False))
        # No active preview: fall back to all stack groups' mean points.
        if not active_stack_wave_name:
            records = []
            for wave_name in self._stack_preview_stack_wave_names():
                record = self._stack_sidecar_pierce_record(wave_name)
                if record is not None:
                    records.append(record)
            return records
        # Active preview: show only the current group so the pierce panel tracks
        # the preview instead of displaying a fixed set of all groups' means.
        records = []
        if include_members:
            source_records = self._load_pierce_points_for_current_event(auto_generate=False)
            for member_name in self._stack_preview_member_wave_names(active_stack_wave_name):
                source_record = source_records.get(member_name)
                if source_record is None:
                    continue
                records.append(PiercePointRecord(
                    wave_name=self._stack_preview_pierce_member_record_name(member_name),
                    longitude=float(source_record.longitude),
                    latitude=float(source_record.latitude),
                ))
        group_record = self._stack_sidecar_pierce_record(active_stack_wave_name)
        if group_record is not None:
            records.append(group_record)
        return records


    def _visible_preview_wave_names(self, preview_state):
        if preview_state is None:
            return []
        return [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name')
        ]

    def _preview_pierce_points(self, preview_state, selected_only=False):
        records = self._load_pierce_points_for_current_event(auto_generate=False)
        group_map = self._preview_group_wave_map()
        metadata = preview_state.get('metadata', []) if preview_state is not None else []
        if selected_only and preview_state is not None:
            selected_indices = {
                idx for idx in preview_state.get('selected_indices', set())
                if 0 <= idx < len(metadata)
            }
            wave_names = [
                metadata[idx].get('wave_name')
                for idx in sorted(selected_indices)
                if metadata[idx].get('wave_name')
            ]
        else:
            wave_names = [
                meta.get('wave_name')
                for meta in metadata
                if meta.get('wave_name')
            ]
        points = []
        for wave_name in wave_names:
            record = records.get(wave_name)
            if record is None:
                record = self._stack_sidecar_pierce_record(wave_name)
            if record is not None:
                record_group_name = group_map.get(str(wave_name), '')
                try:
                    setattr(record, 'group_name', record_group_name)
                except Exception:
                    pass
                points.append(record)
        return points

    def _preview_pierce_status_message(self, metadata, pierce_records):
        if getattr(self, 'stack_mode', False):
            if pierce_records:
                return ''
            return 'No stack pierce points in sidecar metadata'
        file_path = self._pierce_file_path_for_current_event()
        phase_key, model_key = self._pierce_cache_key()
        depth_km = self._pierce_depth_for_model(model_key)
        if not os.path.exists(file_path):
            return (
                f'No pierce cache for {phase_key} {model_key}\n'
                f'{depth_km:.1f} km file is missing'
            )
        visible_wave_names = [
            meta.get('wave_name')
            for meta in (metadata or [])
            if meta.get('wave_name')
        ]
        if not visible_wave_names:
            return f'No visible waveforms for {phase_key} {model_key}'
        if pierce_records:
            return ''
        return (
            f'0/{len(visible_wave_names)} visible waveforms matched\n'
            f'{phase_key} {model_key} pierce cache'
        )

    def _maybe_generate_current_preview_pierce_cache(self, metadata):
        visible_count = len([
            meta.get('wave_name')
            for meta in (metadata or [])
            if meta.get('wave_name')
        ])
        if visible_count == 0:
            return False
        if visible_count > self.preview_pierce_autogen_threshold:
            return False
        file_path = self._pierce_file_path_for_current_event()
        if os.path.exists(file_path):
            return False
        self._ensure_current_preview_pierce_cache()
        return os.path.exists(file_path)

    def _preview_should_defer_side_panels(self, wave_count):
        try:
            count = int(wave_count)
        except (TypeError, ValueError):
            return False
        return count > int(self.preview_deferred_panel_threshold)

    def _preview_should_async_pierce_panel(self, point_count):
        try:
            count = int(point_count)
        except (TypeError, ValueError):
            return False
        return count > int(getattr(self, 'preview_pierce_async_threshold', 140))

    def _draw_preview_deferred_panel(self, ax, title, message, xlabel=None, ylabel=None):
        ax.cla()
        ax.grid(False)
        if title:
            ax.set_title(title, fontsize=10)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=9)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            message,
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=9.5,
            color='#666666',
            wrap=True,
        )
        for spine in ax.spines.values():
            spine.set_alpha(0.35)

    def _pierce_record_style(self, wave_name, selected=False):
        if self._is_stack_preview_member_pierce_name(wave_name):
            return '#6f8794', '#00a6d6' if selected else '#6f8794'
        if self._is_user1_wave(wave_name):
            return self.user1_mark_color, self.user1_selected_color if selected else self.user1_mark_color
        if self._is_user4_wave(wave_name):
            return self.user4_mark_color, self.user4_selected_color if selected else self.user4_mark_color
        if self._is_preview_purple_wave(wave_name):
            return self.preview_mark_color, self.preview_selected_mark_color if selected else self.preview_mark_color
        if self._is_user5_wave(wave_name):
            return self.user5_mark_color, self.user5_selected_color if selected else self.user5_mark_color
        return '#4f81bd', '#ff375f' if selected else '#4f81bd'

    def _group_number_from_wave_name(self, wave_name):
        text = str(wave_name or '')
        for part in text.replace('\\', '/').split('/'):
            normalized_name = self._normalize_preview_group_name(part)
            if normalized_name is None:
                match = re.search(r'(?:^|[^a-z0-9])(?:stack_)?group(\d+)(?:[^a-z0-9]|$)', part, flags=re.IGNORECASE)
                if match is None:
                    continue
                normalized_name = f'group{int(match.group(1))}'
            try:
                return int(normalized_name[5:])
            except (TypeError, ValueError):
                return None
        return None

    def _group_number_from_record(self, record):
        group_name = getattr(record, 'group_name', '')
        normalized_name = self._normalize_preview_group_name(group_name)
        if normalized_name is not None:
            try:
                return int(normalized_name[5:])
            except (TypeError, ValueError):
                return None
        return self._group_number_from_wave_name(getattr(record, 'wave_name', ''))

    def _preview_group_label_positions(self, grouped_points):
        positions = {}
        for group_number, points in sorted((grouped_points or {}).items()):
            longitudes = [point[0] for point in points]
            latitudes = [point[1] for point in points]
            if not longitudes or not latitudes:
                continue
            max_lon = float(np.max(longitudes))
            min_lat = float(np.min(latitudes))
            lon_span = max(float(np.max(longitudes) - np.min(longitudes)), 0.0)
            lat_span = max(float(np.max(latitudes) - np.min(latitudes)), 0.0)
            positions[group_number] = (
                max_lon + max(0.012, lon_span * 0.18 + 0.01),
                min_lat - max(0.008, lat_span * 0.18 + 0.008),
            )
        return positions

    def _draw_preview_group_overlay(self, axp, pierce_records):
        artists = []
        if not bool(getattr(self, 'preview_group_overlay_enabled', False)):
            return artists
        grouped_points = {}
        label_artists = {}
        for record in pierce_records or []:
            group_number = self._group_number_from_record(record)
            if group_number is None:
                continue
            grouped_points.setdefault(group_number, []).append(
                (float(record.longitude), float(record.latitude))
            )
        for group_number, points in sorted(grouped_points.items()):
            group_color = self._preview_group_color(f'group{group_number}')
            longitudes = [point[0] for point in points]
            latitudes = [point[1] for point in points]
            artists.append(
                axp.scatter(
                    longitudes,
                    latitudes,
                    s=30,
                    c=[group_color],
                    alpha=0.96,
                    edgecolors='white',
                    linewidths=0.45,
                    zorder=4,
                )
            )
        sorted_groups = sorted(grouped_points)
        for row_index, group_number in enumerate(sorted_groups):
            group_color = self._preview_group_color(f'group{group_number}')
            text_artist = axp.text(
                    0.98,
                    0.04 + 0.055 * row_index,
                    str(group_number),
                    fontsize=8.8,
                    color=group_color,
                    ha='right',
                    va='bottom',
                    fontweight='bold',
                    transform=axp.transAxes,
                    bbox={
                        'boxstyle': 'round,pad=0.18',
                        'facecolor': 'white',
                        'edgecolor': group_color,
                        'linewidth': 0.9,
                        'alpha': 0.92,
                    },
                    zorder=6,
                )
            artists.append(text_artist)
            label_artists[group_number] = text_artist
        setattr(axp, '_dephasekit_group_label_artists', label_artists)
        return artists

    def _draw_preview_pierce_panel(self, axp, evtdata, pierce_records):
        axp.cla()
        event_lon = float(evtdata.evlo)
        event_lat = float(evtdata.evla)
        axp.set_title(f'{self.preview_pierce_phase} {self.preview_pierce_model}', fontsize=10)
        axp.set_xlabel('Lon', fontsize=9)
        axp.set_ylabel('Lat', fontsize=9)

        all_lons = [event_lon]
        all_lats = [event_lat]
        if pierce_records:
            all_lons.extend(record.longitude for record in pierce_records)
            all_lats.extend(record.latitude for record in pierce_records)

        if self.preview_pierce_range_locked and self.preview_pierce_fixed_bounds is not None:
            lon_min, lon_max, lat_min, lat_max = self.preview_pierce_fixed_bounds
        else:
            lon_min = min(all_lons)
            lon_max = max(all_lons)
            lat_min = min(all_lats)
            lat_max = max(all_lats)
            lon_pad = max(0.5, (lon_max - lon_min) * 0.18 if lon_max > lon_min else 0.8)
            lat_pad = max(0.5, (lat_max - lat_min) * 0.18 if lat_max > lat_min else 0.8)
            lon_min = lon_min - lon_pad
            lon_max = lon_max + lon_pad
            lat_min = lat_min - lat_pad
            lat_max = lat_max + lat_pad
        axp.set_xlim(lon_min, lon_max)
        axp.set_ylim(lat_min, lat_max)

        base_scatter = None
        highlight_scatter = None
        if pierce_records:
            longitudes = np.asarray([record.longitude for record in pierce_records], dtype=float)
            latitudes = np.asarray([record.latitude for record in pierce_records], dtype=float)
            base_colors = [self._pierce_record_style(record.wave_name, selected=False)[0] for record in pierce_records]
            base_scatter = axp.scatter(
                longitudes,
                latitudes,
                s=18,
                c=base_colors,
                alpha=0.85,
                edgecolors='white',
                linewidths=0.35,
                zorder=3,
            )
            highlight_scatter = axp.scatter(
                [],
                [],
                s=40,
                c='#ff375f',
                alpha=0.95,
                edgecolors='white',
                linewidths=0.5,
                zorder=4,
            )
        axp.scatter(
            [event_lon],
            [event_lat],
            marker='*',
            s=120,
            c='red',
            edgecolors='black',
            linewidths=0.5,
            zorder=5,
        )
        group_overlay_artists = self._draw_preview_group_overlay(axp, pierce_records)
        return {
            'axes': axp,
            'base_scatter': base_scatter,
            'highlight_scatter': highlight_scatter,
            'records': list(pierce_records),
            'bounds': (lon_min, lon_max, lat_min, lat_max),
            'group_overlay_artists': group_overlay_artists,
        }

    def _schedule_pending_preview_pierce_render(self, fig):
        if fig is None:
            return
        preview_state = getattr(fig, '_preview_state', None)
        if not preview_state:
            return
        pending = preview_state.get('pending_pierce_render')
        if not pending:
            return
        render_token = pending.get('token')

        def _render():
            current_state = getattr(fig, '_preview_state', None)
            if not current_state:
                return
            current_pending = current_state.get('pending_pierce_render')
            if not current_pending or current_pending.get('token') != render_token:
                return
            axp = current_pending.get('axes')
            evtdata = current_pending.get('evtdata')
            pierce_records = current_pending.get('records', [])
            metadata = current_pending.get('metadata', [])
            if axp is None or evtdata is None:
                current_state['pending_pierce_render'] = None
                return
            pierce_state = self._draw_preview_pierce_panel(axp, evtdata, pierce_records)
            pierce_status_message = self._preview_pierce_status_message(metadata, pierce_records)
            if pierce_status_message:
                self._draw_preview_deferred_panel(
                    axp,
                    f'{self.preview_pierce_phase} {self.preview_pierce_model}',
                    pierce_status_message,
                    xlabel='Lon',
                    ylabel='Lat',
                )
                axp.scatter(
                    [float(evtdata.evlo)],
                    [float(evtdata.evla)],
                    marker='*',
                    s=120,
                    c='red',
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=5,
                )
                pierce_state = {
                    'axes': axp,
                    'base_scatter': None,
                    'highlight_scatter': None,
                    'records': [],
                    'bounds': tuple(axp.get_xlim()) + tuple(axp.get_ylim()),
                }
            current_state['pierce_state'] = pierce_state
            current_state['pending_pierce_render'] = None
            self._attach_pierce_selectors(fig)
            self._activate_pierce_selector(fig, self.preview_pierce_selection_mode)
            fig.canvas.draw_idle()

        QTimer.singleShot(0, _render)

    def _toggle_preview_pierce_range_lock(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is None:
            return
        if self.preview_pierce_range_locked:
            self.preview_pierce_range_locked = False
            self.preview_pierce_fixed_bounds = None
            preview_index = getattr(fig, '_preview_controls', {}).get('preview_index', 0)
            self._refresh_preview_figure(fig, preview_index)
            self._update_preview_mode_button_styles(fig)
            self._set_preview_search_status(fig, 'Pierce range auto', color='#1f4e79')
            return
        bounds = pierce_state.get('bounds')
        if bounds is None:
            axes = pierce_state.get('axes')
            if axes is not None:
                xlim = axes.get_xlim()
                ylim = axes.get_ylim()
                bounds = (float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1]))
        if bounds is None:
            return
        self.preview_pierce_range_locked = True
        self.preview_pierce_fixed_bounds = tuple(float(value) for value in bounds)
        self._update_preview_mode_button_styles(fig)

    def _toggle_preview_group_overlay(self, fig, preview_index):
        self.preview_group_overlay_enabled = not bool(getattr(self, 'preview_group_overlay_enabled', False))
        if self.preview_group_overlay_enabled:
            self.preview_ungrouped_only_enabled = False
        self._refresh_preview_figure(fig, preview_index)
        self._update_preview_mode_button_styles(fig)
        if self.preview_group_overlay_enabled:
            self._set_preview_search_status(fig, 'Group numbers shown on pierce map', color='#1f4e79')
        else:
            self._set_preview_search_status(fig, 'Group numbers hidden; restored normal pierce map', color='#1f4e79')

    def _select_preview_ungrouped_waveforms(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        group_map = self._preview_group_wave_map()
        ungrouped_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name') and not group_map.get(str(meta.get('wave_name')))
        ]
        if not ungrouped_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, ungrouped_wave_names)
        self._apply_preview_selection(fig)
        return len(ungrouped_wave_names)

    def _toggle_preview_ungrouped_only(self, fig, preview_index):
        self.preview_group_overlay_enabled = False
        self.preview_ungrouped_only_enabled = False
        selected_count = self._select_preview_ungrouped_waveforms(fig)
        self._update_preview_mode_button_styles(fig)
        if selected_count == 0:
            self._set_preview_search_status(fig, 'No ungrouped waveforms in current preview', color='#8b0000')
        else:
            self._set_preview_search_status(fig, f'Selected {selected_count} ungrouped waveform(s)', color='#1f4e79')

    def _stack_member_pierce_button_label(self):
        return 'SrcPts:on' if getattr(self, 'preview_stack_show_member_pierce', False) else 'SrcPts:off'

    def _sync_stack_member_pierce_button(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        button = controls.get('stack_member_pierce_button')
        if button is None:
            return
        try:
            button.setText(self._stack_member_pierce_button_label())
        except AttributeError:
            pass

    def _toggle_stack_member_pierce_points(self, fig, preview_index):
        self.preview_stack_show_member_pierce = not bool(getattr(self, 'preview_stack_show_member_pierce', False))
        self._refresh_preview_figure(fig, preview_index)
        self._sync_stack_member_pierce_button(fig)
        status = 'shown' if self.preview_stack_show_member_pierce else 'hidden'
        self._set_preview_search_status(fig, f'Source member pierce points {status}', color='#1f4e79')

    def _sync_stack_preview_display_button(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        button = controls.get('stack_preview_display_button')
        if button is None:
            return
        try:
            button.setText(self._stack_preview_display_button_label())
        except AttributeError:
            pass

    def _toggle_stack_preview_display_mode(self, fig, preview_index):
        next_mode = 'top' if self._stack_preview_display_mode() == 'overlay' else 'overlay'
        self.preview_stack_display_mode = next_mode
        stack_wave_name = getattr(fig, '_stack_preview_wave_name', None)
        if stack_wave_name:
            fig._preview_forced_selected_wave_names = [stack_wave_name]
        self._refresh_preview_figure(fig, preview_index)
        self._sync_stack_preview_display_button(fig)
        mode_label = 'Top distance' if next_mode == 'top' else 'Overlay'
        self._set_preview_search_status(fig, f'Stack preview mode: {mode_label}', color='#1f4e79')

    def _preview_index_from_pierce_click(self, preview_state, event):
        if preview_state is None or event.xdata is None or event.ydata is None:
            return None
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is None:
            return None
        records = pierce_state.get('records', [])
        if not records:
            return None
        closest_index = None
        closest_distance = None
        metadata = preview_state.get('metadata', [])
        for record in records:
            dx = float(record.longitude) - float(event.xdata)
            dy = float(record.latitude) - float(event.ydata)
            distance = dx * dx + dy * dy
            if closest_distance is None or distance < closest_distance:
                for idx, meta in enumerate(metadata):
                    if meta.get('wave_name') in self._preview_wave_names_for_pierce_record(record.wave_name):
                        closest_distance = distance
                        closest_index = idx
                        break
        return closest_index

    def _preview_wave_names_for_pierce_record(self, record_wave_name):
        wave_name = str(record_wave_name or '')
        if self._is_stack_preview_member_pierce_name(wave_name):
            return {wave_name, wave_name.split('member::', 1)[1]}
        return {wave_name}

    def _preview_selected_record_wave_names(self, selected_wave_names):
        record_wave_names = {
            str(wave_name)
            for wave_name in (selected_wave_names or set())
            if wave_name
        }
        if getattr(self, 'stack_mode', False):
            plain_wave_names = list(record_wave_names)
            record_wave_names.update(
                self._stack_preview_pierce_member_record_name(wave_name)
                for wave_name in plain_wave_names
                if not self._is_stack_preview_member_pierce_name(wave_name)
            )
        return record_wave_names

    def _select_preview_wave_names(self, fig, wave_names, mode='single'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None or not wave_names:
            return 0
        metadata = preview_state.get('metadata', [])
        matched_indices = [
            idx for idx, meta in enumerate(metadata)
            if meta.get('wave_name') in wave_names
        ]
        if not matched_indices:
            return 0
        if mode == 'single':
            preview_state['selected_indices'] = set(matched_indices)
        else:
            preview_state['selected_indices'].update(matched_indices)
        preview_state['active_index'] = matched_indices[0]
        preview_state['anchor_index'] = matched_indices[0]
        self._apply_preview_selection(fig)
        return len(matched_indices)

    def _select_pierce_points_in_region(self, fig, x1, y1, x2, y2, shape='rect'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is None:
            return 0
        records = pierce_state.get('records', [])
        if not records:
            return 0

        selected_wave_names = []
        xmin, xmax = sorted((float(x1), float(x2)))
        ymin, ymax = sorted((float(y1), float(y2)))
        if shape == 'ellipse':
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rx = max((xmax - xmin) / 2.0, 1e-9)
            ry = max((ymax - ymin) / 2.0, 1e-9)
            for record in records:
                nx = (float(record.longitude) - cx) / rx
                ny = (float(record.latitude) - cy) / ry
                if (nx * nx + ny * ny) <= 1.0:
                    selected_wave_names.append(record.wave_name)
        else:
            for record in records:
                if xmin <= float(record.longitude) <= xmax and ymin <= float(record.latitude) <= ymax:
                    selected_wave_names.append(record.wave_name)
        return self._select_preview_wave_names(fig, selected_wave_names, mode='single')

    def _select_azimuth_points_in_region(self, fig, x1, y1, x2, y2, shape='rect'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        metadata = preview_state.get('metadata', [])
        evtdata = preview_state.get('evtdata')
        y_values = np.asarray(preview_state.get('y_values', evtdata.gcarc if evtdata is not None else []), dtype=float)
        if evtdata is None or len(metadata) == 0 or y_values.size == 0:
            return 0

        xmin, xmax = sorted((float(x1), float(x2)))
        ymin, ymax = sorted((float(y1), float(y2)))
        selected_wave_names = []
        if shape == 'ellipse':
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            rx = max((xmax - xmin) / 2.0, 1e-9)
            ry = max((ymax - ymin) / 2.0, 1e-9)
            for idx, meta in enumerate(metadata):
                nx = (float(evtdata.az[idx]) - cx) / rx
                ny = (float(y_values[idx]) - cy) / ry
                if (nx * nx + ny * ny) <= 1.0:
                    wave_name = meta.get('wave_name')
                    if wave_name:
                        selected_wave_names.append(wave_name)
        else:
            for idx, meta in enumerate(metadata):
                x_val = float(evtdata.az[idx])
                y_val = float(y_values[idx])
                if xmin <= x_val <= xmax and ymin <= y_val <= ymax:
                    wave_name = meta.get('wave_name')
                    if wave_name:
                        selected_wave_names.append(wave_name)
        return self._select_preview_wave_names(fig, selected_wave_names, mode='single')

    def _set_preview_pierce_view(self, fig, preview_index, phase=None, model=None):
        if phase is not None:
            self.preview_pierce_phase = str(phase)
        if model is not None:
            self.preview_pierce_model = str(model).lower()
        self.preview_pierce_cache.pop(self._pierce_cache_key(), None)
        self._refresh_preview_figure(fig, preview_index)
        self._set_preview_search_status(
            fig,
            f'Pierce: {self.preview_pierce_phase}/{self.preview_pierce_model}',
            color='#1f4e79',
        )

    def _toggle_pierce_selector_mode(self, fig, mode):
        self._activate_pierce_selector(fig, mode)
        mode_label = {
            'point': 'point',
            'rect': 'rect',
            'ellipse': 'circle',
        }.get(self.preview_pierce_selection_mode, self.preview_pierce_selection_mode)
        self._set_preview_search_status(fig, f'Pierce selector: {mode_label}', color='#1f4e79')

    def _activate_pierce_selector(self, fig, mode):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is None:
            return
        requested_mode = str(mode or 'point')
        if self.preview_pierce_selection_mode == requested_mode and requested_mode != 'point':
            requested_mode = 'point'
        self.preview_pierce_selection_mode = requested_mode
        rect_selector = pierce_state.get('rect_selector')
        ellipse_selector = pierce_state.get('ellipse_selector')
        if rect_selector is not None:
            rect_selector.set_active(self.preview_pierce_selection_mode == 'rect')
        if ellipse_selector is not None:
            ellipse_selector.set_active(self.preview_pierce_selection_mode == 'ellipse')
        az_rect_selector = preview_state.get('az_rect_selector')
        az_ellipse_selector = preview_state.get('az_ellipse_selector')
        if az_rect_selector is not None:
            az_rect_selector.set_active(self.preview_pierce_selection_mode == 'rect')
        if az_ellipse_selector is not None:
            az_ellipse_selector.set_active(self.preview_pierce_selection_mode == 'ellipse')
        self._update_preview_mode_button_styles(fig)

    def _preview_mode_button_style(self, active=False, accent='#2f6fed'):
        if active:
            return (
                f'QPushButton {{'
                f'background-color: {accent}; color: white; border: 1px solid {accent}; '
                f'border-radius: 4px; font-weight: 600; padding: 2px 6px;'
                f'}}'
                f'QPushButton:hover {{ background-color: {accent}; color: white; }}'
            )
        return (
            'QPushButton {'
            'background-color: #f1f3f5; color: #222222; border: 1px solid #c9ced6; '
            'border-radius: 4px; padding: 2px 6px;'
            '}'
            'QPushButton:hover { background-color: #e6ebf2; color: #222222; }'
        )

    def _update_preview_mode_button_styles(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        rect_button = controls.get('selector_rect_button')
        circle_button = controls.get('selector_circle_button')
        fixrange_button = controls.get('fixrange_button')
        group_overlay_button = controls.get('group_overlay_button')
        ungrouped_button = controls.get('ungrouped_button')
        layout_button = controls.get('layout_button')
        viewmode_button = controls.get('viewmode_button')
        curve_pick_button = controls.get('curve_pick_button')
        keepmode_button = controls.get('keepmode_button')
        if rect_button is not None:
            rect_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_pierce_selection_mode == 'rect', accent='#1f6feb')
            )
        if circle_button is not None:
            circle_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_pierce_selection_mode == 'ellipse', accent='#0f8a5f')
            )
        if fixrange_button is not None:
            fixrange_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_pierce_range_locked, accent='#b26a00')
            )
        if group_overlay_button is not None:
            group_overlay_button.setStyleSheet(
                self._preview_mode_button_style(
                    bool(getattr(self, 'preview_group_overlay_enabled', False)),
                    accent='#8b5e00',
                )
            )
        if ungrouped_button is not None:
            ungrouped_button.setStyleSheet(
                self._preview_mode_button_style(
                    bool(getattr(self, 'preview_ungrouped_only_enabled', False)),
                    accent='#444444',
                )
            )
        if layout_button is not None:
            layout_button.setText(self._preview_layout_summary())
            layout_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_trace_layout_mode == 'even', accent='#7a3fd1')
            )
        if viewmode_button is not None:
            viewmode_button.setText(self._preview_view_mode_label())
            viewmode_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_view_mode == 'tall', accent='#a23b72')
            )
        if curve_pick_button is not None:
            curve_state = self._preview_curve_pick_state(fig)
            curve_pick_button.setStyleSheet(
                self._preview_mode_button_style(curve_state.get('active', False), accent='#c2410c')
            )
        if keepmode_button is not None:
            keepmode_button.setText(self._preview_keep_mode_label())
            keepmode_button.setStyleSheet(
                self._preview_mode_button_style(self.preview_keep_selection_mode == 'unselected', accent='#0f766e')
            )

    def _preview_keep_mode_label(self):
        if self.preview_keep_selection_mode == 'unselected':
            return 'K:Other'
        return 'K:Sel'

    def _toggle_preview_keep_mode(self, fig):
        if self.preview_keep_selection_mode == 'selected':
            self.preview_keep_selection_mode = 'unselected'
        else:
            self.preview_keep_selection_mode = 'selected'
        self._update_preview_mode_button_styles(fig)
        mode_text = 'keep unselected waveforms' if self.preview_keep_selection_mode == 'unselected' else 'keep selected waveforms'
        self._set_preview_search_status(fig, f'K mode: {mode_text}', color='#1f4e79')

    def _attach_pierce_selectors(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is None:
            return
        axp = pierce_state.get('axes')
        if axp is None:
            return

        def on_rect_select(eclick, erelease):
            selected_count = self._select_pierce_points_in_region(
                fig,
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
                shape='rect',
            )
            if selected_count:
                self._set_preview_search_status(fig, f'Pierce rect selected {selected_count} waveform(s)', color='#1f4e79')

        def on_ellipse_select(eclick, erelease):
            selected_count = self._select_pierce_points_in_region(
                fig,
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
                shape='ellipse',
            )
            if selected_count:
                self._set_preview_search_status(fig, f'Pierce ellipse selected {selected_count} waveform(s)', color='#1f4e79')

        rect_selector = RectangleSelector(
            axp,
            on_rect_select,
            useblit=False,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            spancoords='data',
            interactive=False,
        )
        ellipse_selector = EllipseSelector(
            axp,
            on_ellipse_select,
            useblit=False,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            spancoords='data',
            interactive=False,
        )
        pierce_state['rect_selector'] = rect_selector
        pierce_state['ellipse_selector'] = ellipse_selector
        self._activate_pierce_selector(fig, self.preview_pierce_selection_mode)

    def _preview_axes(self, fig):
        axr = fig.axes[0] if len(fig.axes) > 0 else None
        axb = fig.axes[1] if len(fig.axes) > 1 else None
        axp = fig.axes[2] if len(fig.axes) > 2 else None
        return axr, axb, axp

    def _attach_azimuth_selectors(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        _axr, axb, _axp = self._preview_axes(fig)
        if axb is None:
            return

        def on_rect_select(eclick, erelease):
            selected_count = self._select_azimuth_points_in_region(
                fig,
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
                shape='rect',
            )
            if selected_count:
                self._set_preview_search_status(fig, f'Az rect selected {selected_count} waveform(s)', color='#1f4e79')

        def on_ellipse_select(eclick, erelease):
            selected_count = self._select_azimuth_points_in_region(
                fig,
                eclick.xdata,
                eclick.ydata,
                erelease.xdata,
                erelease.ydata,
                shape='ellipse',
            )
            if selected_count:
                self._set_preview_search_status(fig, f'Az ellipse selected {selected_count} waveform(s)', color='#1f4e79')

        rect_selector = RectangleSelector(
            axb,
            on_rect_select,
            useblit=False,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            spancoords='data',
            interactive=False,
        )
        ellipse_selector = EllipseSelector(
            axb,
            on_ellipse_select,
            useblit=False,
            button=[1],
            minspanx=0.01,
            minspany=0.01,
            spancoords='data',
            interactive=False,
        )
        preview_state['az_rect_selector'] = rect_selector
        preview_state['az_ellipse_selector'] = ellipse_selector
        rect_selector.set_active(self.preview_pierce_selection_mode == 'rect')
        ellipse_selector.set_active(self.preview_pierce_selection_mode == 'ellipse')

    def _lowq_preview_directory(self):
        return self._target_directory_for_bucket("LowQ_sac")

    def _phase_display_label(self, tmarker):
        marker_key = str(tmarker or '')
        if marker_key.startswith('t'):
            marker_key = marker_key[1:]
        return self.phase_display_labels.get(marker_key, f"t{marker_key}")

    def _alignment_marker_key(self, marker=None):
        marker_key = str(marker or self.tmarker or '')
        if marker_key.startswith('t'):
            marker_key = marker_key[1:]
        return marker_key

    def _wave_station_label(self, wave_index):
        if wave_index is not None and 0 <= wave_index < len(self.filenames):
            return self.filenames[wave_index]
        return ''

    def missing_alignment_wave_names(self, marker=None):
        marker_key = self._alignment_marker_key(marker)
        marker_values = self.markers.get(marker_key)
        if marker_values is None:
            return []
        missing_wave_names = []
        for idx, wave_name in enumerate(self.ori_sacnames):
            marker_value = marker_values.get(wave_name, math.nan)
            if math.isnan(marker_value):
                missing_wave_names.append(self._wave_station_label(idx) or wave_name)
        return missing_wave_names

    def missing_alignment_wave_indices(self, marker=None):
        marker_key = self._alignment_marker_key(marker)
        marker_values = self.markers.get(marker_key)
        if marker_values is None:
            return []
        missing_indices = []
        for idx, wave_name in enumerate(self.ori_sacnames):
            if self._is_preview_hidden_wave(wave_name):
                continue
            marker_value = marker_values.get(wave_name, math.nan)
            if math.isnan(marker_value):
                missing_indices.append(idx)
        return missing_indices

    def alignment_status_summary(self, marker=None, max_names=6):
        marker_key = self._alignment_marker_key(marker)
        missing_wave_names = self.missing_alignment_wave_names(marker=marker_key)
        total_count = len(self.ori_sacnames)
        missing_count = len(missing_wave_names)
        picked_count = total_count - missing_count
        phase_label = self._phase_display_label(marker_key)
        stack_skip_count = len(getattr(self, 'stack_skipped_wave_files', []))
        stack_repair_report = getattr(self, 'stack_repair_report', None) or {}
        stack_health_report = getattr(self, 'stack_health_report', None) or {}
        repaired_count = len(stack_repair_report.get('sidecars_updated', []))
        if stack_repair_report.get('marker_updated'):
            repaired_count += 1
        invalid_sac_count = len(stack_health_report.get('invalid_sac_files', []))
        missing_sidecar_count = len(stack_health_report.get('missing_sidecars', []))
        orphan_sidecar_count = len(stack_health_report.get('orphan_sidecars', []))
        invalid_sidecar_count = len(stack_health_report.get('invalid_sidecars', []))
        repairable_sidecar_count = len(stack_health_report.get('sidecars_needing_repair', []))
        if missing_count == 0:
            summary = f'Align {phase_label} complete {picked_count}/{total_count}'
            if stack_skip_count:
                summary = f'{summary} | Stack skipped {stack_skip_count}'
            if repaired_count:
                summary = f'{summary} | Stack repaired {repaired_count}'
            if invalid_sac_count:
                summary = f'{summary} | Stack invalid {invalid_sac_count}'
            if missing_sidecar_count:
                summary = f'{summary} | Stack missing sidecar {missing_sidecar_count}'
            if orphan_sidecar_count:
                summary = f'{summary} | Stack orphan sidecar {orphan_sidecar_count}'
            if invalid_sidecar_count:
                summary = f'{summary} | Stack invalid sidecar {invalid_sidecar_count}'
            if repairable_sidecar_count:
                summary = f'{summary} | Stack sidecar repair {repairable_sidecar_count}'
            return summary
        preview_names = ', '.join(missing_wave_names[:max_names])
        if missing_count > max_names:
            preview_names = f'{preview_names}, ...'
        summary = f'Align {phase_label} missing {missing_count}/{total_count} [{preview_names}]'
        if stack_skip_count:
            summary = f'{summary} | Stack skipped {stack_skip_count}'
        if repaired_count:
            summary = f'{summary} | Stack repaired {repaired_count}'
        if invalid_sac_count:
            summary = f'{summary} | Stack invalid {invalid_sac_count}'
        if missing_sidecar_count:
            summary = f'{summary} | Stack missing sidecar {missing_sidecar_count}'
        if orphan_sidecar_count:
            summary = f'{summary} | Stack orphan sidecar {orphan_sidecar_count}'
        if invalid_sidecar_count:
            summary = f'{summary} | Stack invalid sidecar {invalid_sidecar_count}'
        if repairable_sidecar_count:
            summary = f'{summary} | Stack sidecar repair {repairable_sidecar_count}'
        return summary

    def _parse_standard_phase_tokens(self, text):
        raw_tokens = [item.strip() for item in str(text or '').split(',') if item.strip()]
        if not raw_tokens:
            return [], None
        phase_keys = []
        invalid_tokens = []
        for token in raw_tokens:
            token_lower = token.lower()
            if token_lower.startswith('t') and token_lower[1:] in self.marker_styles:
                marker_key = token_lower[1:]
            elif token_lower in self.marker_styles:
                marker_key = token_lower
            else:
                invalid_tokens.append(token)
                continue
            if marker_key == '4':
                continue
            if marker_key not in phase_keys:
                phase_keys.append(marker_key)
        if invalid_tokens:
            return [], f"Unsupported phase token(s): {', '.join(invalid_tokens)}"
        return phase_keys, None

    def _canonical_standard_phase_tokens(self, text):
        phase_keys, error_message = self._parse_standard_phase_tokens(text)
        if error_message is not None:
            return None, error_message
        return ','.join(f"t{marker_key}" for marker_key in phase_keys), None

    def _notify_phase_tokens_changed(self):
        if self.phase_tokens_change_callback is not None:
            self.phase_tokens_change_callback(self.standard_export_phase_tokens)

    def set_standard_phase_tokens(self, text, preview_index=None, refresh=False, sync_controls=True):
        canonical_tokens, error_message = self._canonical_standard_phase_tokens(text)
        if error_message is not None:
            return None, error_message
        canonical_tokens = canonical_tokens or ''
        self.standard_export_phase_tokens = canonical_tokens
        self._notify_phase_tokens_changed()
        if sync_controls and self.plotfig is not None and hasattr(self.plotfig, '_preview_controls'):
            controls = getattr(self.plotfig, '_preview_controls', {})
            box_std_phases = controls.get('std_phases')
            if box_std_phases is not None and box_std_phases.text != canonical_tokens:
                box_std_phases.set_val(canonical_tokens)
            if preview_index is None:
                preview_index = controls.get('preview_index')
        if refresh and preview_index is not None:
            if self.plotfig is not None and plt.fignum_exists(self.plotfig.number):
                self._refresh_preview_figure(self.plotfig, preview_index)
            self._refresh_compare_for_preview_index(preview_index)
        return canonical_tokens, None

    def _standard_phase_label(self, marker_key, duplicate_labels=None):
        if marker_key == '4':
            return None
        display_label = self.phase_display_labels.get(marker_key, f"t{marker_key}")
        prefix = self.phase_label_prefixes.get(marker_key, '')
        if prefix:
            suffix = 't' if prefix.lower() == 'theory' else 'a'
            display_label = rf'{display_label}$_{{{suffix}}}$'
        duplicate_labels = duplicate_labels or set()
        if display_label in duplicate_labels:
            return f"t{marker_key}:{display_label}"
        return display_label

    def _standard_phase_legend_label(self, marker_key):
        if marker_key == '4':
            return None
        display_label = self.phase_display_labels.get(marker_key, f"t{marker_key}")
        prefix = self.phase_label_prefixes.get(marker_key, '')
        if prefix:
            suffix = 't' if prefix.lower() == 'theory' else 'a'
            return rf'{display_label}$_{{{suffix}}}$'
        return display_label

    def _phase_keys_with_alignment(self, phase_keys, align_marker_key):
        merged = []
        align_key = str(align_marker_key or '')
        if align_key.startswith('t'):
            align_key = align_key[1:]
        if align_key in self.marker_styles:
            merged.append(align_key)
        for marker_key in phase_keys:
            if marker_key not in merged:
                merged.append(marker_key)
        return merged

    def _visible_phase_keys_in_evtdata(self, evtdata, align_marker_key, phase_keys, reference_times=None):
        if evtdata is None or not phase_keys:
            return []
        xmin = evtdata.x1
        xmax = evtdata.x2
        visible_phase_keys = []
        for marker_key in phase_keys:
            for tr in evtdata.wave_ori:
                wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
                relative_time = self._preview_relative_phase_time(
                    align_marker_key,
                    marker_key,
                    wave_name,
                    reference_times=reference_times,
                    trace=tr,
                )
                if math.isnan(relative_time):
                    continue
                if xmin <= relative_time <= xmax:
                    visible_phase_keys.append(marker_key)
                    break
        return visible_phase_keys

    def _preview_standard_wave_color(self, wave_name):
        if self._is_user1_wave(wave_name):
            return '#00461f'
        if self._is_user5_wave(wave_name):
            return '#006d77'
        if self._is_user4_wave(wave_name):
            return '#9a4f00'
        if self._is_preview_purple_wave(wave_name):
            return '#2f007d'
        return 'black'

    def _preview_standard_wave_style(self, trace, wave_name):
        # Stack trace stands out in red/bold on standard (gcarc/az) plots, like
        # in the live preview; member traces keep their per-state color and the
        # thin default width.
        if getattr(trace.stats, 'dephasekit_stack_preview_role', '') == 'stack':
            return STACK_TRACE_COLOR, STACK_TRACE_LINEWIDTH
        return self._preview_standard_wave_color(wave_name), 0.35

    def _preview_layout_summary(self):
        return 'Even' if self.preview_trace_layout_mode == 'even' else 'Real'

    def _preview_view_mode_label(self):
        return 'Tall' if getattr(self, 'preview_view_mode', 'wide') == 'tall' else 'Wide'

    def _preview_window_size_hint(self):
        if getattr(self, 'preview_view_mode', 'wide') == 'tall':
            return 860, 1380
        return None

    def _preview_overlay_positions(self):
        if getattr(self, 'preview_view_mode', 'wide') == 'tall':
            return {
                'info': (0.50, 0.938),
                'status': (0.50, 0.922),
                'info_size': 10,
                'status_size': 8.0,
            }
        return {
            'info': (0.50, 0.875),
            'status': (0.50, 0.845),
            'info_size': 11,
            'status_size': 8.5,
        }

    def _current_user1_station_names(self):
        station_names = []
        seen = set()
        for index, wave_name in enumerate(self.ori_sacnames):
            if not self._is_user1_wave(wave_name):
                continue
            try:
                tr = self.wave[index]
                station_name = f"{tr.stats.network}.{tr.stats.station}"
            except Exception:
                parts = str(wave_name).split('.')
                station_name = '.'.join(parts[:2]) if len(parts) >= 2 else str(wave_name)
            if station_name not in seen:
                seen.add(station_name)
                station_names.append(station_name)
        return station_names

    def _format_user1_station_summary(self, station_names, per_line=6):
        if not station_names:
            return None
        lines = []
        for start in range(0, len(station_names), per_line):
            lines.append(', '.join(station_names[start:start + per_line]))
        body = '\n'.join(lines)
        return f"Moved U1 stations ({len(station_names)}):\n{body}"

    def _add_snapshot_station_summary(self, fig):
        summary_text = self._format_user1_station_summary(self._current_user1_station_names())
        if not summary_text:
            return None
        preview_axes = self._preview_axes(fig)
        target_ax = None
        if preview_axes is not None:
            _axr, axb, axp = preview_axes
            target_ax = axp if axp is not None else axb
            if target_ax is None:
                target_ax = axb
        if target_ax is not None:
            return target_ax.text(
                0.98, 0.02, summary_text,
                ha='right', va='bottom',
                fontsize=7.8, color='#303030',
                bbox=dict(boxstyle='round,pad=0.22', facecolor='white', edgecolor='#bbbbbb', alpha=0.88),
                transform=target_ax.transAxes,
                zorder=20,
            )
        return fig.text(
            0.985, 0.02, summary_text,
            ha='right', va='bottom',
            fontsize=8.5, color='#303030',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='#bbbbbb', alpha=0.88),
            transform=fig.transFigure,
            zorder=20,
        )

    def _preview_y_axis_config(self, evtdata, order='gcarc'):
        if self.preview_trace_layout_mode == 'even':
            spacing_step = max(self.preview_even_spacing_min, float(self.preview_even_spacing_step))
            y_values = np.arange(evtdata.sta_num, dtype=float) * spacing_step + spacing_step
            max_ticks = min(8, evtdata.sta_num)
            if evtdata.sta_num <= max_ticks:
                tick_indices = np.arange(evtdata.sta_num)
            else:
                tick_indices = np.linspace(0, evtdata.sta_num - 1, max_ticks, dtype=int)
                tick_indices = np.unique(tick_indices)
            source_values = evtdata.az if order == 'az' else evtdata.gcarc
            tick_positions = y_values[tick_indices]
            tick_labels = [f"{source_values[idx]:.1f}" for idx in tick_indices]
            ylabel = r'Azimuth($^\circ$)' if order == 'az' else r'Epicenter distance($^\circ$)'
            return y_values, tick_positions, tick_labels, ylabel
        if order == 'az':
            y_values = np.asarray(evtdata.az, dtype=float)
            ylabel = r'Azimuth($^\circ$)'
        else:
            y_values = np.asarray(evtdata.gcarc, dtype=float)
            ylabel = r'Epicenter distance($^\circ$)'
        ymin = float(np.min(y_values))
        ymax = float(np.max(y_values))
        if np.isclose(ymin, ymax):
            tick_positions = np.array([ymin])
        else:
            tick_positions = np.linspace(ymin, ymax, min(7, max(2, evtdata.sta_num)))
        tick_labels = [f"{tick:.1f}" for tick in tick_positions]
        return y_values, tick_positions, tick_labels, ylabel

    def _adjust_preview_even_spacing(self, fig, preview_index, direction):
        delta = float(direction) * self.preview_even_spacing_adjust_step
        new_spacing = min(
            self.preview_even_spacing_max,
            max(self.preview_even_spacing_min, self.preview_even_spacing_step + delta)
        )
        if abs(new_spacing - self.preview_even_spacing_step) < 1e-9:
            self._set_preview_search_status(fig, f'Even gap {self.preview_even_spacing_step:.1f}', color='#1f4e79')
            return
        self.preview_even_spacing_step = new_spacing
        if self.preview_trace_layout_mode == 'even':
            self._refresh_preview_figure(fig, preview_index)
            self._refresh_compare_for_preview_index(preview_index)
            self._set_preview_search_status(fig, f'Even gap {self.preview_even_spacing_step:.1f}', color='#1f4e79')
            return
        self._set_preview_search_status(
            fig,
            f'Even gap set to {self.preview_even_spacing_step:.1f}; switch to Even to use it',
            color='#1f4e79'
        )

    def set_page(self):
        axs = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]  # 将所有的子图放在一个列表中
        self._sync_pick_window_visible_pages()
        for slot_index, ax in enumerate(axs):
            wave_index = self._visible_wave_index_for_page_slot(slot_index)
            if wave_index is not None:
                if self.axis_mode == 'relative' and not self._has_valid_alignment_time(wave_index):
                    x1, x2 = self.xlim[0], self.xlim[1]
                else:
                    x1, x2 = self._axis_window_for_wave(wave_index)
                ax.set_xlim(x1, x2)
                diff = x2 - x1
                if diff <= 10:
                    interval = 1
                elif 10 < diff <= 20:
                    interval = 2
                elif 20 < diff <= 50:
                    interval = 5
                else:
                    interval = 10
                # major
                # axs[ax_index].xaxis.set_major_locator(ticker.MultipleLocator(10))
                # minor
                ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
                #
                ax.set_xticks(np.arange(x1, x2, interval))
                #
                ax.tick_params(axis='x', which='major', length=4, labelsize=10)  # 主刻度标签的大小
                ax.tick_params(axis='x', which='minor', length=1, labelsize=0)  # 隐藏次刻度标签
            else:
                ax.axis('off')

    def init_variables(self):
        # 主窗固定有 5 个 subplot 槽位；单条/少量波形目录也要预留足够槽位，
        # 否则空槽位清理时会出现 list assignment 越界。
        slot_count = max(self.sta_num, 5)
        self.A1lines = [[] for i in range(slot_count)]  # 每一条波形数据点的数组

    def init_figure(self, width=21, height=11, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax1 = self.fig.add_subplot(5, 1, 1)
        self.ax2 = self.fig.add_subplot(5, 1, 2)
        self.ax3 = self.fig.add_subplot(5, 1, 3)
        self.ax4 = self.fig.add_subplot(5, 1, 4)
        self.ax5 = self.fig.add_subplot(5, 1, 5)

        # 设置子图的位置和尺寸，如果需要可以调整
        # self.ax1.set_position([left, bottom, width, height])
        # self.ax2.set_position([left, bottom, width, height])
        # self.ax3.set_position([left, bottom, width, height])
        # self.ax4.set_position([left, bottom, width, height])
        # self.ax5.set_position([left, bottom, width, height])

    def set_figure(self):
        """
        visible=True 表示网格线是可见的，
        which='major' 表示只为主刻度添加网格线。
        axis='y' 表示只在 y轴上添加网格线，
        """
        linewidth = 0.1
        self.ax1.grid(visible=True, linewidth=linewidth)
        self.ax2.grid(visible=True, linewidth=linewidth)
        self.ax3.grid(visible=True, linewidth=linewidth)
        self.ax4.grid(visible=True, linewidth=linewidth)
        self.ax5.grid(visible=True, linewidth=linewidth)
        self.ax1.set_ylabel("Amplitude")
        self.ax2.set_ylabel("Amplitude")
        self.ax3.set_ylabel("Amplitude")
        self.ax4.set_ylabel("Amplitude")
        self.ax5.set_ylabel("Amplitude")

    def read_sac(self, dt=0.01, order='gcarc'):
        if not isinstance(order, str):
            raise TypeError('The order must be str type')
        elif not order in ['baz', 'gcarc', 'az']:
            raise ValueError('The order must be \'baz\', \'gcarc\' or \'az\'')
        else:
            pass
        self.order = order
        # wavepath
        if self.wavepath.endswith(os.sep):
            self.wavepath = self.wavepath.rstrip(os.sep)
        if self.stack_mode:
            self._sync_stack_workspace_sac_headers_from_sidecars()
        tmp_files = self._wave_files_for_suffix()
        # 可选成员过滤：只保留 member_filter 集合里的波形（直跳 group 拾取窗用）
        if getattr(self, 'member_filter', None):
            if self.stack_mode:
                tmp_files = [
                    f for f in tmp_files
                    if stack_wave_name_from_path(self.wavepath, f) in self.member_filter
                ]
            else:
                tmp_files = [
                    f for f in tmp_files
                    if basename(f) in self.member_filter
                ]
        if len(tmp_files) != 0:
            if self.stack_mode:
                # iter_stack_sac_paths already returns naturally-sorted paths
                # (group2 < group10); avoid a lexicographic re-sort here.
                self.ori_sacnames = np.array([
                    stack_wave_name_from_path(self.wavepath, tmp_file)
                    for tmp_file in tmp_files
                ])
            else:
                self.ori_sacnames = np.array([
                    basename(tmp_file)
                    for tmp_file in sorted(tmp_files, key=lambda p: _natural_sort_key(basename(p)))
                ])
            # sorted() 防止不同系统排序不一致
            self.ori_evtname = basename(self._semantic_event_dir())

        else:
            print('No valid waveforms in {}'.format(self.wavepath))
            sys.exit(1)

        # get dt
        st = obspy.Stream()
        valid_wave_names = []
        skipped_wave_files = []
        for tmp_file in tmp_files:
            wave_name = stack_wave_name_from_path(self.wavepath, tmp_file) if self.stack_mode else basename(tmp_file)
            try:
                tr = obspy.read(tmp_file)[0]
            except Exception as exc:
                if self.stack_mode:
                    skipped_wave_files.append({'wave_name': wave_name, 'reason': str(exc)})
                    continue
                raise
            tr.stats.dephasekit_wave_name = wave_name
            if self.stack_mode:
                self._apply_stack_sidecar_to_trace(tr, wave_name)
            st += tr
            valid_wave_names.append(wave_name)
        st = st.sort()
        if len(st) == 0:
            print('No valid waveforms in {}'.format(self.wavepath))
            sys.exit(1)
        self.ori_sacnames = np.array([
            str(getattr(tr.stats, 'dephasekit_wave_name', valid_wave_names[idx]))
            for idx, tr in enumerate(st)
        ])
        self.stack_skipped_wave_files = skipped_wave_files
        delta_values = [tr.stats.delta for tr in st]
        self.dt = float(np.median(delta_values))
        target_fs = 1.0 / self.dt
        """
        delta_values = [round(tr.stats.delta, 6) for tr in st]
        delta_count = Counter(delta_values)
        target_delta, max_count = delta_count.most_common(1)[0]
        self.dt = target_delta
        target_fs = 1.0 / target_delta
        """
        #
        # self.wave = obspy.read(join(self.wavepath, '*' + self.suffix)).sort().resample(1.0 / self.dt, window="hann")
        self.wave_raw = st.copy()
        for tr in self.wave_raw:
                if abs(tr.stats.sampling_rate - target_fs) > 1e-3:
                    tr.resample(target_fs, window="hann")
        self.wave = self.wave_raw.copy()
        self.network = np.array([tr.stats.network for tr in self.wave])
        self.station = np.array([tr.stats.station for tr in self.wave])
        self.year = np.array([str(int(_safe_float(_sac_attr(tr, 'nzyear', 0)))).zfill(4) for tr in self.wave])
        self.jday = np.array([str(int(_safe_float(_sac_attr(tr, 'nzjday', 0)))).zfill(3) for tr in self.wave])
        self.hour = np.array([str(int(_safe_float(_sac_attr(tr, 'nzhour', 0)))).zfill(2) for tr in self.wave])
        self.min = np.array([str(int(_safe_float(_sac_attr(tr, 'nzmin', 0)))).zfill(2) for tr in self.wave])
        self.sec = np.array([str(int(_safe_float(_sac_attr(tr, 'nzsec', 0)))).zfill(2) for tr in self.wave])
        # self.channel = np.array([tr.stats.sac.kcmpnm for tr in self.wave])
        self.channel = np.array([tr.stats.channel for tr in self.wave])
        self.filenames = [f"{n}.{s}" for n, s in zip(self.network, self.station)]
        if self.stack_mode:
            self.evtname = self._semantic_event_name()
        else:
            dsm_event_name = _event_name_from_dsm_path(getattr(self, 'wavepath', ''))
            self.evtname = dsm_event_name or f"{self.year[0]}.{self.jday[0]}.{self.hour[0]}.{self.min[0]}.{self.sec[0]}"
        #
        self.wavenames = [f"{ns}.{self.evtname}.{ch}{self.suffix}" for ns, ch in zip(self.filenames, self.channel)]
        #
        self.tmarker_t = np.array([
            _sac_float(tr, self.tmarker, math.nan)
            for tr in self.wave
        ])

        self.t0, self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8, self.t9, \
            = [], [], [], [], [], [], [], [], [], []
        for tr in self.wave:
            for attr, list_name in zip(['t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'],
                                       [self.t0, self.t1, self.t2, self.t3, self.t4, self.t5,
                                        self.t6, self.t7, self.t8, self.t9]):
                list_name.append(_sac_float(tr, attr, math.nan))
        (self.t0, self.t1, self.t2, self.t3, self.t4, self.t5,
         self.t6, self.t7, self.t8, self.t9) = map(np.array,
                                                   [self.t0, self.t1, self.t2, self.t3, self.t4, self.t5,
                                                    self.t6, self.t7, self.t8, self.t9])
        if self.stack_mode:
            self.tmarker_t = np.array([
                value if not math.isnan(value) else 0.0
                for value in self.tmarker_t
            ], dtype=float)
            for marker_values in (self.t0, self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8, self.t9):
                marker_values[:] = np.asarray(marker_values, dtype=float)
        self.sta_num = len(self.wave)
        self.baz = np.array([_sac_float(tr, 'baz', 0.0) for tr in self.wave], dtype=float)
        self.az = np.array([_sac_float(tr, 'az', 0.0) for tr in self.wave], dtype=float)
        self.gcarc = np.array([_sac_float(tr, 'gcarc', 0.0) for tr in self.wave], dtype=float)
        #
        self._sort(order)

        for sac_file, t0, t1, t2, t3, t4, t5, t6, t7, t8, t9 in zip(self.ori_sacnames, self.t0, self.t1, self.t2, self.t3,
                                                                    self.t4, self.t5,
                                                                    self.t6, self.t7, self.t8, self.t9):
            self.markers['0'][sac_file] = t0
            self.markers['1'][sac_file] = t1
            self.markers['2'][sac_file] = t2
            self.markers['3'][sac_file] = t3
            self.markers['4'][sac_file] = t4
            self.markers['5'][sac_file] = t5
            self.markers['6'][sac_file] = t6
            self.markers['7'][sac_file] = t7
            self.markers['8'][sac_file] = t8
            self.markers['9'][sac_file] = t9
        for sac_file, tr in zip(self.ori_sacnames, self.wave):
            for user_key in self.user_markers:
                if user_key == 'user3':
                    self.user_markers[user_key][sac_file] = math.nan
                    continue
                try:
                    self.user_markers[user_key][sac_file] = getattr(tr.stats.sac, user_key)
                except AttributeError:
                    self.user_markers[user_key][sac_file] = math.nan

        self.axpages, self.waveidx = indexpags(self.sta_num, self.maxidx)
        self.preview_pierce_cache = {}
        self.preview_pierce_generation_attempted = False
        self.theory_time_cache = {}
        self.theory_delta_summary_cache = None

        title_text = (
            "evt:{} (Latitude: {:.2f}\N{DEGREE SIGN}, Longitude: {:.2f}\N{DEGREE SIGN}, Depth:{:.1f} km)".format(
                self.evtname, _sac_float(self.wave[0], 'evla', 0.0), _sac_float(self.wave[0], 'evlo', 0.0), _sac_float(self.wave[0], 'evdp', 0.0))
        )
        self.fig.suptitle(title_text, fontsize=20)

    def _filtered_stream_copy(self, stream):
        filtered_stream = stream.copy()
        freqmin = self.bandpass_settings.get('freqmin')
        freqmax = self.bandpass_settings.get('freqmax')
        corners = self.bandpass_settings.get('corners', 2)
        passes = self.bandpass_settings.get('passes', 2)
        if freqmin is None or freqmax is None:
            return filtered_stream
        if freqmin <= 0 or freqmax <= freqmin:
            return filtered_stream
        filtered_stream.filter(
            'bandpass',
            freqmin=freqmin,
            freqmax=freqmax,
            corners=max(1, corners),
            zerophase=self._bandpass_zerophase_enabled(passes),
        )
        return filtered_stream

    def _filtered_trace_for_preview(self, wave_name):
        target_fs = 1.0 / self.dt
        if getattr(self, 'stack_mode', False):
            tr = self._trace_from_runtime_dir(wave_name)
        else:
            tr = self._trace_from_loaded_wave(wave_name)
            if tr is None:
                tr = self._trace_from_runtime_dir(wave_name)
        if abs(tr.stats.sampling_rate - target_fs) > 1e-3:
            tr.resample(target_fs, window="hann")
        self._apply_bandpass_to_trace(tr, self.bandpass_settings)
        tr.data = np.asarray(tr.data, dtype=float) * self._wave_polarity_factor(wave_name)
        return tr

    def _apply_bandpass_to_trace(self, tr, bandpass_settings):
        freqmin = bandpass_settings.get('freqmin')
        freqmax = bandpass_settings.get('freqmax')
        corners = bandpass_settings.get('corners', 2)
        passes = bandpass_settings.get('passes', 2)
        if freqmin is not None and freqmax is not None and freqmin > 0 and freqmax > freqmin:
            tr.filter(
                'bandpass',
                freqmin=freqmin,
                freqmax=freqmax,
                corners=max(1, corners),
                zerophase=self._bandpass_zerophase_enabled(passes),
            )
        return tr

    def _bandpass_zerophase_enabled(self, passes):
        # Match the SAC-style mental model users expect:
        # p1 = single-pass causal-looking filter, p2 = zero-phase forward/backward.
        return int(passes) >= 2

    def _current_bandpass_profile(self):
        freqmin = self.bandpass_settings.get('freqmin')
        freqmax = self.bandpass_settings.get('freqmax')
        if freqmin is None or freqmax is None or freqmin <= 0 or freqmax <= freqmin:
            return None
        return {
            'freqmin': float(freqmin),
            'freqmax': float(freqmax),
            'corners': int(self.bandpass_settings.get('corners', 2)),
            'passes': int(self.bandpass_settings.get('passes', 2)),
        }

    def _bandpass_profile_key(self, profile):
        return (
            round(float(profile['freqmin']), 6),
            round(float(profile['freqmax']), 6),
            int(profile['corners']),
            int(profile['passes']),
        )

    def _normalize_bandpass_profile(self, profile):
        if not isinstance(profile, dict):
            return None
        try:
            freqmin = float(profile['freqmin'])
            freqmax = float(profile['freqmax'])
            corners = int(profile.get('corners', 2))
            passes = int(profile.get('passes', 2))
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

    def _format_bandpass_label(self, profile):
        if profile is None:
            return 'Raw'
        return (
            f"BP {profile['freqmin']:g}-{profile['freqmax']:g} "
            f"n{int(profile['corners'])} p{int(profile['passes'])}"
        )

    def _short_bandpass_label(self, profile):
        if profile is None:
            return 'Raw'
        return f"{profile['freqmin']:g}-{profile['freqmax']:g}"

    def _emit_compare_status(self, message, timeout_ms=3000):
        if self.compare_status_callback is not None:
            try:
                self.compare_status_callback(message, timeout_ms=timeout_ms)
                return
            except TypeError:
                self.compare_status_callback(message)
                return
        print(message)

    def _theory_time_cache_key(self, model, phase, evdp, gcarc):
        return (
            str(model or self.theory_time_model).lower(),
            str(phase or ''),
            round(float(evdp), 4),
            round(float(gcarc), 4),
        )

    def _run_taup_onlytime(self, evdp, gcarc, phase, model=None):
        model_key = str(model or self.theory_time_model).lower()
        evdp_value = _safe_float(evdp)
        gcarc_value = _safe_float(gcarc)
        if math.isnan(evdp_value) or math.isnan(gcarc_value):
            return math.nan
        cache_key = self._theory_time_cache_key(model_key, phase, evdp_value, gcarc_value)
        if cache_key in self.theory_time_cache:
            return self.theory_time_cache[cache_key]
        cmd = [
            self.preview_pierce_taup_bin,
            'time',
            '--onlytime',
            '-h',
            f'{evdp_value:g}',
            '-p',
            str(phase),
            '--deg',
            f'{gcarc_value:g}',
            '--mod',
            model_key,
        ]
        value = math.nan
        try:
            completed = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
            for line in (completed.stdout or '').splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    value = float(line)
                    break
                except ValueError:
                    continue
        except (subprocess.SubprocessError, OSError, ValueError):
            value = math.nan
        self.theory_time_cache[cache_key] = value
        return value

    def _theory_phase_deltas_for_gcarc(self, evdp, gcarc, model=None):
        model_key = str(model or self.theory_time_model).lower()
        batch_key = self._theory_time_cache_key(model_key, 'P,pP,sP', evdp, gcarc)
        cached = self.theory_time_cache.get(batch_key)
        if isinstance(cached, dict):
            p_time = cached.get('P', math.nan)
            pp_time = cached.get('pP', math.nan)
            sp_time = cached.get('sP', math.nan)
        else:
            p_time = math.nan
            pp_time = math.nan
            sp_time = math.nan
            evdp_value = _safe_float(evdp)
            gcarc_value = _safe_float(gcarc)
            if not math.isnan(evdp_value) and not math.isnan(gcarc_value):
                try:
                    completed = subprocess.run(
                        [
                            self.preview_pierce_taup_bin,
                            'time',
                            '--onlytime',
                            '-h',
                            f'{evdp_value:g}',
                            '-p',
                            'P,pP,sP',
                            '--deg',
                            f'{gcarc_value:g}',
                            '--mod',
                            model_key,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    values = []
                    for token in (completed.stdout or '').replace('\n', ' ').split():
                        try:
                            values.append(float(token))
                        except ValueError:
                            continue
                    if len(values) >= 3:
                        p_time, pp_time, sp_time = values[:3]
                except (subprocess.SubprocessError, OSError, ValueError):
                    pass
            cached = {'P': p_time, 'pP': pp_time, 'sP': sp_time}
            self.theory_time_cache[batch_key] = cached
        return {
            'model': model_key,
            'P': p_time,
            'pP': pp_time,
            'sP': sp_time,
            'pP-P': pp_time - p_time if not math.isnan(pp_time) and not math.isnan(p_time) else math.nan,
            'sP-P': sp_time - p_time if not math.isnan(sp_time) and not math.isnan(p_time) else math.nan,
        }

    def _prime_theory_time_cache_from_summary(self, summary):
        if not isinstance(summary, dict):
            return
        per_wave = summary.get('per_wave')
        if not isinstance(per_wave, dict):
            return
        model_key = str(summary.get('model') or self.theory_time_model).lower()
        evdp = _safe_float(summary.get('evdp', math.nan))
        if math.isnan(evdp):
            evdp = _safe_float(getattr(self.wave[0].stats.sac, 'evdp', math.nan)) if getattr(self, 'wave', None) else math.nan
        if math.isnan(evdp):
            return
        gcarc_by_wave = {
            str(wave_name): _safe_float(gcarc)
            for wave_name, gcarc in zip(getattr(self, 'ori_sacnames', []), getattr(self, 'gcarc', []))
        }
        for wave_name, delta_info in per_wave.items():
            gcarc = gcarc_by_wave.get(str(wave_name), math.nan)
            if math.isnan(gcarc) or not isinstance(delta_info, dict):
                continue
            batch_key = self._theory_time_cache_key(model_key, 'P,pP,sP', evdp, gcarc)
            self.theory_time_cache[batch_key] = {
                'P': _safe_float(delta_info.get('P', math.nan)),
                'pP': _safe_float(delta_info.get('pP', math.nan)),
                'sP': _safe_float(delta_info.get('sP', math.nan)),
            }

    def _ensure_event_theory_summary(self, model=None):
        # In stack mode ppk startup skips ensure_event_theory_summary (it only runs
        # for non-stack event dirs), so the source event's summary may not exist yet.
        # Generate it on demand from the source event SAC files so the std export
        # header can show the theory model / mean gcarc / pP-P / sP-P line.
        if not getattr(self, 'stack_mode', False):
            return
        model_key = str(model or self.theory_time_model).lower()
        try:
            import ppk
        except Exception:
            return
        try:
            ppk.ensure_event_theory_summary(
                event_dir=self.runtime_event_dir,
                model=model_key,
                suffix=str(getattr(self, 'suffix', None) or '.sac'),
                taup_bin=self.preview_pierce_taup_bin,
            )
        except Exception as exc:
            print(f'Theory summary generation skipped: {exc}')

    def _event_theory_delta_summary(self, model=None):
        model_key = str(model or self.theory_time_model).lower()
        cached = self.theory_delta_summary_cache
        if cached is not None and cached.get('model') == model_key:
            return cached
        cache_path = self._theory_summary_cache_path(model_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as handle:
                    file_cached = json.load(handle)
                if isinstance(file_cached, dict) and file_cached.get('model') == model_key:
                    if not isinstance(file_cached.get('per_wave'), dict):
                        file_cached['per_wave'] = {}
                    self._prime_theory_time_cache_from_summary(file_cached)
                    self.theory_delta_summary_cache = file_cached
                    return file_cached
            except (OSError, ValueError, TypeError):
                pass
        self._ensure_event_theory_summary(model=model_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as handle:
                    file_cached = json.load(handle)
                if isinstance(file_cached, dict) and file_cached.get('model') == model_key:
                    if not isinstance(file_cached.get('per_wave'), dict):
                        file_cached['per_wave'] = {}
                    self._prime_theory_time_cache_from_summary(file_cached)
                    self.theory_delta_summary_cache = file_cached
                    return file_cached
            except (OSError, ValueError, TypeError):
                pass
        return None

    def _current_wave_theory_delta(self, wave_name=None, model=None):
        summary = self._event_theory_delta_summary(model=model)
        if summary is None:
            return None
        target_wave_name = wave_name or self.current_pick_wave_name
        if not target_wave_name:
            return None
        return summary.get('per_wave', {}).get(target_wave_name)

    def _format_theory_delta_text(self, delta_value):
        if delta_value is None or math.isnan(delta_value):
            return '--'
        return f'{float(delta_value):.2f}s'

    def event_theory_delta_summary_text(self, model=None):
        summary = self._event_theory_delta_summary(model=model)
        if summary is None:
            return ''
        model_key = summary.get('model', str(model or self.theory_time_model).lower())
        return (
            f"Theory[{model_key}] "
            f"pP-P:{self._format_theory_delta_text(summary.get('pP-P_mean', math.nan))}  "
            f"sP-P:{self._format_theory_delta_text(summary.get('sP-P_mean', math.nan))}"
        )

    def current_wave_theory_delta_text(self, wave_name=None, model=None):
        delta_info = self._current_wave_theory_delta(wave_name=wave_name, model=model)
        model_key = str(model or self.theory_time_model).lower()
        if delta_info is None:
            return f"Theory[{model_key}] pP-P:--  sP-P:--"
        return (
            f"Theory[{model_key}] "
            f"pP-P:{self._format_theory_delta_text(delta_info.get('pP-P', math.nan))}  "
            f"sP-P:{self._format_theory_delta_text(delta_info.get('sP-P', math.nan))}"
        )

    def active_wave_theory_delta_text(self, model=None):
        return self.current_wave_theory_delta_text(wave_name=self.current_pick_wave_name, model=model)

    def _std_export_option_value(self, options, key):
        if options is None:
            return self.standard_export_options.get(key, False)
        return bool(options.get(key, self.standard_export_options.get(key, False)))

    def _standard_export_selected_phase_keys(self, options=None):
        raw_phase_keys = None
        if options is not None:
            raw_phase_keys = options.get('phase_keys')
        if raw_phase_keys is None:
            raw_phase_keys = self.standard_export_options.get('phase_keys')
        phase_keys = []
        for marker_key in raw_phase_keys or []:
            key = str(marker_key)
            if key == '4':
                continue
            if key in self.marker_styles and key not in phase_keys:
                phase_keys.append(key)
        if phase_keys:
            return phase_keys
        return [str(i) for i in range(10) if i != 4]

    def _standard_export_selected_axes(self, options=None):
        export_gcarc = self._std_export_option_value(options, 'export_gcarc')
        export_az = self._std_export_option_value(options, 'export_az')
        export_pierce = self._std_export_option_value(options, 'export_pierce')
        export_pierce_group = self._std_export_option_value(options, 'export_pierce_group')
        return {
            'gcarc': export_gcarc,
            'az': export_az,
            'pierce': export_pierce,
            'pierce_group': export_pierce_group,
        }

    def _std_export_bandpass_profile(self, options=None):
        option_profile = None if options is None else options.get('bandpass_profile')
        if option_profile is not None:
            normalized = self._normalize_bandpass_profile(option_profile)
            if normalized is not None:
                return normalized
        profile = self._current_bandpass_profile()
        if profile is not None:
            return profile
        return {
            'freqmin': 0.02,
            'freqmax': 0.2,
            'corners': 2,
            'passes': 1,
        }

    def _standard_export_event_name(self, evtdata=None):
        event_name = self._semantic_event_name()
        if event_name:
            return event_name
        evtname = getattr(evtdata, 'evtname', '') if evtdata is not None else ''
        return str(evtname or '')

    def _standard_export_header_lines(self, evtdata, options=None):
        if evtdata is None:
            return []
        line1_parts = []
        line2_parts = []

        if self._std_export_option_value(options, 'event_name'):
            line1_parts.append(f'Event {self._standard_export_event_name(evtdata)}')
        if self._std_export_option_value(options, 'depth'):
            line1_parts.append(f'Depth {evtdata.evdp:.1f} km')
        if self._std_export_option_value(options, 'bandpass'):
            profile = self._std_export_bandpass_profile(options)
            line1_parts.append(f'bp:{profile["freqmin"]:g}-{profile["freqmax"]:g} n{int(profile["corners"])} p{int(profile["passes"])}')

        theory_summary = self._event_theory_delta_summary()
        if theory_summary is not None:
            evtdata_gcarc = np.asarray(getattr(evtdata, 'gcarc', []), dtype=float)
            gcarc_mean = float(np.mean(evtdata_gcarc)) if evtdata_gcarc.size else math.nan
            gcarc_text = f'gcarc(avg):{gcarc_mean:.2f}°' if not math.isnan(gcarc_mean) else ''
            if self._std_export_option_value(options, 'theory_model'):
                line2_parts.append(f'mod:{theory_summary.get("model", self.theory_time_model)}')
            if self._std_export_option_value(options, 'gcarc_mean') and gcarc_text:
                line2_parts.append(gcarc_text)
            if self._std_export_option_value(options, 'pp_minus_p'):
                line2_parts.append(f'pP-P:{self._format_theory_delta_text(theory_summary.get("pP-P_mean", math.nan))}')
            if self._std_export_option_value(options, 'sp_minus_p'):
                line2_parts.append(f'sP-P:{self._format_theory_delta_text(theory_summary.get("sP-P_mean", math.nan))}')

        header_lines = []
        if line1_parts:
            header_lines.append('   |   '.join(line1_parts))
        if line2_parts:
            header_lines.append('   |   '.join(line2_parts))
        return header_lines

    def _prompt_standard_export_options(self, parent_window=None):
        dialog = QDialog(parent_window)
        dialog.setWindowTitle('Std Export Options')
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        checkboxes = {}
        option_items = [
            ('event_name', 'Event name'),
            ('depth', 'Depth'),
            ('bandpass', 'Bandpass'),
            ('theory_model', 'Theory model'),
            ('gcarc_mean', 'Mean gcarc'),
            ('pp_minus_p', 'Mean pP-P'),
            ('sp_minus_p', 'Mean sP-P'),
            ('phase_legend', 'Phase legend'),
        ]
        for key, label in option_items:
            checkbox = QCheckBox(label, dialog)
            checkbox.setChecked(bool(self.standard_export_options.get(key, False)))
            checkboxes[key] = checkbox
            form.addRow(checkbox)

        header_all_checkbox = QCheckBox('Header all', dialog)
        header_all_checkbox.setChecked(all(box.isChecked() for box in checkboxes.values()))
        form.addRow(header_all_checkbox)

        selected_phase_keys = self._standard_export_selected_phase_keys()
        phase_group = QGroupBox('Std phase dots/labels', dialog)
        phase_layout = QGridLayout(phase_group)
        phase_checkboxes = {}
        phase_all_checkbox = QCheckBox('Phase all', phase_group)
        display_phase_keys = [str(i) for i in range(10) if i != 4]
        phase_all_checkbox.setChecked(all(key in selected_phase_keys for key in display_phase_keys))
        phase_layout.addWidget(phase_all_checkbox, 0, 0, 1, 5)
        for marker_index in range(10):
            marker_key = str(marker_index)
            if marker_key == '4':
                continue
            phase_box = QCheckBox(f't{marker_key}', phase_group)
            phase_box.setChecked(marker_key in selected_phase_keys)
            phase_checkboxes[marker_key] = phase_box
            display_index = len(phase_checkboxes) - 1
            row = display_index // 5 + 1
            col = display_index % 5
            phase_layout.addWidget(phase_box, row, col)

        axis_group = QGroupBox('Std panels', dialog)
        axis_layout = QVBoxLayout(axis_group)
        axis_checkboxes = {}
        axis_items = [
            ('export_gcarc', 'Epicenter distance'),
            ('export_az', 'Azimuth'),
            ('export_pierce', 'Pierce map'),
            ('export_pierce_group', 'Pierce map (by group)'),
        ]
        for key, label in axis_items:
            checkbox = QCheckBox(label, axis_group)
            checkbox.setChecked(bool(self.standard_export_options.get(key, False)))
            axis_checkboxes[key] = checkbox
            axis_layout.addWidget(checkbox)

        def sync_header_all_checkbox():
            header_all_checkbox.blockSignals(True)
            try:
                header_all_checkbox.setChecked(all(box.isChecked() for box in checkboxes.values()))
            finally:
                header_all_checkbox.blockSignals(False)

        def sync_phase_all_checkbox():
            phase_all_checkbox.blockSignals(True)
            try:
                phase_all_checkbox.setChecked(all(box.isChecked() for box in phase_checkboxes.values()))
            finally:
                phase_all_checkbox.blockSignals(False)

        def set_all_header_checkboxes(checked):
            for box in checkboxes.values():
                box.setChecked(bool(checked))

        def set_all_phase_checkboxes(checked):
            for box in phase_checkboxes.values():
                box.setChecked(bool(checked))

        header_all_checkbox.toggled.connect(set_all_header_checkboxes)
        phase_all_checkbox.toggled.connect(set_all_phase_checkboxes)
        for box in checkboxes.values():
            box.toggled.connect(sync_header_all_checkbox)
        for box in phase_checkboxes.values():
            box.toggled.connect(sync_phase_all_checkbox)

        layout.addLayout(form)
        layout.addWidget(axis_group)
        layout.addWidget(phase_group)

        bp_profile_options = []
        seen_bp_keys = set()

        def add_bp_option(profile):
            normalized = self._normalize_bandpass_profile(profile)
            if normalized is None:
                return
            key = self._bandpass_profile_key(normalized)
            if key in seen_bp_keys:
                return
            seen_bp_keys.add(key)
            bp_profile_options.append(normalized)

        add_bp_option({'freqmin': 0.02, 'freqmax': 0.2, 'corners': 2, 'passes': 1})
        add_bp_option(self._current_bandpass_profile())
        for preset in getattr(self, 'compare_preset_profiles', []):
            add_bp_option(preset)
        for preset in getattr(self, 'compare_default_bandpass_profiles', []):
            add_bp_option(preset)

        bp_combo = QComboBox(dialog)
        selected_profile = self._std_export_bandpass_profile(self.standard_export_options)
        selected_bp_index = 0
        for index, profile in enumerate(bp_profile_options):
            bp_combo.addItem(self._format_bandpass_label(profile), profile)
            if self._bandpass_profile_key(profile) == self._bandpass_profile_key(selected_profile):
                selected_bp_index = index
        if bp_profile_options:
            bp_combo.setCurrentIndex(selected_bp_index)
        form.addRow('Std BP', bp_combo)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return None
        selected_options = {
            key: checkbox.isChecked()
            for key, checkbox in checkboxes.items()
        }
        selected_options.update({
            key: checkbox.isChecked()
            for key, checkbox in axis_checkboxes.items()
        })
        selected_options['phase_keys'] = [
            marker_key for marker_key, checkbox in phase_checkboxes.items()
            if checkbox.isChecked()
        ]
        if bp_profile_options:
            selected_options['bandpass_profile'] = self._normalize_bandpass_profile(
                bp_combo.currentData()
            )
        self.standard_export_options.update(selected_options)
        return selected_options

    def _prompt_preview_stack_options(self, preview_index, parent_window=None):
        if preview_index >= len(self.preview_modes):
            return None
        tmarker, x1, x2 = self.preview_modes[preview_index]
        dialog = QDialog(parent_window)
        dialog.setWindowTitle('Preview Stack Options')
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        scope_combo = QComboBox(dialog)
        scope_combo.addItem('Visible', 'visible')
        scope_combo.addItem('Selected', 'selected')
        group_names = self._list_preview_group_names_for_stack()
        for group_name in group_names:
            scope_combo.addItem(f'Group {group_name}', f'group:{group_name}')
        _set_default_scope = self._preview_stack_default_scope(preview_index)
        for index in range(scope_combo.count()):
            if scope_combo.itemData(index) == _set_default_scope:
                scope_combo.setCurrentIndex(index)
                break
        form.addRow('Scope', scope_combo)

        saved_config_combo = QComboBox(dialog)
        saved_config_combo.addItem('Current / New', None)
        saved_config_delete_button = QPushButton('Delete config', dialog)
        saved_config_delete_button.setEnabled(False)
        saved_config_row = QHBoxLayout()
        saved_config_row.addWidget(saved_config_combo, stretch=1)
        saved_config_row.addWidget(saved_config_delete_button)
        form.addRow('Saved config', saved_config_row)

        marker_combo = QComboBox(dialog)
        for marker_key in sorted(self.marker_styles.keys(), key=int):
            if marker_key == '4':
                continue
            marker_combo.addItem(f't{marker_key}', marker_key)
        current_marker_index = marker_combo.findText(f't{tmarker}')
        if current_marker_index >= 0:
            marker_combo.setCurrentIndex(current_marker_index)
        form.addRow('Align marker', marker_combo)

        x1_edit = QLineEdit(dialog)
        x1_edit.setText(f'{float(x1):g}')
        x2_edit = QLineEdit(dialog)
        x2_edit.setText(f'{float(x2):g}')
        form.addRow('Window x1', x1_edit)
        form.addRow('Window x2', x2_edit)

        polarity_combo = QComboBox(dialog)
        polarity_combo.addItem('Apply user4 flips', 'apply_user4')
        polarity_combo.addItem('Keep as is', 'keep')
        polarity_combo.addItem('Reject mixed polarity', 'reject_mixed')
        form.addRow('Polarity', polarity_combo)

        normalize_combo = QComboBox(dialog)
        normalize_combo.addItem('RMS', 'rms')
        normalize_combo.addItem('Peak', 'peak')
        normalize_combo.addItem('Off', 'off')
        form.addRow('Normalize', normalize_combo)

        stack_type_combo = QComboBox(dialog)
        stack_type_combo.addItem('Linear', 'linear')
        stack_type_combo.addItem('PWS', 'pws')
        stack_type_combo.addItem('SMatStack', 'smatstack')
        form.addRow('Stack type', stack_type_combo)

        smatstack_max_shift_edit = QLineEdit(dialog)
        smatstack_max_shift_edit.setText('5')
        form.addRow('SMat max shift (s)', smatstack_max_shift_edit)

        moveout_combo = QComboBox(dialog)
        moveout_combo.addItem('Off', 'off')
        moveout_combo.addItem('Theory pP moveout', 'phase')
        moveout_combo.addItem('Theory sP moveout', 'phase_s')
        form.addRow('Moveout', moveout_combo)

        label_edit = QLineEdit(dialog)
        label_edit.setPlaceholderText('optional, e.g. pp_precursor_test1')
        form.addRow('Output label', label_edit)

        note_label = QLabel('Default remains Linear. SMatStack searches small shifts inside the stack window; moveout is theory-based.', dialog)
        note_label.setStyleSheet('color: #666666;')

        def _set_combo_to_data(combo, target_data):
            for index in range(combo.count()):
                if combo.itemData(index) == target_data:
                    combo.setCurrentIndex(index)
                    return

        def _apply_saved_stack_config(saved_options):
            if not isinstance(saved_options, dict):
                return
            align_marker = saved_options.get('align_marker')
            if align_marker is not None:
                _set_combo_to_data(marker_combo, str(align_marker))
            x1_value = saved_options.get('x1')
            x2_value = saved_options.get('x2')
            if x1_value is not None:
                x1_edit.setText(f'{float(x1_value):g}')
            if x2_value is not None:
                x2_edit.setText(f'{float(x2_value):g}')
            _set_combo_to_data(polarity_combo, str(saved_options.get('polarity', 'apply_user4')))
            _set_combo_to_data(normalize_combo, str(saved_options.get('normalize', 'rms')))
            _set_combo_to_data(stack_type_combo, str(saved_options.get('stack_type', 'linear')))
            smatstack_max_shift_edit.setText(f"{float(saved_options.get('smatstack_max_shift_s', 5.0)):g}")
            moveout_mode = str(saved_options.get('moveout_mode', 'off')).strip().lower()
            moveout_phase = self._normalize_marker_key(saved_options.get('moveout_phase', ''))
            if moveout_mode == 'phase' and moveout_phase == '3':
                _set_combo_to_data(moveout_combo, 'phase_s')
            elif moveout_mode == 'phase':
                _set_combo_to_data(moveout_combo, 'phase')
            else:
                _set_combo_to_data(moveout_combo, 'off')
            label_edit.setText(str(saved_options.get('label', '') or ''))

        def _refresh_saved_stack_configs():
            scope_value = scope_combo.currentData()
            previous = saved_config_combo.blockSignals(True)
            try:
                saved_config_combo.clear()
                saved_config_combo.addItem('Current / New', None)
                if not isinstance(scope_value, str) or not scope_value.startswith('group:'):
                    return
                group_name = scope_value.split(':', 1)[1]
                saved_options = self._saved_stack_options_for_group(group_name)
                if not saved_options:
                    return
                for idx, options in enumerate(saved_options, start=1):
                    align_text = f"t{self._normalize_marker_key(options.get('align_marker'))}"
                    x1_text = f"{float(options.get('x1')):g}"
                    x2_text = f"{float(options.get('x2')):g}"
                    stack_text = str(options.get('stack_type', 'linear')).upper()
                    normalize_text = str(options.get('normalize', 'rms')).upper()
                    label_text = str(options.get('label', '') or '').strip()
                    summary = f'{idx}: {align_text} {x1_text}/{x2_text} {stack_text}/{normalize_text}'
                    if str(options.get('stack_type', 'linear')).lower() == 'smatstack':
                        summary += f" smax={float(options.get('smatstack_max_shift_s', 5.0)):g}s"
                    if label_text:
                        summary += f' [{label_text}]'
                    saved_config_combo.addItem(summary, options)
                default_index = self._saved_stack_config_default_combo_index(saved_options)
                saved_config_combo.setCurrentIndex(default_index)
                _apply_saved_stack_config(saved_config_combo.currentData())
                if len(saved_options) == 1:
                    note_label.setText(f'Loaded saved config for {group_name}; edit and Stack to overwrite this group.')
                else:
                    note_label.setText(f'Loaded first saved config for {group_name}; switch Saved config if needed.')
                saved_config_delete_button.setEnabled(saved_config_combo.currentData() is not None)
            finally:
                saved_config_combo.blockSignals(previous)

        def _on_saved_config_changed(_index):
            saved_options = saved_config_combo.currentData()
            saved_config_delete_button.setEnabled(saved_options is not None)
            if saved_options is None:
                return
            _apply_saved_stack_config(saved_options)

        def _delete_current_saved_config():
            saved_options = saved_config_combo.currentData()
            if not isinstance(saved_options, dict):
                note_label.setText('Choose a saved config before deleting.')
                return
            stack_wave_name = str(saved_options.get('stack_wave_name') or '').strip()
            if not stack_wave_name:
                note_label.setText('Saved config is missing a stack wave name.')
                return
            reply = QMessageBox.question(
                dialog,
                'Delete stack config',
                f'Delete {stack_wave_name} and its saved analysis package?',
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            stack_event_dir = self.wavepath if getattr(self, 'stack_mode', False) else self._stack_data_event_directory()
            result = delete_stack_config(stack_event_dir, stack_wave_name)
            self.stack_sidecars = load_stack_sidecar_map(stack_event_dir)
            _refresh_saved_stack_configs()
            removed_count = len(result.get('removed', []) or [])
            if removed_count:
                note_label.setText(f'Deleted {stack_wave_name} ({removed_count} artifacts).')
            else:
                note_label.setText(f'No saved artifacts found for {stack_wave_name}.')

        scope_combo.currentIndexChanged.connect(lambda _index: _refresh_saved_stack_configs())
        saved_config_combo.currentIndexChanged.connect(_on_saved_config_changed)
        saved_config_delete_button.clicked.connect(_delete_current_saved_config)
        _refresh_saved_stack_configs()

        layout.addLayout(form)
        layout.addWidget(note_label)
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=dialog,
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        if dialog.exec() != int(QDialog.DialogCode.Accepted):
            return None
        return {
            'scope': scope_combo.currentData(),
            'align_marker': marker_combo.currentData(),
            'x1': x1_edit.text(),
            'x2': x2_edit.text(),
            'polarity': polarity_combo.currentData(),
            'normalize': normalize_combo.currentData(),
            'stack_type': stack_type_combo.currentData(),
            'smatstack_max_shift_s': smatstack_max_shift_edit.text(),
            'moveout_mode': 'phase' if moveout_combo.currentData() in ('phase', 'phase_s') else 'off',
            'moveout_phase': '2' if moveout_combo.currentData() == 'phase' else ('3' if moveout_combo.currentData() == 'phase_s' else ''),
            'label': label_edit.text(),
        }

    def _standard_export_pierce_records_from_evtdata(self, evtdata):
        if evtdata is None:
            return []
        records = self._load_pierce_points_for_current_event(auto_generate=True)
        pierce_records = []
        for tr in getattr(evtdata, 'wave_ori', []):
            wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
            if not wave_name:
                continue
            record = records.get(wave_name)
            if record is None:
                record = self._stack_sidecar_pierce_record(wave_name)
            if record is not None:
                pierce_records.append(record)
        return pierce_records

    def _standard_pierce_record_color(self, wave_name):
        if (
            not self._is_user1_wave(wave_name)
            and not self._is_user5_wave(wave_name)
            and not self._is_user4_wave(wave_name)
            and not self._is_preview_purple_wave(wave_name)
        ):
            return '#111111'
        base_color, _selected_color = self._pierce_record_style(wave_name, selected=False)
        return base_color

    def _standard_pierce_record_label(self, wave_name):
        if not getattr(self, 'stack_mode', False):
            return ''
        if not self._stack_wave_summary(wave_name):
            return ''
        return 'STACK'

    def _standard_pierce_plot_region(self):
        return (-32.0, -23.0, -61.0, -55.0)

    def _standard_pierce_cache_signature(self, evtdata, pierce_records, export_options=None):
        payload = {
            'event': os.path.basename(os.path.abspath(self._semantic_event_dir())) if evtdata is not None else '',
            'phase': str(getattr(self, 'preview_pierce_phase', '')),
            'model': str(getattr(self, 'preview_pierce_model', '')),
            'header': list(self._standard_export_header_lines(evtdata, export_options)) if evtdata is not None else [],
            'event_lon': float(getattr(evtdata, 'evlo', math.nan)) if evtdata is not None else math.nan,
            'event_lat': float(getattr(evtdata, 'evla', math.nan)) if evtdata is not None else math.nan,
            'records': [
                {
                    'wave_name': str(getattr(record, 'wave_name', '')),
                    'lon': round(float(record.longitude), 6),
                    'lat': round(float(record.latitude), 6),
                    'color': self._standard_pierce_record_color(getattr(record, 'wave_name', '')),
                    'flip': bool(self._is_user4_wave(getattr(record, 'wave_name', ''))),
                    'label': self._standard_pierce_record_label(getattr(record, 'wave_name', '')),
                }
                for record in pierce_records or []
            ],
        }
        signature_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(signature_text.encode('utf-8')).hexdigest()[:16]

    def _try_reuse_standard_pierce_cache(self, output_dir, signature):
        if not signature:
            return None, None
        cache_main = os.path.join(output_dir, f'pierce_cache_{signature}.png')
        cache_zoom = os.path.join(output_dir, f'pierce_zoom_cache_{signature}.png')
        if os.path.isfile(cache_main):
            zoom_path = cache_zoom if os.path.isfile(cache_zoom) else None
            return cache_main, zoom_path
        return None, None

    def _store_standard_pierce_cache(self, output_path, zoom_output_path, output_dir, signature):
        if not signature or not output_path or not os.path.isfile(output_path):
            return
        cache_main = os.path.join(output_dir, f'pierce_cache_{signature}.png')
        try:
            if os.path.abspath(output_path) != os.path.abspath(cache_main):
                shutil.copy2(output_path, cache_main)
        except OSError:
            pass
        if zoom_output_path and os.path.isfile(zoom_output_path):
            cache_zoom = os.path.join(output_dir, f'pierce_zoom_cache_{signature}.png')
            try:
                if os.path.abspath(zoom_output_path) != os.path.abspath(cache_zoom):
                    shutil.copy2(zoom_output_path, cache_zoom)
            except OSError:
                pass

    def _copy_standard_pierce_cache_to_timestamped_outputs(self, cached_output_path, cached_zoom_path, output_dir, timestamp_tag):
        if not cached_output_path or not os.path.isfile(cached_output_path):
            return None, None
        output_path = os.path.join(output_dir, f"pierce_{timestamp_tag}.png")
        zoom_output_path = os.path.join(output_dir, f"pierce_zoom_{timestamp_tag}.png")
        try:
            if os.path.abspath(cached_output_path) != os.path.abspath(output_path):
                shutil.copy2(cached_output_path, output_path)
            else:
                output_path = cached_output_path
        except OSError:
            output_path = cached_output_path
        if cached_zoom_path and os.path.isfile(cached_zoom_path):
            try:
                if os.path.abspath(cached_zoom_path) != os.path.abspath(zoom_output_path):
                    shutil.copy2(cached_zoom_path, zoom_output_path)
                else:
                    zoom_output_path = cached_zoom_path
            except OSError:
                zoom_output_path = cached_zoom_path
        else:
            zoom_output_path = None
        return output_path, zoom_output_path

    def _standard_pierce_plot_script_path(self):
        return os.path.join(os.path.dirname(__file__), 'plot_standard_pierce_map.sh')

    def _standard_pierce_cpt_path(self):
        return os.path.join(os.path.dirname(__file__), 'cpt', 'south_sandwich_reference.cpt')

    def _save_standard_pierce_main_plot(self, evtdata, pierce_records, output_dir, timestamp_tag, export_options=None):
        if evtdata is None or evtdata.sta_num == 0 or not pierce_records:
            return None
        lon_min, lon_max, lat_min, lat_max = self._standard_pierce_plot_region()
        fig = plt.figure(figsize=(8.6, 10.8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Lon', fontsize=12)
        ax.set_ylabel('Lat', fontsize=12)
        ax.grid(color='gray', linestyle='--', linewidth=0.3, axis='both', alpha=0.35)
        ax.set_title(f'{self.preview_pierce_phase} {self.preview_pierce_model}', fontsize=11)

        normal_lons = []
        normal_lats = []
        normal_colors = []
        flip_lons = []
        flip_lats = []
        flip_colors = []
        for record in pierce_records:
            color = self._standard_pierce_record_color(record.wave_name)
            if self._is_user4_wave(record.wave_name):
                flip_lons.append(float(record.longitude))
                flip_lats.append(float(record.latitude))
                flip_colors.append(color)
            else:
                normal_lons.append(float(record.longitude))
                normal_lats.append(float(record.latitude))
                normal_colors.append(color)

        if normal_lons:
            ax.scatter(
                normal_lons,
                normal_lats,
                s=24,
                c=normal_colors,
                alpha=0.9,
                edgecolors='white',
                linewidths=0.25,
                zorder=3,
            )
        if flip_lons:
            ax.scatter(
                flip_lons,
                flip_lats,
                s=30,
                c=flip_colors,
                alpha=0.92,
                marker='^',
                edgecolors='white',
                linewidths=0.3,
                zorder=4,
            )

        for record in pierce_records:
            label_text = self._standard_pierce_record_label(record.wave_name)
            if not label_text:
                continue
            ax.text(
                float(record.longitude),
                float(record.latitude) + 0.035,
                label_text,
                fontsize=6.2,
                color='#111111',
                ha='center',
                va='bottom',
                fontweight='bold',
                zorder=5,
            )

        ax.scatter(
            [float(evtdata.evlo)],
            [float(evtdata.evla)],
            marker='*',
            s=150,
            c='red',
            edgecolors='black',
            linewidths=0.6,
            zorder=6,
        )

        header_lines = self._standard_export_header_lines(evtdata, export_options)
        if header_lines:
            fig.subplots_adjust(top=0.92)
            axes_box = ax.get_position()
            title_x = axes_box.x0 + axes_box.width * 0.5
            fig.text(
                title_x,
                0.975,
                '\n'.join(header_lines),
                fontsize=14,
                ha='center',
                va='top',
                linespacing=1.18,
            )
        else:
            fig.subplots_adjust(top=0.965)

        output_path = os.path.join(output_dir, f"pierce_{timestamp_tag}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def _apply_standard_header_to_image(self, image_path, header_lines):
        if not header_lines:
            return image_path
        image = plt.imread(image_path)
        height, width = image.shape[:2]
        figure_width = 8.6
        image_height = figure_width * (float(height) / float(width))
        figure_height = image_height + 0.95
        fig = plt.figure(figsize=(figure_width, figure_height))
        _force_qt_arrow_cursor_for_figure(fig)
        ax = fig.add_axes([0.04, 0.06, 0.92, image_height / figure_height])
        ax.imshow(image)
        ax.axis('off')
        axes_box = ax.get_position()
        title_x = axes_box.x0 + axes_box.width * 0.5
        fig.text(
            title_x,
            0.975,
            '\n'.join(header_lines),
            fontsize=14,
            ha='center',
            va='top',
            linespacing=1.18,
        )
        fig.savefig(image_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return image_path

    def _standard_pierce_zoom_bounds(self, evtdata, pierce_records):
        event_lon = float(evtdata.evlo)
        event_lat = float(evtdata.evla)
        all_lons = [event_lon]
        all_lats = [event_lat]
        if pierce_records:
            all_lons.extend(float(record.longitude) for record in pierce_records)
            all_lats.extend(float(record.latitude) for record in pierce_records)
        lon_min = min(all_lons)
        lon_max = max(all_lons)
        lat_min = min(all_lats)
        lat_max = max(all_lats)
        lon_pad = max(0.12, (lon_max - lon_min) * 0.08 if lon_max > lon_min else 0.2)
        lat_pad = max(0.12, (lat_max - lat_min) * 0.08 if lat_max > lat_min else 0.2)
        return (
            lon_min - lon_pad,
            lon_max + lon_pad,
            lat_min - lat_pad,
            lat_max + lat_pad,
        )

    def _save_standard_pierce_zoom_plot(self, evtdata, pierce_records, output_dir, timestamp_tag, export_options=None):
        if evtdata is None or evtdata.sta_num == 0 or not pierce_records:
            return None
        fig = plt.figure(figsize=(8.6, 10.8))
        ax = fig.add_subplot(1, 1, 1)
        lon_min, lon_max, lat_min, lat_max = self._standard_pierce_zoom_bounds(evtdata, pierce_records)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Lon', fontsize=12)
        ax.set_ylabel('Lat', fontsize=12)
        ax.grid(color='gray', linestyle='--', linewidth=0.3, axis='both', alpha=0.45)
        ax.set_title(f'{self.preview_pierce_phase} {self.preview_pierce_model} zoom', fontsize=11)

        longitudes = np.asarray([float(record.longitude) for record in pierce_records], dtype=float)
        latitudes = np.asarray([float(record.latitude) for record in pierce_records], dtype=float)
        base_colors = [self._pierce_record_style(record.wave_name, selected=False)[0] for record in pierce_records]
        overlap_counts = Counter(
            (round(float(record.longitude), 6), round(float(record.latitude), 6))
            for record in pierce_records
        )
        point_sizes = []
        for lon_value, lat_value in zip(longitudes, latitudes):
            repeat_count = overlap_counts[(round(float(lon_value), 6), round(float(lat_value), 6))]
            if repeat_count <= 1:
                point_sizes.append(48.0)
            else:
                point_sizes.append(min(180.0, 72.0 + 12.0 * min(repeat_count, 12)))
        point_sizes = np.asarray(point_sizes, dtype=float)
        ax.scatter(
            longitudes,
            latitudes,
            s=point_sizes,
            c=base_colors,
            alpha=0.9,
            edgecolors='white',
            linewidths=0.35,
            zorder=3,
        )
        for (lon_key, lat_key), count in overlap_counts.items():
            if count <= 1:
                continue
            ax.text(
                float(lon_key),
                float(lat_key),
                str(count),
                fontsize=6.2,
                color='white',
                ha='center',
                va='center',
                fontweight='bold',
                zorder=4,
            )
        for record in pierce_records:
            label_text = self._standard_pierce_record_label(record.wave_name)
            if not label_text:
                continue
            ax.text(
                float(record.longitude),
                float(record.latitude) + 0.035,
                label_text,
                fontsize=6.4,
                color='#111111',
                ha='center',
                va='bottom',
                fontweight='bold',
                zorder=5,
            )
        ax.scatter(
            [float(evtdata.evlo)],
            [float(evtdata.evla)],
            marker='*',
            s=150,
            c='red',
            edgecolors='black',
            linewidths=0.6,
            zorder=5,
        )

        header_lines = self._standard_export_header_lines(evtdata, export_options)
        if header_lines:
            fig.subplots_adjust(top=0.92)
            axes_box = ax.get_position()
            title_x = axes_box.x0 + axes_box.width * 0.5
            fig.text(
                title_x,
                0.975,
                '\n'.join(header_lines),
                fontsize=14,
                ha='center',
                va='top',
                linespacing=1.18,
            )
        else:
            fig.subplots_adjust(top=0.965)

        output_path = os.path.join(output_dir, f"pierce_zoom_{timestamp_tag}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def _standard_pierce_group_number(self, wave_name):
        # Resolve the group number for a pierce record. Stack wave names embed
        # the group (stack_groupN_...); member wave names do not, so fall back to
        # the group of the stack wave currently being previewed.
        group_number = self._group_number_from_wave_name(wave_name)
        if group_number is not None:
            return group_number
        if getattr(self, 'stack_mode', False):
            stack_wave_name = getattr(self.plotfig, '_stack_preview_wave_name', None) if self.plotfig is not None else None
            if not stack_wave_name:
                stack_wave_name = self._current_stack_preview_wave_name()
            if stack_wave_name:
                return self._group_number_from_wave_name(stack_wave_name)
        return None

    def _save_standard_pierce_group_plot(self, evtdata, pierce_records, output_dir, timestamp_tag, export_options=None):
        # Pierce map colored by stack group (GRP#). Each record is assigned the
        # palette color of its group; a legend lists the groups present. Uses the
        # Python matplotlib renderer (like the zoom plot) so it stays fast and does
        # not depend on GMT.
        if evtdata is None or evtdata.sta_num == 0 or not pierce_records:
            return None
        fig = plt.figure(figsize=(9.2, 11.0))
        ax = fig.add_subplot(1, 1, 1)
        lon_min, lon_max, lat_min, lat_max = self._standard_pierce_zoom_bounds(evtdata, pierce_records)
        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_xlabel('Lon', fontsize=12)
        ax.set_ylabel('Lat', fontsize=12)
        ax.grid(color='gray', linestyle='--', linewidth=0.3, axis='both', alpha=0.45)
        ax.set_title(
            f'{self.preview_pierce_phase} {self.preview_pierce_model} by group',
            fontsize=11,
        )

        # Group records by group number; records without a group go to None.
        grouped = {}
        for record in pierce_records:
            group_number = self._standard_pierce_group_number(record.wave_name)
            grouped.setdefault(group_number, []).append(record)

        overlap_counts = Counter(
            (round(float(record.longitude), 6), round(float(record.latitude), 6))
            for record in pierce_records
        )
        legend_handles = []
        for group_number in sorted(grouped.keys(), key=lambda value: (value is None, value if value is not None else 0)):
            records = grouped[group_number]
            if group_number is None:
                group_color = '#888888'
                group_label = 'ungrouped'
            else:
                group_color = self._preview_group_color(f'group{group_number}')
                group_label = f'GRP{group_number}'
            longitudes = np.asarray([float(record.longitude) for record in records], dtype=float)
            latitudes = np.asarray([float(record.latitude) for record in records], dtype=float)
            point_sizes = []
            for record in records:
                repeat_count = overlap_counts[(round(float(record.longitude), 6), round(float(record.latitude), 6))]
                point_sizes.append(48.0 if repeat_count <= 1 else min(180.0, 72.0 + 12.0 * min(repeat_count, 12)))
            ax.scatter(
                longitudes,
                latitudes,
                s=np.asarray(point_sizes, dtype=float),
                c=group_color,
                alpha=0.9,
                edgecolors='white',
                linewidths=0.35,
                zorder=3,
                label=group_label,
            )
            legend_handles.append((group_label, group_color))

        ax.scatter(
            [float(evtdata.evlo)],
            [float(evtdata.evla)],
            marker='*',
            s=150,
            c='red',
            edgecolors='black',
            linewidths=0.6,
            zorder=5,
            label='Event',
        )

        if legend_handles:
            ax.legend(loc='best', fontsize=8, framealpha=0.9)

        header_lines = self._standard_export_header_lines(evtdata, export_options)
        if header_lines:
            fig.subplots_adjust(top=0.92)
            axes_box = ax.get_position()
            title_x = axes_box.x0 + axes_box.width * 0.5
            fig.text(
                title_x,
                0.975,
                '\n'.join(header_lines),
                fontsize=14,
                ha='center',
                va='top',
                linespacing=1.18,
            )
        else:
            fig.subplots_adjust(top=0.965)

        output_path = os.path.join(output_dir, f"pierce_group_{timestamp_tag}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path

    def _save_standard_phase_legend_image(self, phase_keys, output_dir, timestamp_tag):
        unique_phase_keys = []
        for marker_key in phase_keys:
            key = str(marker_key)
            if key in self.marker_styles and key not in unique_phase_keys:
                unique_phase_keys.append(key)
        if not unique_phase_keys:
            return None

        row_count = len(unique_phase_keys)
        fig_height = max(2.6, 0.52 * row_count + 0.35)
        fig, ax = plt.subplots(figsize=(2.2, fig_height))
        ax.set_xlim(0, 1.8)
        ax.set_ylim(0, row_count)
        ax.axis('off')

        for row_index, marker_key in enumerate(unique_phase_keys):
            y = row_count - row_index - 0.5
            _marker_name, color = self.marker_styles[marker_key]
            label_text = self._standard_phase_legend_label(marker_key)
            rect = plt.matplotlib.patches.FancyBboxPatch(
                (0.22, y - 0.12), 0.38, 0.24,
                boxstyle='round,pad=0.02,rounding_size=0.06',
                linewidth=0,
                facecolor=color,
                edgecolor='none',
                transform=ax.transData,
                clip_on=False,
            )
            ax.add_patch(rect)
            ax.text(
                0.72, y, label_text,
                fontsize=17.5, color='#111111', va='center', ha='left', fontweight='bold'
            )

        output_path = os.path.join(output_dir, f'phase_legend_{timestamp_tag}.png')
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        return output_path

    def _capture_marker_state(self):
        return {
            'markers': {
                key: dict(value)
                for key, value in self.markers.items()
            },
            'user_markers': {
                key: dict(value)
                for key, value in self.user_markers.items()
            },
            'dirty_marker_wave_names': set(getattr(self, 'dirty_marker_wave_names', set())),
        }

    def _restore_marker_state(self, state):
        if not state:
            return
        self.markers = {
            key: dict(value)
            for key, value in state.get('markers', {}).items()
        }
        self.user_markers = {
            key: dict(value)
            for key, value in state.get('user_markers', {}).items()
        }
        for marker_key in self.marker_styles:
            self.markers.setdefault(marker_key, {})
        for user_key in ('user1', 'user2', 'user3', 'user4', 'user5'):
            self.user_markers.setdefault(user_key, {})
        self.dirty_marker_wave_names = set(state.get('dirty_marker_wave_names', set()))
        for wave_index, wave_name in enumerate(getattr(self, 'ori_sacnames', [])):
            for marker_key in self.marker_styles:
                marker_time = self.markers.get(marker_key, {}).get(wave_name, math.nan)
                marker_attr = f't{marker_key}'
                if hasattr(self, marker_attr):
                    marker_array = getattr(self, marker_attr)
                    if wave_index < len(marker_array):
                        marker_array[wave_index] = marker_time

    def set_compare_preset_library(self, preset_profiles, default_profiles=None):
        normalized_presets = []
        preset_keys = set()
        for item in preset_profiles or []:
            normalized = self._normalize_bandpass_profile(item)
            if normalized is None:
                continue
            key = self._bandpass_profile_key(normalized)
            if key in preset_keys:
                continue
            normalized_presets.append(normalized)
            preset_keys.add(key)
        normalized_defaults = []
        default_keys = set()
        for item in default_profiles or []:
            normalized = self._normalize_bandpass_profile(item)
            if normalized is None:
                continue
            key = self._bandpass_profile_key(normalized)
            if key not in preset_keys or key in default_keys:
                continue
            normalized_defaults.append(normalized)
            default_keys.add(key)
        self.compare_preset_profiles = normalized_presets
        self.compare_default_bandpass_profiles = normalized_defaults[:max(0, self.max_compare_columns - 1)]
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            preview_index = self.comparefig._compare_state.get('preview_index', 0)
            self.plot_compare_preview(preview_index)

    def add_current_bandpass_to_compare(self):
        profile = self._current_bandpass_profile()
        if profile is None:
            print("Current BP is invalid; compare list unchanged.")
            return False
        new_key = self._bandpass_profile_key(profile)
        for existing in self.compare_bandpass_profiles:
            if self._bandpass_profile_key(existing) == new_key:
                print(f"Compare BP already exists: {self._format_bandpass_label(profile)}")
                return False
        self.compare_bandpass_profiles.append(profile)
        print(f"Added compare BP: {self._format_bandpass_label(profile)}")
        return True

    def clear_compare_bandpasses(self):
        self.compare_bandpass_profiles = []
        print("Cleared compare BP list.")

    def _default_compare_profiles(self):
        if self.compare_default_bandpass_profiles:
            return list(self.compare_default_bandpass_profiles[:max(0, self.max_compare_columns - 1)])
        current_profile = self._current_bandpass_profile()
        if current_profile is None:
            return []
        return [current_profile]

    def _ensure_compare_profiles(self):
        if not self.compare_bandpass_profiles:
            self.compare_bandpass_profiles = self._default_compare_profiles()
        return self.compare_bandpass_profiles

    def set_bandpass_settings(self, freqmin=None, freqmax=None, corners=2, passes=2):
        self.bandpass_settings = {
            'freqmin': freqmin,
            'freqmax': freqmax,
            'corners': int(corners),
            'passes': int(passes),
        }
        if hasattr(self, 'wave_raw'):
            self.wave = self._filtered_stream_copy(self.wave_raw)
            self.refresh_current_page()
            preview_index = 0
            if self.plotfig is not None and plt.fignum_exists(self.plotfig.number):
                preview_controls = getattr(self.plotfig, '_preview_controls', {})
                if preview_controls:
                    preview_index = preview_controls.get('preview_index', 0)
                self._refresh_preview_figure(self.plotfig, preview_index)
            if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
                self.plot_compare_preview(preview_index)

    def apply_sac_bandpass_and_reload(self, freqmin=None, freqmax=None, corners=2, passes=2, order='gcarc'):
        if freqmin is None or freqmax is None:
            return False, 'BP requires both freqmin and freqmax'
        if freqmin <= 0 or freqmax <= freqmin:
            return False, 'BP requires freqmax > freqmin > 0'
        if not self._wave_files_for_suffix():
            return False, 'No SAC files matched current suffix'

        previous_state = self._capture_marker_state()
        previous_page = self.ipage
        previous_wave_name = self.current_pick_wave_name
        previous_station_name = self.current_pick_station_name
        previous_hidden = set(self.preview_hidden_wave_names)
        previous_selected = set(self.preview_selected_wave_names)
        previous_jump = self.preview_jump_highlight_wave_name
        preview_index = 0
        if self.plotfig is not None and hasattr(self.plotfig, '_preview_controls'):
            preview_index = getattr(self.plotfig, '_preview_controls', {}).get('preview_index', 0)

        sac_script = '\n'.join([
            'wild echo off',
            'r *.sac',
            f'bp c {float(freqmin):g} {float(freqmax):g} n {int(corners)} p {int(passes)}',
            'w over',
            'q',
            '',
        ])
        try:
            subprocess.run(
                ['sac'],
                input=sac_script,
                text=True,
                cwd=self.wavepath,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or '').strip() or (exc.stdout or '').strip() or str(exc)
            return False, f'SAC BP failed: {detail}'

        self.read_sac(order=order)
        self._restore_marker_state(previous_state)
        self.preview_hidden_wave_names = previous_hidden
        self.preview_selected_wave_names = previous_selected
        self.preview_jump_highlight_wave_name = previous_jump
        self.current_pick_wave_name = previous_wave_name
        self.current_pick_station_name = previous_station_name
        self.ipage = max(0, min(previous_page, self.axpages - 1)) if self.axpages > 0 else 0
        self.refresh_current_page()
        if self.plotfig is not None and plt.fignum_exists(self.plotfig.number):
            self._refresh_preview_figure(self.plotfig, preview_index)
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            self.plot_compare_preview(preview_index)
        return True, f'Applied SAC BP c {float(freqmin):g} {float(freqmax):g} n {int(corners)} p {int(passes)}'

    def restore_event_from_backup(self, backup_event_path, order='gcarc'):
        backup_event_path = os.path.abspath(str(backup_event_path or '').strip())
        if backup_event_path == '':
            return False, 'Backup event path is empty', None
        if not os.path.isdir(backup_event_path):
            return False, f'Backup event path does not exist: {backup_event_path}', None

        source_files = [
            path for path in sorted(glob.glob(join(backup_event_path, '*' + str(self.suffix or ''))))
            if os.path.isfile(path)
        ]
        if not source_files:
            return False, f'No SAC files found in backup event path: {backup_event_path}', None

        current_files = self._wave_files_for_suffix()
        if not current_files:
            return False, 'Current event has no SAC files to replace', None

        marker_state = self._capture_marker_state()
        previous_page = self.ipage
        previous_wave_name = self.current_pick_wave_name
        previous_station_name = self.current_pick_station_name
        previous_hidden = set(self.preview_hidden_wave_names)
        previous_selected = set(self.preview_selected_wave_names)
        previous_jump = self.preview_jump_highlight_wave_name
        preview_index = 0
        if self.plotfig is not None and hasattr(self.plotfig, '_preview_controls'):
            preview_index = getattr(self.plotfig, '_preview_controls', {}).get('preview_index', 0)

        target_files_by_name = {basename(path): path for path in current_files}
        source_files_by_name = {basename(path): path for path in source_files}
        matching_names = sorted(set(target_files_by_name) & set(source_files_by_name))
        match_summary = f'{len(matching_names)}/{len(current_files)}'
        if not matching_names:
            return False, f'No matching SAC filenames between backup event and current event ({match_summary})', match_summary

        marker_keys = [str(idx) for idx in range(10)]
        user_keys = ('user1', 'user2', 'user3', 'user4', 'user5')
        temp_parent = tempfile.mkdtemp(prefix='dephasekit_restore_', dir='/tmp')
        temp_event_dir = os.path.join(temp_parent, os.path.basename(self.wavepath.rstrip(os.sep)))
        os.makedirs(temp_event_dir, exist_ok=True)

        try:
            for wave_name in matching_names:
                source_path = source_files_by_name[wave_name]
                temp_path = os.path.join(temp_event_dir, wave_name)
                shutil.copy2(source_path, temp_path)

                st = obspy.read(temp_path)
                sac = st[0].stats.sac
                for marker_key in marker_keys:
                    sac[f't{marker_key}'] = math.nan
                    sac[f'kt{marker_key}'] = ''
                for user_key in user_keys:
                    sac[user_key] = math.nan

                for marker_key in marker_keys:
                    marker_value = marker_state['markers'].get(marker_key, {}).get(wave_name, math.nan)
                    if not math.isnan(marker_value):
                        sac[f't{marker_key}'] = marker_value
                for user_key in user_keys:
                    user_value = marker_state['user_markers'].get(user_key, {}).get(wave_name, math.nan)
                    if not math.isnan(user_value):
                        sac[user_key] = user_value
                st.write(temp_path, format='SAC')

            for wave_name in matching_names:
                temp_path = os.path.join(temp_event_dir, wave_name)
                target_path = target_files_by_name[wave_name]
                shutil.move(temp_path, target_path)
        except Exception as exc:
            shutil.rmtree(temp_parent, ignore_errors=True)
            return False, f'Backup restore failed: {exc} ({match_summary})', match_summary

        shutil.rmtree(temp_parent, ignore_errors=True)

        self.read_sac(order=order)
        self._restore_marker_state(marker_state)
        self.preview_hidden_wave_names = previous_hidden
        self.preview_selected_wave_names = previous_selected
        self.preview_jump_highlight_wave_name = previous_jump
        self.current_pick_wave_name = previous_wave_name
        self.current_pick_station_name = previous_station_name
        self.ipage = max(0, min(previous_page, self.axpages - 1)) if self.axpages > 0 else 0
        self.refresh_current_page()
        if self.plotfig is not None and plt.fignum_exists(self.plotfig.number):
            self._refresh_preview_figure(self.plotfig, preview_index)
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            self.plot_compare_preview(preview_index)
        return True, f'Restored backup SAC data from {backup_event_path} ({match_summary})', match_summary

    def set_alignment_marker(self, marker, xlim=None):
        self.set_view_settings(marker, xlim=xlim)

    def set_view_settings(self, marker=None, xlim=None, axis_mode=None):
        if marker is not None:
            if not hasattr(self, marker):
                raise ValueError(f'Unsupported alignment marker: {marker}')
            marker_changed = marker != getattr(self, 'tmarker', None)
            self.tmarker = marker
            if marker_changed or not hasattr(self, 'tmarker_t'):
                self.tmarker_t = np.array(getattr(self, marker), copy=True)
        if xlim is not None:
            self.xlim = list(xlim)
        if axis_mode is not None:
            self.axis_mode = axis_mode
        self.ipage = 0
        self.Change_time_window()

    def _window_start_end_absolute(self, wave_index):
        if getattr(self, 'stack_mode', False):
            return self.xlim[0], self.xlim[1]
        alignment_time = self._relative_reference_time_for_wave(wave_index)
        x1 = alignment_time + self.xlim[0]
        x2 = alignment_time + self.xlim[1]
        return x1, x2

    def _axis_window_for_wave(self, wave_index):
        if getattr(self, 'stack_mode', False):
            return self.xlim[0], self.xlim[1]
        if self.axis_mode == 'relative':
            return self.xlim[0], self.xlim[1]
        return self._window_start_end_absolute(wave_index)

    def _relative_reference_time_for_wave(self, wave_index):
        alignment_time = self._alignment_time_for_wave(wave_index)
        if not math.isnan(alignment_time):
            return alignment_time
        return self._first_available_marker_time_for_wave(wave_index)

    def _first_available_marker_time_for_wave(self, wave_index):
        if wave_index < 0:
            return math.nan
        for marker_key in [str(i) for i in range(10)]:
            marker_attr = f't{marker_key}'
            if not hasattr(self, marker_attr):
                continue
            marker_values = getattr(self, marker_attr)
            if wave_index >= len(marker_values):
                continue
            marker_time = marker_values[wave_index]
            if not math.isnan(marker_time):
                return float(marker_time)
        return math.nan

    def _display_x_value(self, absolute_time, wave_index):
        if getattr(self, 'stack_mode', False):
            return absolute_time
        if self.axis_mode == 'relative':
            return absolute_time - self._relative_reference_time_for_wave(wave_index)
        return absolute_time

    def _stack_marker_display_x_value(self, marker_time, wave_index):
        if not getattr(self, 'stack_mode', False):
            return self._display_x_value(marker_time, wave_index)
        marker_time = _safe_float(marker_time)
        if math.isnan(marker_time):
            return marker_time
        wave_name = ''
        if 0 <= wave_index < len(self.ori_sacnames):
            wave_name = self.ori_sacnames[wave_index]
        sidecar_window = self._stack_sidecar_relative_window(wave_name)
        if sidecar_window is not None:
            window_length = float(sidecar_window[1] - sidecar_window[0])
            if 0.0 <= float(marker_time) <= window_length:
                return float(marker_time)
        if 0 <= wave_index < len(getattr(self, 'wave', [])):
            try:
                trace_start, trace_end = self._trace_time_bounds(self.wave[wave_index])
            except Exception:
                trace_start, trace_end = math.nan, math.nan
            if not math.isnan(trace_start) and not math.isnan(trace_end):
                tolerance = max(float(getattr(self, 'dt', 0.02) or 0.02) * 2.0, 1e-3)
                if (trace_start - tolerance) <= marker_time <= (trace_end + tolerance):
                    return marker_time
        if sidecar_window is not None:
            align_marker_key = self._stack_align_marker_key(wave_name)
            align_value = math.nan
            if align_marker_key is not None:
                align_value = _safe_float(self.markers.get(str(align_marker_key), {}).get(wave_name, math.nan))
                if math.isnan(align_value):
                    sidecar_markers = self._stack_sidecar_for_wave(wave_name).get('markers', {}) or {}
                    align_value = _safe_float(sidecar_markers.get(f't{align_marker_key}', math.nan))
            if math.isnan(align_value):
                align_value = marker_time
            if not math.isnan(align_value):
                shifted_time = float(marker_time) - float(align_value) - float(sidecar_window[0])
                if self._is_display_x_in_window(shifted_time, wave_index):
                    return shifted_time
        if self._is_display_x_in_window(marker_time, wave_index):
            return marker_time
        return marker_time

    def _is_display_x_in_window(self, display_x, wave_index, pad=1e-6):
        window_x1, window_x2 = self._axis_window_for_wave(wave_index)
        return (window_x1 - pad) <= display_x <= (window_x2 + pad)

    def _event_x_to_absolute(self, display_x, wave_index):
        if getattr(self, 'stack_mode', False):
            return display_x
        if self.axis_mode == 'relative':
            return display_x + self._relative_reference_time_for_wave(wave_index)
        return display_x

    def _alignment_time_for_wave(self, wave_index):
        if wave_index < 0 or wave_index >= len(self.tmarker_t):
            return math.nan
        return self.tmarker_t[wave_index]

    def _has_valid_alignment_time(self, wave_index):
        alignment_time = self._alignment_time_for_wave(wave_index)
        return not math.isnan(alignment_time)

    def _trace_sample_at_time(self, trace, sample_time):
        if trace is None or math.isnan(sample_time):
            return math.nan
        idx = int((sample_time - trace.stats.sac.b) / trace.stats.delta)
        if idx < 0 or idx >= len(trace.data):
            return math.nan
        return trace.data[idx]

    def _sort(self, order):
        if order == 'baz':
            idx = np.argsort(self.baz)
        elif order == 'gcarc':
            idx = np.argsort(self.gcarc)
        elif order == 'az':
            idx = np.argsort(self.az)
        # elif order == 'date':
        #     idx = pd.to_datetime(self.filenames, format='%Y.%j.%H.%M.%S').argsort()
        else:
            pass

        self.baz = self.baz[idx]
        self.az = self.az[idx]
        self.gcarc = self.gcarc[idx]

        self.tmarker_t = self.tmarker_t[idx]
        self.t0 = self.t0[idx]
        self.t1 = self.t1[idx]
        self.t2 = self.t2[idx]
        self.t3 = self.t3[idx]
        self.t4 = self.t4[idx]
        self.t5 = self.t5[idx]
        self.t6 = self.t6[idx]
        self.t7 = self.t7[idx]
        self.t8 = self.t8[idx]
        self.t9 = self.t9[idx]
        #
        if hasattr(self, 'wave_raw'):
            self.wave_raw = obspy.Stream([self.wave_raw[i] for i in idx])
        self.wave = obspy.Stream([self.wave[i] for i in idx])
        self.filenames = [self.filenames[i] for i in idx]
        self.wavenames = [self.wavenames[i] for i in idx]
        self.ori_sacnames = [self.ori_sacnames[i] for i in idx]

    def plotwave(self):
        # bound = np.zeros(self.time_axis.shape[0])
        axs = [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]  # 将所有的子图放在一个列表中
        self._sync_pick_window_visible_pages()
        for slot_index, ax in enumerate(axs):
            wave_index = self._visible_wave_index_for_page_slot(slot_index)
            if wave_index is not None:
                current_wave_name = self.ori_sacnames[wave_index]
                reference_time = 0.0 if getattr(self, 'stack_mode', False) else self._relative_reference_time_for_wave(wave_index)
                if math.isnan(reference_time):
                    ax.cla()
                    ax.text(
                        0.50, 0.50,
                        f'Missing {self.tmarker} and all fallback markers',
                        color='#b22222', fontsize=11,
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes
                    )
                    ax.set_xlim(self.xlim[0], self.xlim[1])
                    continue
                abs_x1 = reference_time + self.xlim[0]
                abs_x2 = reference_time + self.xlim[1]
                start_index = int((abs_x1 - self.wave[wave_index].stats.sac.b) / self.wave[wave_index].stats.delta)
                end_index = int((abs_x2 - self.wave[wave_index].stats.sac.b) / self.wave[wave_index].stats.delta)
                # print(self.wave)
                # print(start_index, end_index,self.wave[i].stats.delta)
                # print(self.dt)
                polarity_factor = self._wave_polarity_factor(current_wave_name)
                clipped_start_index = max(0, start_index)
                clipped_end_index = min(len(self.wave[wave_index].data), end_index)
                r_amp_axis = (
                    self.wave[wave_index].data[clipped_start_index:clipped_end_index]
                    * self.enf
                    * polarity_factor
                )
                # print(r_amp_axis)
                self.time_axis = (
                    self.wave[wave_index].times()[clipped_start_index:clipped_end_index]
                    + self.wave[wave_index].stats.sac.b
                )
                if self.axis_mode == 'relative' and not getattr(self, 'stack_mode', False):
                    self.time_axis = self.time_axis - self._relative_reference_time_for_wave(wave_index)
                wave_color = "black"
                if current_wave_name in self.preview_selected_wave_names:
                    wave_color = "#ff5fa2"
                elif current_wave_name == self.preview_jump_highlight_wave_name:
                    wave_color = "#ff5fa2"
                elif self._is_user1_wave(current_wave_name):
                    wave_color = self.user1_mark_color
                elif self._is_user5_wave(current_wave_name):
                    wave_color = self.user5_mark_color
                elif self._is_user4_wave(current_wave_name):
                    wave_color = self.user4_mark_color
                elif self._is_preview_purple_wave(current_wave_name):
                    wave_color = self.preview_mark_color
                # page_index = i // 5  # 计算当前是第几页c
                self.A1lines[slot_index] = ax.plot(self.time_axis, r_amp_axis, color=wave_color, linewidth=0.5)
                # axs[ax_index].axvline(x=self.tmarker_t[i], color='#d62728', linewidth=0.5)
                # axs[ax_index].text(x=self.tmarker_t[i], y=-2, s='PKIKP', color='#d62728', fontsize=9)
                # axs[ax_index].axvline(x=self.PKiKP_t[i], color='#17becf', linewidth=0.5)
                # axs[ax_index].text(x=self.PKiKP_t[i], y=-2, s='PKiKP', color='#17becf', fontsize=9)
                station_label = '{}'.format(self.filenames[wave_index])
                if getattr(self, 'stack_mode', False):
                    sidecar = self._stack_sidecar_for_wave(current_wave_name)
                    group_name = str(sidecar.get('group_name') or '').strip()
                    align_marker = str(sidecar.get('align_marker') or '').strip()
                    stack_parts = []
                    if group_name:
                        stack_parts.append(group_name)
                    if align_marker:
                        stack_parts.append(align_marker)
                    used_count = self._stack_meta_int(current_wave_name, 'wave_count_used', default=0)
                    if used_count > 0:
                        stack_parts.append(f'N={used_count}')
                    if stack_parts:
                        station_label = f"{station_label} [{' / '.join(stack_parts)}]"
                if self._is_user4_wave(current_wave_name):
                    station_label = f'{station_label} [flip]'
                if self._is_user5_wave(current_wave_name):
                    station_label = f'{station_label} [user5]'
                ax.text(0.01, 0.95, station_label, color="#F4606C",
                        horizontalalignment='left', verticalalignment='top',
                        transform=ax.transAxes)
                ax.text(
                    0.01, 0.80,
                    'dis:{:.2f} az:{:.2f} baz:{:.2f}'.format(self.gcarc[wave_index], self.az[wave_index], self.baz[wave_index]),
                    color="#81D8CF", horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes)
                ax.text(
                    0.01, 0.66,
                    self.current_wave_theory_delta_text(current_wave_name),
                    color="#5a5a5a", horizontalalignment='left', verticalalignment='top',
                    transform=ax.transAxes, fontsize=8.8
                )
                self._draw_crustal_text_artist(ax, current_wave_name)
                # axs[ax_index].text(0.01, 0.65, 'tplo:{:.2f} tpla:{:.2f}'.format(self.tplo[i], self.tpla[i]),
                #                    color="#81D8CF", horizontalalignment='left', verticalalignment='top',
                #                    transform=axs[ax_index].transAxes)
                for marker_key, marker_map in self.markers.items():
                    click_time = marker_map.get(current_wave_name, math.nan)
                    if not math.isnan(click_time):
                        display_time = self._stack_marker_display_x_value(click_time, wave_index)
                        if not self._is_display_x_in_window(display_time, wave_index):
                            continue
                        self._draw_marker_artists(ax, current_wave_name, marker_key, display_time)
                # ===== 计算并绘制 t6 - t7 与振幅比 =====
                t7 = self.markers['7'].get(current_wave_name, math.nan) if '7' in self.markers else math.nan
                t6 = self.markers['6'].get(current_wave_name, math.nan) if '6' in self.markers else math.nan

                # 如果 t7 和 t6 都存在，则计算差值
                if not math.isnan(t7) and not math.isnan(t6):
                    dt = t6 - t7
                    amp7 = self._trace_sample_at_time(self.wave[wave_index], t7)
                    amp6 = self._trace_sample_at_time(self.wave[wave_index], t6)
                    if not math.isnan(amp7):
                        amp7 *= polarity_factor
                    if not math.isnan(amp6):
                        amp6 *= polarity_factor

                    # 显示在左下角
                    ax.text(
                        0.02, 0.20,
                        "Δt(t6-t7) = ",
                        transform=ax.transAxes
                    )

                    ax.text(
                        0.07, 0.19,
                        f"{dt:.2f}s",
                        color="#CF4B00",
                        fontsize=11,
                        transform=ax.transAxes
                    )
                    

                    ax.text(
                        0.02, 0.05,
                        f"Amp_ratio = ",
                        transform=ax.transAxes
                    )

                    if math.isnan(amp7) or math.isnan(amp6) or np.isclose(amp6, 0.0):
                        amp_ratio_text = "n/a"
                    else:
                        amp_ratio_text = f"{amp7 / amp6:.2f}"

                    ax.text(
                        0.07, 0.04,
                        amp_ratio_text,
                        color="#9CC6DB",
                        fontsize=11,
                        transform=ax.transAxes
                    )
            
            else:
                self.A1lines[slot_index] = []
                ax.axis('off')  # 关闭子图轴

    def onkeypress(self, event):
        # 记录当前按下的数字键
        if hasattr(event, 'key') and event.key:
            self.key = str(event.key).lower()
        elif hasattr(event, 'text') and callable(event.text):
            self.key = event.text().lower()
        self.pick_mode_armed = self.key in self.markers or self.key in ('d', 's')

    def clear_pick_mode(self):
        self.key = None
        self.pick_mode_armed = False

    def _set_current_pick_by_index(self, wave_name_index):
        if wave_name_index is None or wave_name_index < 0 or wave_name_index >= len(self.ori_sacnames):
            return False
        self._remember_pick_wave(wave_name_index)
        return True

    def jump_to_wave_name(self, wave_name, refresh=True):
        if not wave_name:
            return False
        if self._is_preview_hidden_wave(wave_name):
            return False
        try:
            wave_name_index = self.ori_sacnames.index(wave_name)
        except ValueError:
            return False
        visible_indices = self._visible_wave_indices()
        try:
            visible_position = visible_indices.index(wave_name_index)
        except ValueError:
            return False
        self.clear_pick_mode()
        self._set_current_pick_by_index(wave_name_index)
        self.ipage = visible_position // self.maxidx
        if refresh:
            self.refresh_current_page()
            self.fig.canvas.draw_idle()
        return True

    def jump_from_preview_to_wave_name(self, wave_name, refresh=True):
        jumped = self.jump_to_wave_name(wave_name, refresh=False)
        if not jumped:
            return False
        self.preview_jump_highlight_wave_name = wave_name
        if refresh:
            self.refresh_current_page()
            self.fig.canvas.draw_idle()
        return True

    def _switch_stack_preview_wave(self, fig, preview_index, direction):
        stack_names = self._stack_preview_stack_wave_names()
        if not stack_names:
            self._set_preview_search_status(fig, 'No stack waveforms available', color='#8b0000')
            return False
        current_name = getattr(fig, '_stack_preview_wave_name', None) or self._current_stack_preview_wave_name()
        try:
            current_index = stack_names.index(str(current_name))
        except ValueError:
            current_index = 0
        next_index = (current_index + int(direction)) % len(stack_names)
        next_name = stack_names[next_index]
        fig._stack_preview_wave_name = next_name
        fig._preview_forced_selected_wave_names = [next_name]
        self.current_pick_wave_name = next_name
        try:
            self.jump_to_wave_name(next_name, refresh=True)
        except Exception:
            pass
        self._refresh_preview_figure(fig, preview_index)
        self._set_preview_search_status(
            fig,
            f'Stack {next_index + 1}/{len(stack_names)}: {next_name}',
            color='#1f4e79',
        )
        return True

    def set_jump_target_mode(self, mode):
        mode = str(mode or '').lower()
        if mode not in ('user1', 'user2', 'user4', 'user5'):
            return False
        self.jump_target_mode = mode
        return True

    def jump_target_mode_label(self, mode=None):
        mode = str(mode or self.jump_target_mode or 'user2').lower()
        if mode == 'user1':
            return 'U1'
        if mode == 'user5':
            return 'U5'
        if mode == 'user4':
            return 'Flip'
        return 'U2'

    def _jump_target_wave_indices(self, mode=None):
        mode = str(mode or self.jump_target_mode or 'user2').lower()
        marked_indices = []
        for idx, wave_name in enumerate(self.ori_sacnames):
            if self._is_preview_hidden_wave(wave_name):
                continue
            if mode == 'user1':
                if self._is_user1_wave(wave_name):
                    marked_indices.append(idx)
            elif mode == 'user5':
                if self._is_user5_wave(wave_name):
                    marked_indices.append(idx)
            elif mode == 'user4':
                if self._is_user4_wave(wave_name):
                    marked_indices.append(idx)
            else:
                if self._is_preview_purple_wave(wave_name) and not self._is_user1_wave(wave_name):
                    marked_indices.append(idx)
        return marked_indices

    def focus_first_jump_target_wave(self, mode=None, refresh=True):
        marked_indices = self._jump_target_wave_indices(mode=mode)
        if not marked_indices:
            return None
        target_index = marked_indices[0]
        self.jump_to_wave_name(self.ori_sacnames[target_index], refresh=refresh)
        return target_index

    def _user_marker_value(self, wave_name, marker_key):
        return self.user_markers.get(marker_key, {}).get(wave_name, math.nan)

    def _has_user_marker(self, wave_name, marker_key):
        return not math.isnan(self._user_marker_value(wave_name, marker_key))

    def _set_user_marker(self, wave_name, marker_key, enabled, value=1.0):
        if not wave_name or marker_key not in self.user_markers:
            return False
        changed = False
        if enabled:
            changed = self._clear_incompatible_user_markers(wave_name, marker_key)
        current_enabled = self._has_user_marker(wave_name, marker_key)
        if enabled == current_enabled:
            return changed
        self.user_markers[marker_key][wave_name] = value if enabled else math.nan
        self._mark_wave_markers_dirty(wave_name)
        # In stack mode, member user-marker edits must stay visible immediately,
        # but the source SAC write is queued so repeated preview edits do not
        # block the GUI on disk I/O.
        if getattr(self, 'stack_mode', False):
            try:
                is_stack_trace = wave_name in self.ori_sacnames
            except Exception:
                is_stack_trace = False
            if not is_stack_trace:
                self.stack_manual_user_keys.add((str(wave_name), str(marker_key)))
                write_value = value if enabled else math.nan
                try:
                    self._propagate_member_user_marker_to_siblings(wave_name, marker_key, write_value)
                    self._queue_source_marker_write(wave_name, marker_key)
                except Exception as exc:
                    print(f'warn: member user-marker queue failed for {wave_name} {marker_key}: {exc}')
        return True

    def _write_member_user_marker_to_source(self, wave_name, marker_key, value):
        """Write a stack-preview member user-marker edit back to the member's
        source SAC file and propagate to sibling original-event windows."""
        if not getattr(self, 'stack_mode', False):
            return False
        source_path = self._source_wave_path(wave_name)
        if not source_path or not os.path.exists(source_path):
            return False
        try:
            st = obspy.read(source_path)
        except Exception:
            return False
        sac = st[0].stats.sac
        numeric = _safe_float(value)
        write_value = float(numeric) if not math.isnan(numeric) else math.nan
        try:
            setattr(sac, marker_key, write_value)
        except Exception:
            return False
        try:
            st.write(source_path, format='SAC')
        except Exception as exc:
            print(f'warn: failed to write member user-marker to {source_path}: {exc}')
            return False
        self._propagate_member_user_marker_to_siblings(wave_name, marker_key, write_value)
        return True

    def _propagate_member_user_marker_to_siblings(self, wave_name, marker_key, value):
        try:
            from PySide6.QtWidgets import QApplication
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        source_dir = str(getattr(self, 'runtime_event_dir', ''))
        if not source_dir:
            return
        numeric = _safe_float(value)
        v = float(numeric) if not math.isnan(numeric) else math.nan
        for widget in app.topLevelWidgets():
            try:
                other = widget.mpl.wavefig
            except Exception:
                continue
            if other is self:
                continue
            if getattr(other, 'stack_mode', False):
                continue
            if str(getattr(other, 'runtime_event_dir', '')) != source_dir:
                continue
            if marker_key in getattr(other, 'user_markers', {}):
                other.user_markers[marker_key][wave_name] = v
            idx = other._wave_index_by_name(wave_name)
            if idx is not None and 0 <= idx < len(getattr(other, 'wave', [])):
                try:
                    setattr(other.wave[idx].stats.sac, marker_key, v)
                except Exception:
                    pass


    def _clear_incompatible_user_markers(self, wave_name, marker_key):
        changed = False
        if marker_key in ('user1', 'user4'):
            changed = self._set_user_marker(wave_name, 'user2', False) or changed
            changed = self._set_user_marker(wave_name, 'user5', False) or changed
        elif marker_key == 'user2':
            changed = self._set_user_marker(wave_name, 'user5', False) or changed
        elif marker_key == 'user5':
            changed = self._set_user_marker(wave_name, 'user2', False) or changed
        return changed

    def _toggle_user1_marker(self, wave_name):
        if not wave_name:
            return False
        next_enabled = not self._is_user1_wave(wave_name)
        changed = self._set_user_marker(wave_name, 'user1', next_enabled)
        if next_enabled:
            self._set_user_marker(wave_name, 'user2', False)
            self._set_user_marker(wave_name, 'user5', False)
        return changed

    def _toggle_user4_marker(self, wave_name):
        if not wave_name:
            return False
        return self._set_user_marker(wave_name, 'user4', not self._is_user4_wave(wave_name), value=-1.0)

    def _toggle_user5_marker(self, wave_name):
        if not wave_name:
            return False
        next_enabled = not self._is_user5_wave(wave_name)
        changed = self._set_user_marker(wave_name, 'user5', next_enabled)
        if next_enabled:
            self._set_user_marker(wave_name, 'user2', False)
        return changed

    def _is_preview_purple_wave(self, wave_name):
        return self._has_user_marker(wave_name, 'user2')

    def _is_user5_wave(self, wave_name):
        return self._has_user_marker(wave_name, 'user5')

    def _is_user4_wave(self, wave_name):
        return self._has_user_marker(wave_name, 'user4')

    def _wave_polarity_factor(self, wave_name):
        return -1.0 if self._is_user4_wave(wave_name) else 1.0

    def _is_preview_hidden_wave(self, wave_name):
        return wave_name in self.preview_hidden_wave_names

    def _visible_wave_indices(self):
        return [
            idx for idx, wave_name in enumerate(self.ori_sacnames)
            if not self._is_preview_hidden_wave(wave_name)
        ]

    def _visible_wave_count(self):
        return len(self._visible_wave_indices())

    def _visible_wave_index_for_page_slot(self, slot_index):
        if slot_index < 0:
            return None
        visible_indices = self._visible_wave_indices()
        visible_position = self.ipage * self.maxidx + slot_index
        if visible_position < 0 or visible_position >= len(visible_indices):
            return None
        return visible_indices[visible_position]

    def _sync_pick_window_visible_pages(self):
        visible_count = self._visible_wave_count()
        if visible_count <= 0:
            self.axpages = 1
            self.waveidx = []
            self.ipage = 0
            return
        self.axpages, self.waveidx = indexpags(visible_count, self.maxidx)
        self.ipage = min(max(self.ipage, 0), self.axpages - 1)
        if (self.current_pick_wave_name is None
                or self.current_pick_wave_name not in self.ori_sacnames
                or self._is_preview_hidden_wave(self.current_pick_wave_name)):
            first_visible_index = self._visible_wave_index_for_page_slot(0)
            if first_visible_index is not None:
                self._remember_pick_wave(first_visible_index)

    def _is_user1_wave(self, wave_name):
        return self._has_user_marker(wave_name, 'user1')

    def _preview_wave_colors(self, meta, is_selected):
        if meta.get('stack_preview_role') == 'stack':
            return '#c62828', 2.0 if is_selected else 1.35
        if meta.get('is_user1_marked', False):
            return self.user1_mark_color, 1.7 if is_selected else 0.95
        if meta.get('is_user4_marked', False):
            return self.user4_mark_color, 1.7 if is_selected else 0.95
        if meta.get('is_marked_m', False):
            return self.preview_mark_color, 1.55 if is_selected else 0.95
        if meta.get('is_user5_marked', False):
            return self.user5_mark_color, 1.7 if is_selected else 0.95
        if is_selected:
            return '#ff375f', 1.35
        return 'black', 0.2

    def _preview_selection_glow_color(self, meta):
        if meta.get('is_user1_marked', False):
            return '#fff2a8'
        if meta.get('is_user4_marked', False):
            return '#ffe4b5'
        if meta.get('is_marked_m', False):
            return '#f3c8ff'
        if meta.get('is_user5_marked', False):
            return self.user5_selected_color
        return '#ffd4df'

    def _apply_preview_line_style(self, line, meta, is_selected, line_color, line_width):
        line.set_color(line_color)
        line.set_linewidth(line_width)
        if is_selected:
            glow_width = max(line_width + 2.4, line_width * 2.2)
            glow_color = self._preview_selection_glow_color(meta)
            line.set_path_effects([
                path_effects.Stroke(linewidth=glow_width, foreground=glow_color, alpha=0.55),
                path_effects.Normal(),
            ])
        else:
            line.set_path_effects([])

    def _refresh_pick_window_if_available(self, focus_current_wave=False):
        if not hasattr(self, 'fig') or self.fig is None:
            return
        self._sync_pick_window_visible_pages()
        if focus_current_wave and self.current_pick_wave_name:
            self.jump_to_wave_name(self.current_pick_wave_name, refresh=False)
        self.refresh_current_page()
        self.fig.canvas.draw_idle()

    def _sync_pick_highlight_from_preview_selection(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self.preview_selected_wave_names = set()
            return
        metadata = preview_state.get('metadata', [])
        selected_wave_names = {
            metadata[idx].get('wave_name')
            for idx in preview_state.get('selected_indices', set())
            if 0 <= idx < len(metadata) and metadata[idx].get('wave_name')
        }
        self.preview_selected_wave_names = {
            wave_name for wave_name in selected_wave_names
            if not self._is_preview_hidden_wave(wave_name)
        }

    def marked_wave_position(self, wave_name, mode=None):
        if not wave_name:
            return None, 0
        if self._is_preview_hidden_wave(wave_name):
            return None, 0
        marked_indices = self._jump_target_wave_indices(mode=mode)
        if not marked_indices:
            return None, 0
        try:
            target_index = self.ori_sacnames.index(wave_name)
        except ValueError:
            return None, len(marked_indices)
        for position, marked_index in enumerate(marked_indices, start=1):
            if marked_index == target_index:
                return position, len(marked_indices)
        return None, len(marked_indices)

    def jump_to_marked_wave(self, step=1, mode=None):
        marked_indices = self._jump_target_wave_indices(mode=mode)
        if not marked_indices:
            return None
        current_index = None
        if self.current_pick_wave_name in self.ori_sacnames:
            current_index = self.ori_sacnames.index(self.current_pick_wave_name)
        if step >= 0:
            if current_index is None:
                current_index = self.ipage * self.maxidx - 1
            for idx in marked_indices:
                if idx > current_index:
                    target_index = idx
                    break
            else:
                target_index = marked_indices[0]
        else:
            if current_index is None:
                current_index = min(self.sta_num, (self.ipage + 1) * self.maxidx)
            for idx in reversed(marked_indices):
                if idx < current_index:
                    target_index = idx
                    break
            else:
                target_index = marked_indices[-1]
        self.jump_to_wave_name(self.ori_sacnames[target_index], refresh=True)
        return target_index

    def jump_to_missing_alignment_wave(self, step=1, marker=None):
        missing_indices = self.missing_alignment_wave_indices(marker=marker)
        if not missing_indices:
            return None
        current_index = None
        if self.current_pick_wave_name in self.ori_sacnames:
            current_index = self.ori_sacnames.index(self.current_pick_wave_name)
        if step >= 0:
            if current_index is None:
                current_index = self.ipage * self.maxidx - 1
            for idx in missing_indices:
                if idx > current_index:
                    target_index = idx
                    break
            else:
                target_index = missing_indices[0]
        else:
            if current_index is None:
                current_index = min(self.sta_num, (self.ipage + 1) * self.maxidx)
            for idx in reversed(missing_indices):
                if idx < current_index:
                    target_index = idx
                    break
            else:
                target_index = missing_indices[-1]
        self.jump_to_wave_name(self.ori_sacnames[target_index], refresh=True)
        return target_index

    def find_wave_index(self, query):
        query = str(query or '').strip().lower()
        if query == '':
            return None
        for idx, wave_name in enumerate(self.ori_sacnames):
            if self._is_preview_hidden_wave(wave_name):
                continue
            station_name = self.filenames[idx].lower()
            full_wave_name = wave_name.lower()
            if query in station_name or query in full_wave_name:
                return idx
        return None

    def active_preview_wave_name(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return None
        metadata = preview_state.get('metadata', [])
        active_index = preview_state.get('active_index', 0)
        if active_index < 0 or active_index >= len(metadata):
            return None
        return metadata[active_index].get('wave_name')

    def _wave_index_by_name(self, wave_name):
        if not wave_name:
            return None
        try:
            return self.ori_sacnames.index(wave_name)
        except ValueError:
            return None

    def _marker_sample_step_seconds(self, step_samples):
        if self.dt is None:
            return 0.0
        return float(step_samples) * float(self.dt)

    def _trace_time_bounds(self, trace):
        trace_start = float(trace.stats.sac.b)
        trace_end = trace_start + (len(trace.data) - 1) * float(trace.stats.delta)
        return trace_start, trace_end

    def _set_wave_marker_time(self, wave_name, marker_key, marker_time, update_alignment_reference=False):
        # Picking changes marker data; changing the view alignment must stay explicit.
        if marker_key not in self.markers or not wave_name:
            return False
        # The stack trace's align marker is structural (= -x1) and read-only.
        # Silently ignore edits to it so the pick window doesn't pretend to
        # accept a change that can't persist. Re-align by re-stacking instead.
        if self._is_stack_trace_align_marker(wave_name, marker_key):
            try:
                print(f'Stack trace align marker t{marker_key} is read-only (re-stack to re-align)')
            except Exception:
                pass
            return False
        if wave_name not in self.markers[marker_key]:
            # In stack mode the wave_name may be a member waveform that was
            # never seeded into self.markers (e.g. picked before its preview
            # loaded). Create the entry instead of rejecting the pick so member
            # marker edits persist in-memory for the session.
            if not getattr(self, 'stack_mode', False):
                return False
            self.markers[marker_key][wave_name] = math.nan
        if getattr(self, 'stack_mode', False):
            self.stack_manual_marker_keys.add((str(wave_name), str(marker_key)))
        self.markers[marker_key][wave_name] = marker_time
        self._mark_wave_markers_dirty(wave_name)
        wave_index = self._wave_index_by_name(wave_name)
        marker_attr = f't{marker_key}'
        if wave_index is not None and hasattr(self, marker_attr):
            marker_array = getattr(self, marker_attr)
            if 0 <= wave_index < len(marker_array):
                marker_array[wave_index] = marker_time
            if 0 <= wave_index < len(getattr(self, 'wave', [])):
                trace = self.wave[wave_index]
                sac = getattr(trace.stats, 'sac', None)
                if sac is not None:
                    try:
                        setattr(sac, marker_attr, marker_time)
                    except Exception:
                        pass
            if 0 <= wave_index < len(getattr(self, 'wave_raw', [])):
                raw_trace = self.wave_raw[wave_index]
                raw_sac = getattr(raw_trace.stats, 'sac', None)
                if raw_sac is not None:
                    try:
                        setattr(raw_sac, marker_attr, marker_time)
                    except Exception:
                        pass
            if update_alignment_reference and self.tmarker == marker_attr and wave_index < len(self.tmarker_t):
                self.tmarker_t[wave_index] = marker_time
        # Stack-preview member picks are absolute source-SAC times. Keep sibling
        # original-event windows in sync immediately, then queue the slower disk
        # write so key-repeat marker edits remain responsive.
        if getattr(self, 'stack_mode', False):
            try:
                is_stack_trace = wave_name in self.ori_sacnames
            except Exception:
                is_stack_trace = False
            if not is_stack_trace:
                try:
                    numeric = _safe_float(marker_time)
                    write_value = float(numeric) if not math.isnan(numeric) else math.nan
                    self._propagate_member_marker_to_siblings(wave_name, marker_key, write_value)
                    self._queue_source_marker_write(wave_name, f't{marker_key}')
                except Exception as exc:
                    print(f'warn: member marker queue failed for {wave_name} t{marker_key}: {exc}')
        return True

    def _write_member_marker_to_source(self, wave_name, marker_key, marker_time):
        """Write a stack-preview member marker pick back to the member's source
        SAC file (absolute time) and propagate to sibling original-event windows.
        """
        if not getattr(self, 'stack_mode', False):
            return False
        source_path = self._source_wave_path(wave_name)
        if not source_path or not os.path.exists(source_path):
            return False
        try:
            st = obspy.read(source_path)
        except Exception:
            return False
        sac = st[0].stats.sac
        header_key = f't{marker_key}'
        numeric = _safe_float(marker_time)
        value = float(numeric) if not math.isnan(numeric) else math.nan
        try:
            setattr(sac, header_key, value)
        except Exception:
            return False
        try:
            st.write(source_path, format='SAC')
        except Exception as exc:
            print(f'warn: failed to write member marker to {source_path}: {exc}')
            return False
        self._propagate_member_marker_to_siblings(wave_name, marker_key, value)
        return True

    def _propagate_member_marker_to_siblings(self, wave_name, marker_key, absolute_time):
        """Update open original-event windows sharing this source dir so their
        in-memory markers (and their finish() write) reflect the stack pick."""
        try:
            from PySide6.QtWidgets import QApplication
        except Exception:
            return
        app = QApplication.instance()
        if app is None:
            return
        source_dir = str(getattr(self, 'runtime_event_dir', ''))
        if not source_dir:
            return
        key_str = str(marker_key)
        numeric = _safe_float(absolute_time)
        value = float(numeric) if not math.isnan(numeric) else math.nan
        for widget in app.topLevelWidgets():
            try:
                other = widget.mpl.wavefig
            except Exception:
                continue
            if other is self:
                continue
            if getattr(other, 'stack_mode', False):
                continue
            if str(getattr(other, 'runtime_event_dir', '')) != source_dir:
                continue
            if key_str in getattr(other, 'markers', {}):
                other.markers[key_str][wave_name] = value
            idx = other._wave_index_by_name(wave_name)
            if idx is not None and 0 <= idx < len(getattr(other, 'wave', [])):
                try:
                    setattr(other.wave[idx].stats.sac, f't{key_str}', value)
                except Exception:
                    pass

    def _preview_marker_reference_time(self, tmarker, wave_name):
        marker_key = str(tmarker or '')
        if marker_key.startswith('t'):
            marker_key = marker_key[1:]
        return self.markers.get(marker_key, {}).get(wave_name, math.nan)

    def _preview_alignment_reference_time(self, align_marker_key, wave_name, reference_times=None):
        if reference_times is not None and wave_name in reference_times:
            try:
                frozen_time = float(reference_times[wave_name])
            except (TypeError, ValueError):
                frozen_time = math.nan
            if not math.isnan(frozen_time):
                return frozen_time
        return self._preview_marker_reference_time(align_marker_key, wave_name)

    def _preview_relative_phase_time(self, align_marker_key, phase_marker_key, wave_name, reference_times=None, trace=None):
        align_time = self._preview_alignment_reference_time(align_marker_key, wave_name, reference_times=reference_times)
        phase_time = self.markers.get(str(phase_marker_key), {}).get(wave_name, math.nan)
        if math.isnan(phase_time) and trace is not None:
            phase_time = _sac_float(trace, f't{self._normalize_marker_key(phase_marker_key)}', math.nan)
        if math.isnan(align_time) or math.isnan(phase_time):
            return math.nan
        return phase_time - align_time

    def _preview_reference_times_from_figure(self, fig, expected_tmarker=None):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is not None:
            if expected_tmarker is not None and preview_state.get('tmarker') != self._normalize_marker_key(expected_tmarker):
                return None
            return preview_state.get('reference_times')
        if expected_tmarker is not None:
            stored_tmarker = getattr(fig, '_preview_reference_tmarker', None)
            if stored_tmarker is not None and stored_tmarker != self._normalize_marker_key(expected_tmarker):
                return None
        return getattr(fig, '_preview_reference_times', None)

    def _preview_reference_times_from_evtdata(self, evtdata):
        if evtdata is None:
            return {}
        reference_times = {}
        for wave_index, tr in enumerate(evtdata.wave_ori):
            wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
            if not wave_name or wave_index >= len(evtdata.reference_t):
                continue
            try:
                reference_time = float(evtdata.reference_t[wave_index])
            except (TypeError, ValueError):
                continue
            if not math.isnan(reference_time):
                reference_times[wave_name] = reference_time
        return reference_times

    def _preview_reference_mode_label(self, tmarker):
        return 'theory'

    def _adjust_preview_alignment_marker(self, fig, preview_index, sample_step, use_selected_set=False):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return False
        if preview_index >= len(self.preview_modes):
            self._set_preview_search_status(fig, 'Invalid preview index', color='#8b0000')
            return False
        marker_key = self.preview_modes[preview_index][0]
        delta_seconds = self._marker_sample_step_seconds(sample_step)
        if np.isclose(delta_seconds, 0.0):
            self._set_preview_search_status(fig, 'Sampling interval is unavailable', color='#8b0000')
            return False
        metadata = preview_state.get('metadata', [])
        if use_selected_set:
            target_indices = sorted(preview_state.get('selected_indices', []))
        else:
            target_indices = [preview_state.get('active_index', 0)]
        target_indices = [idx for idx in target_indices if 0 <= idx < len(metadata)]
        if not target_indices:
            self._set_preview_search_status(fig, 'No active preview waveform', color='#8b0000')
            return False
        updated_wave_names = []
        edge_wave_names = []
        failed_wave_names = []
        first_target_time = None
        for target_index in target_indices:
            wave_name = metadata[target_index].get('wave_name')
            if not wave_name:
                continue
            current_time = self.markers.get(marker_key, {}).get(wave_name, math.nan)
            if math.isnan(current_time):
                failed_wave_names.append(wave_name)
                continue
            wave_index = self._wave_index_by_name(wave_name)
            if wave_index is None or wave_index >= len(self.wave):
                failed_wave_names.append(wave_name)
                continue
            trace = self.wave[wave_index]
            trace_start, trace_end = self._trace_time_bounds(trace)
            target_time = min(max(current_time + delta_seconds, trace_start), trace_end)
            if np.isclose(target_time, current_time):
                edge_wave_names.append(wave_name)
                continue
            if not self._set_wave_marker_time(
                    wave_name,
                    marker_key,
                    target_time,
                    update_alignment_reference=False):
                failed_wave_names.append(wave_name)
                continue
            if first_target_time is None:
                first_target_time = target_time
            updated_wave_names.append(wave_name)
        if not updated_wave_names:
            if edge_wave_names and not failed_wave_names:
                self._set_preview_search_status(
                    fig,
                    f't{marker_key} already at trace edge for {len(edge_wave_names)} waveform(s)',
                    color='#8b0000'
                )
            else:
                self._set_preview_search_status(fig, f'Failed to update t{marker_key} for selected waveform(s)', color='#8b0000')
            return False
        first_wave_name = updated_wave_names[0]
        first_wave_index = self._wave_index_by_name(first_wave_name)
        base_station_name = self.filenames[first_wave_index] if first_wave_index is not None and first_wave_index < len(self.filenames) else first_wave_name
        station_name = self._wave_display_name(first_wave_name, base_station_name)
        self.current_pick_wave_name = first_wave_name
        self.current_pick_station_name = station_name
        if use_selected_set:
            fig._preview_forced_selected_wave_names = list(updated_wave_names)
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_compare_for_preview_index(preview_index)
        self._refresh_pick_window_if_available(focus_current_wave=use_selected_set)
        direction = '+' if delta_seconds > 0 else ''
        delta_applied = delta_seconds
        if len(updated_wave_names) == 1 and not use_selected_set:
            status_text = f't{marker_key} {direction}{delta_applied:.3f}s -> {first_target_time:.3f}s for {station_name}'
        else:
            status_text = f't{marker_key} {direction}{delta_applied:.3f}s for {len(updated_wave_names)} waveform(s)'
            if edge_wave_names:
                status_text += f'; skipped {len(edge_wave_names)} at edge'
        self._set_preview_search_status(
            fig,
            status_text,
            color='#1f4e79'
        )
        return True

    def _nudge_preview_reference_times(self, fig, preview_index, sample_step, use_selected_set=False):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return False
        if preview_index >= len(self.preview_modes):
            self._set_preview_search_status(fig, 'Invalid preview index', color='#8b0000')
            return False
        delta_seconds = self._marker_sample_step_seconds(sample_step)
        if np.isclose(delta_seconds, 0.0):
            self._set_preview_search_status(fig, 'Sampling interval is unavailable', color='#8b0000')
            return False
        metadata = preview_state.get('metadata', [])
        if use_selected_set:
            target_indices = sorted(preview_state.get('selected_indices', []))
        else:
            target_indices = [preview_state.get('active_index', 0)]
        target_indices = [idx for idx in target_indices if 0 <= idx < len(metadata)]
        if not target_indices:
            self._set_preview_search_status(fig, 'No active preview waveform', color='#8b0000')
            return False

        align_marker_key = self.preview_modes[preview_index][0]
        reference_times = dict(preview_state.get('reference_times', {}) or {})
        updated_wave_names = []
        edge_wave_names = []
        failed_wave_names = []
        first_target_time = None
        for target_index in target_indices:
            wave_name = metadata[target_index].get('wave_name')
            if not wave_name:
                continue
            wave_index = self._wave_index_by_name(wave_name)
            if wave_index is None or wave_index >= len(self.wave):
                failed_wave_names.append(wave_name)
                continue
            current_marker_time = self.markers.get(align_marker_key, {}).get(wave_name, math.nan)
            if math.isnan(current_marker_time):
                failed_wave_names.append(wave_name)
                continue
            trace = self.wave[wave_index]
            trace_start, trace_end = self._trace_time_bounds(trace)
            target_marker_time = min(max(current_marker_time + delta_seconds, trace_start), trace_end)
            if np.isclose(target_marker_time, current_marker_time):
                edge_wave_names.append(wave_name)
                continue
            if not self._set_wave_marker_time(
                    wave_name,
                    align_marker_key,
                    target_marker_time,
                    update_alignment_reference=False):
                failed_wave_names.append(wave_name)
                continue
            reference_times[wave_name] = float(target_marker_time)
            if first_target_time is None:
                first_target_time = target_marker_time
            updated_wave_names.append(wave_name)

        if not updated_wave_names:
            if edge_wave_names:
                self._set_preview_search_status(
                    fig,
                    f'Preview waveform already at trace edge for {len(edge_wave_names)} waveform(s)',
                    color='#8b0000'
                )
            else:
                self._set_preview_search_status(fig, 'Failed to move preview waveform(s)', color='#8b0000')
            return False

        preview_state['reference_times'] = reference_times
        fig._preview_reference_times = reference_times
        if use_selected_set:
            fig._preview_forced_selected_wave_names = list(updated_wave_names)
        first_wave_name = updated_wave_names[0]
        first_wave_index = self._wave_index_by_name(first_wave_name)
        base_station_name = self.filenames[first_wave_index] if first_wave_index is not None and first_wave_index < len(self.filenames) else first_wave_name
        station_name = self._wave_display_name(first_wave_name, base_station_name)
        self.current_pick_wave_name = first_wave_name
        self.current_pick_station_name = station_name
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_compare_for_preview_index(preview_index)
        self._refresh_pick_window_if_available(focus_current_wave=use_selected_set)

        direction = '+' if delta_seconds > 0 else ''
        if len(updated_wave_names) == 1 and not use_selected_set:
            status_text = f't{align_marker_key} {direction}{delta_seconds:.3f}s -> {first_target_time:.3f}s for {station_name}'
        else:
            status_text = f't{align_marker_key} {direction}{delta_seconds:.3f}s for {len(updated_wave_names)} waveform(s)'
        if edge_wave_names:
            status_text += f'; skipped {len(edge_wave_names)} at edge'
        self._set_preview_search_status(fig, status_text, color='#1f4e79')
        return True

    def _preview_curve_pick_state(self, fig):
        state = getattr(fig, '_preview_curve_pick', None)
        if state is None:
            state = {
                'active': False,
                'finished': False,
                'points': [],
                'artist': None,
            }
            fig._preview_curve_pick = state
        return state

    def _clear_preview_curve_pick(self, fig):
        state = self._preview_curve_pick_state(fig)
        artist = state.get('artist')
        if artist is not None:
            try:
                artist.remove()
            except ValueError:
                pass
        state['artist'] = None
        state['points'] = []
        state['active'] = False
        state['finished'] = False
        self._update_preview_mode_button_styles(fig)

    def _refresh_preview_curve_artist(self, fig, axr):
        state = self._preview_curve_pick_state(fig)
        points = state.get('points', [])
        artist = state.get('artist')
        if artist is None:
            artist, = axr.plot(
                [],
                [],
                color=self.preview_curve_pick_color,
                linewidth=1.2,
                marker='o',
                markersize=3.5,
                zorder=7,
            )
            state['artist'] = artist
        if points:
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            artist.set_data(xs, ys)
            artist.set_visible(True)
        else:
            artist.set_data([], [])
            artist.set_visible(False)

    def _start_preview_curve_pick(self, fig, axr):
        state = self._preview_curve_pick_state(fig)
        if state.get('active', False):
            self._clear_preview_curve_pick(fig)
            self._clear_preview_peak_submit_guard(fig)
            fig.canvas.draw_idle()
            self._set_preview_search_status(fig, 'Preview pick cancelled', color='#8b0000')
            return
        self._clear_preview_curve_pick(fig)
        self._clear_preview_peak_submit_guard(fig)
        state['active'] = True
        self._refresh_preview_curve_artist(fig, axr)
        fig.canvas.draw_idle()
        self._update_preview_mode_button_styles(fig)
        self._set_preview_search_status(
            fig,
            'Preview pick: left-click to draw, right-click to finish, p to cancel',
            color='#1f4e79'
        )

    def _finish_preview_curve_pick(self, fig):
        state = self._preview_curve_pick_state(fig)
        point_count = len(state.get('points', []))
        if point_count < 2:
            self._clear_preview_curve_pick(fig)
            fig.canvas.draw_idle()
            self._set_preview_search_status(fig, 'Need at least 2 points for preview pick', color='#8b0000')
            return False
        state['active'] = False
        state['finished'] = True
        fig.canvas.draw_idle()
        self._update_preview_mode_button_styles(fig)
        self._set_preview_search_status(
            fig,
            'Preview pick ready; use A to align or enter pk/peak and press Enter',
            color='#1f4e79'
        )
        return True

    def _is_right_click(self, event):
        button = getattr(event, 'button', None)
        if button == 3:
            return True
        try:
            return getattr(button, 'name', '').lower() == 'right'
        except Exception:
            return False

    def _is_left_click(self, event):
        button = getattr(event, 'button', None)
        if button == 1:
            return True
        try:
            return getattr(button, 'name', '').lower() == 'left'
        except Exception:
            return False

    def _is_curve_finish_click(self, event):
        button = getattr(event, 'button', None)
        if button is None:
            return False
        return not self._is_left_click(event)

    def _segment_intersection_along_first_segment(self, p0, p1, q0, q1, tol=1e-9):
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        q0 = np.asarray(q0, dtype=float)
        q1 = np.asarray(q1, dtype=float)
        r = p1 - p0
        s = q1 - q0
        denom = r[0] * s[1] - r[1] * s[0]
        qp = q0 - p0
        if np.isclose(denom, 0.0, atol=tol):
            return None
        t = (qp[0] * s[1] - qp[1] * s[0]) / denom
        u = (qp[0] * r[1] - qp[1] * r[0]) / denom
        if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
            return max(0.0, min(float(t), 1.0)), p0 + t * r
        return None

    def _segment_intersection_point(self, p0, p1, q0, q1, tol=1e-9):
        intersection = self._segment_intersection_along_first_segment(p0, p1, q0, q1, tol=tol)
        if intersection is None:
            return None
        return intersection[1]

    def _preview_curve_wave_intersection_x(self, line, curve_points, target_x_hint=None):
        if line is None or len(curve_points) < 2:
            return None
        wave_x = np.asarray(line.get_xdata(), dtype=float)
        wave_y = np.asarray(line.get_ydata(), dtype=float)
        if wave_x.size < 2 or wave_y.size < 2:
            return None
        for curve_idx in range(len(curve_points) - 1):
            curve_p0 = curve_points[curve_idx]
            curve_p1 = curve_points[curve_idx + 1]
            first_intersection = None
            for wave_idx in range(wave_x.size - 1):
                wave_p0 = (wave_x[wave_idx], wave_y[wave_idx])
                wave_p1 = (wave_x[wave_idx + 1], wave_y[wave_idx + 1])
                intersection = self._segment_intersection_along_first_segment(
                    curve_p0,
                    curve_p1,
                    wave_p0,
                    wave_p1,
                )
                if intersection is None:
                    continue
                curve_fraction, point = intersection
                if first_intersection is None or curve_fraction < first_intersection[0]:
                    first_intersection = (curve_fraction, point)
            if first_intersection is not None:
                return float(first_intersection[1][0])
        return None

    def _parse_preview_curve_pick_request(self, request_text, default_half_window=0.5):
        text = str(request_text or '').strip().lower()
        if text == '':
            return None, None, 'Enter pk/peak as N or N window, e.g. 7, 7 1, or t7 1'
        tokens = text.split()
        mode = str(getattr(self, 'preview_peak_pick_mode', 'pk') or 'pk')
        if tokens[0] in ('peakm', 'pkm'):
            mode = 'pkm'
            if len(tokens) > 1:
                tokens = tokens[1:]
        elif tokens[0] in ('peak', 'pk') and len(tokens) > 1:
            tokens = tokens[1:]
        marker_token = tokens[0]
        if marker_token.startswith('t'):
            marker_key = marker_token[1:]
        else:
            marker_key = marker_token
        if marker_key not in self.marker_styles:
            return None, None, f'Unsupported pk marker: {tokens[0]}'
        half_window = float(default_half_window)
        if len(tokens) >= 2:
            try:
                half_window = float(tokens[1])
            except ValueError:
                return None, None, 'pk window must be numeric, e.g. 7 1 or t7 1'
        if half_window <= 0:
            return None, None, 'pk window must be > 0'
        return marker_key, half_window, mode, None

    def _preview_pick_request_text(self, fig, request_text):
        if request_text is not None:
            return request_text
        controls = getattr(fig, '_preview_controls', {})
        request_box = controls.get('curve_pick_request')
        if request_box is None:
            return ''
        return request_box.text

    def _normalized_preview_pick_submit_text(self, request_text):
        return ' '.join(str(request_text or '').strip().lower().split())

    def _arm_preview_peak_submit_guard(self, fig, request_text):
        normalized_text = self._normalized_preview_pick_submit_text(request_text)
        if normalized_text:
            fig._preview_ignore_next_peak_submit_text = normalized_text

    def _clear_preview_peak_submit_guard(self, fig):
        if hasattr(fig, '_preview_ignore_next_peak_submit_text'):
            delattr(fig, '_preview_ignore_next_peak_submit_text')

    def _consume_preview_peak_submit_guard(self, fig, request_text):
        normalized_text = self._normalized_preview_pick_submit_text(request_text)
        guarded_text = getattr(fig, '_preview_ignore_next_peak_submit_text', None)
        if guarded_text is None:
            return False
        self._clear_preview_peak_submit_guard(fig)
        if normalized_text == guarded_text:
            return True
        return False

    def _preview_peak_target_indices(self, preview_state):
        if preview_state is None:
            return []
        metadata = preview_state.get('metadata', [])
        selected_indices = sorted(preview_state.get('selected_indices', []))
        if selected_indices:
            return [idx for idx in selected_indices if 0 <= idx < len(metadata)]
        return list(range(len(metadata)))

    def _preview_target_entries(self, preview_state):
        metadata = preview_state.get('metadata', [])
        lines = preview_state.get('lines', [])
        y_values = np.asarray(preview_state.get('y_values', []), dtype=float)
        selected_indices = sorted(preview_state.get('selected_indices', []))
        target_indices = self._preview_peak_target_indices(preview_state)
        entries = []
        for target_index in target_indices:
            if target_index >= len(metadata) or target_index >= len(lines) or target_index >= len(y_values):
                continue
            meta = metadata[target_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            entries.append({
                'index': target_index,
                'meta': meta,
                'wave_name': wave_name,
                'display_name': meta.get('name', wave_name),
                'line': lines[target_index],
                'baseline_y': y_values[target_index],
            })
        return selected_indices, entries

    def _preview_line_peak_time_near(self, line, baseline_y, reference_time, center_x, half_window_seconds):
        if line is None or reference_time is None or not np.isfinite(reference_time):
            return None
        wave_x = np.asarray(line.get_xdata(), dtype=float)
        wave_y = np.asarray(line.get_ydata(), dtype=float)
        if wave_x.size == 0 or wave_y.size == 0 or wave_x.size != wave_y.size:
            return None
        search_start = float(center_x) - float(half_window_seconds)
        search_end = float(center_x) + float(half_window_seconds)
        mask = (wave_x >= search_start) & (wave_x <= search_end)
        if not np.any(mask):
            return None
        relative_x = wave_x[mask]
        relative_amp = wave_y[mask] - float(baseline_y)
        if relative_x.size == 0 or not np.isfinite(relative_amp).any():
            return None
        abs_amp = np.abs(relative_amp)
        peak_offset = int(np.nanargmax(abs_amp))
        peak_x = float(relative_x[peak_offset])
        if 0 < peak_offset < len(relative_x) - 1:
            x_segment = relative_x[peak_offset - 1:peak_offset + 2]
            orientation = 1.0 if relative_amp[peak_offset] >= 0 else -1.0
            y_segment = relative_amp[peak_offset - 1:peak_offset + 2] * orientation
            if np.all(np.isfinite(x_segment)) and np.all(np.isfinite(y_segment)):
                try:
                    a, b, _c = np.polyfit(x_segment, y_segment, 2)
                except np.linalg.LinAlgError:
                    a = 0.0
                    b = 0.0
                if not np.isclose(a, 0.0):
                    vertex_x = -b / (2.0 * a)
                    if float(np.min(x_segment)) <= vertex_x <= float(np.max(x_segment)):
                        peak_x = float(vertex_x)
        return float(reference_time) + peak_x

    def _preview_line_peak_time_near_visual(self, line, baseline_y, reference_time, center_x, half_window_seconds):
        if line is None or reference_time is None or not np.isfinite(reference_time):
            return None
        wave_x = np.asarray(line.get_xdata(), dtype=float)
        wave_y = np.asarray(line.get_ydata(), dtype=float)
        if wave_x.size == 0 or wave_y.size == 0 or wave_x.size != wave_y.size:
            return None
        search_start = float(center_x) - float(half_window_seconds)
        search_end = float(center_x) + float(half_window_seconds)
        mask = (wave_x >= search_start) & (wave_x <= search_end)
        if not np.any(mask):
            return None
        relative_x = wave_x[mask]
        relative_amp = wave_y[mask] - float(baseline_y)
        if relative_x.size == 0 or not np.isfinite(relative_amp).any():
            return None
        abs_amp = np.abs(relative_amp)
        if relative_x.size >= 2 and np.isfinite(center_x):
            center_amp = float(np.interp(float(center_x), relative_x, relative_amp))
        else:
            center_index = int(np.nanargmin(np.abs(relative_x - float(center_x))))
            center_amp = float(relative_amp[center_index])

        def local_extrema_offsets(values, mode):
            offsets = []
            for offset in range(1, len(values) - 1):
                left = values[offset - 1]
                current = values[offset]
                right = values[offset + 1]
                if not (np.isfinite(left) and np.isfinite(current) and np.isfinite(right)):
                    continue
                if mode == 'peak':
                    if current >= left and current >= right:
                        offsets.append(offset)
                else:
                    if current <= left and current <= right:
                        offsets.append(offset)
            return offsets

        peak_offsets = local_extrema_offsets(relative_amp, 'peak')
        trough_offsets = local_extrema_offsets(relative_amp, 'trough')
        peak_offset_set = set(peak_offsets)
        trough_offset_set = set(trough_offsets)
        candidate_offsets = peak_offsets + trough_offsets
        if candidate_offsets:
            peak_offset = min(
                candidate_offsets,
                key=lambda idx: (
                    (float(relative_x[idx]) - float(center_x)) ** 2
                    + (float(relative_amp[idx]) - float(center_amp)) ** 2,
                    abs(float(relative_x[idx]) - float(center_x)),
                    -float(abs_amp[idx]),
                ),
            )
        else:
            peak_offset = int(np.nanargmin(np.abs(relative_x - float(center_x))))
        peak_x = float(relative_x[peak_offset])
        if 0 < peak_offset < len(relative_x) - 1:
            x_segment = relative_x[peak_offset - 1:peak_offset + 2]
            if peak_offset in peak_offset_set:
                orientation = 1.0
            elif peak_offset in trough_offset_set:
                orientation = -1.0
            else:
                orientation = 1.0 if relative_amp[peak_offset] >= center_amp else -1.0
            y_segment = relative_amp[peak_offset - 1:peak_offset + 2] * orientation
            if np.all(np.isfinite(x_segment)) and np.all(np.isfinite(y_segment)):
                try:
                    a, b, _c = np.polyfit(x_segment, y_segment, 2)
                except np.linalg.LinAlgError:
                    a = 0.0
                    b = 0.0
                if not np.isclose(a, 0.0):
                    vertex_x = -b / (2.0 * a)
                    if float(np.min(x_segment)) <= vertex_x <= float(np.max(x_segment)):
                        peak_x = float(vertex_x)
        return float(reference_time) + peak_x

    def _normalize_marker_key(self, marker_key):
        marker_key = str(marker_key or '')
        if marker_key.startswith('t'):
            marker_key = marker_key[1:]
        return marker_key

    def _finish_preview_marker_updates(self, fig, preview_index, updated_wave_names, focus_current_wave=True):
        if not updated_wave_names:
            return
        self.current_pick_wave_name = updated_wave_names[0]
        first_index = self._wave_index_by_name(updated_wave_names[0])
        if first_index is not None and first_index < len(self.filenames):
            self.current_pick_station_name = self._wave_display_name(updated_wave_names[0], self.filenames[first_index])
        self._clear_preview_curve_pick(fig)
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_compare_for_preview_index(preview_index)
        self._refresh_pick_window_if_available(focus_current_wave=focus_current_wave)

    def _curve_marker_time_for_preview_line(self, line, baseline_y, reference_time, curve_points, half_window=None):
        curve_x = self._preview_curve_wave_intersection_x(line, curve_points)
        if curve_x is None:
            return None
        if half_window is None:
            return float(reference_time) + float(curve_x)
        return self._preview_line_peak_time_near(
            line,
            baseline_y,
            reference_time,
            curve_x,
            half_window,
        )

    def _apply_preview_curve_peak_pick(self, fig, preview_index, request_text=None):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return False
        curve_state = self._preview_curve_pick_state(fig)
        points = list(curve_state.get('points', []))
        if not curve_state.get('finished', False) or len(points) < 2:
            self._set_preview_search_status(fig, 'Finish the curve before peak', color='#8b0000')
            return False
        if preview_index >= len(self.preview_modes):
            self._set_preview_search_status(fig, 'Invalid preview index', color='#8b0000')
            return False
        request_text = self._preview_pick_request_text(fig, request_text)
        marker_key, half_window, mode, error_message = self._parse_preview_curve_pick_request(
            request_text,
            default_half_window=self.preview_peak_half_window_default,
        )
        if error_message is not None:
            self._set_preview_search_status(fig, error_message, color='#8b0000')
            return False

        reference_marker_key = self._normalize_marker_key(preview_state.get('tmarker'))
        selected_indices, target_entries = self._preview_target_entries(preview_state)
        updated_wave_names = []
        skipped_wave_names = []

        for entry in target_entries:
            wave_name = entry['wave_name']
            reference_time = self._preview_alignment_reference_time(
                reference_marker_key,
                wave_name,
                reference_times=preview_state.get('reference_times'),
            )
            if math.isnan(reference_time):
                skipped_wave_names.append(entry['display_name'])
                continue
            curve_x = self._preview_curve_wave_intersection_x(entry['line'], points)
            if curve_x is None:
                skipped_wave_names.append(entry['display_name'])
                continue
            if mode == 'pkm':
                peak_time = self._preview_line_peak_time_near_visual(
                    entry['line'],
                    entry['baseline_y'],
                    reference_time,
                    curve_x,
                    half_window,
                )
            else:
                peak_time = self._preview_line_peak_time_near(
                    entry['line'],
                    entry['baseline_y'],
                    reference_time,
                    curve_x,
                    half_window,
                )
            if peak_time is None:
                skipped_wave_names.append(entry['display_name'])
                continue
            if self._set_wave_marker_time(
                    wave_name,
                    marker_key,
                    round(float(peak_time), 3),
                    update_alignment_reference=False):
                updated_wave_names.append(wave_name)
            else:
                skipped_wave_names.append(entry['display_name'])

        if not updated_wave_names:
            skipped_summary = ''
            if skipped_wave_names:
                skipped_summary = f'; skipped {len(skipped_wave_names)}'
            self._set_preview_search_status(fig, f'X applied to 0 waveforms{skipped_summary}', color='#8b0000')
            return False

        self._finish_preview_marker_updates(fig, preview_index, updated_wave_names)

        skipped_summary = ''
        if skipped_wave_names:
            skipped_summary = f'; skipped {len(skipped_wave_names)}'
        scope_label = 'selected' if selected_indices else 'visible'
        self._set_preview_search_status(
            fig,
            f'Picked t{marker_key} via {mode} for {len(updated_wave_names)} {scope_label} waveform(s), window {half_window:g}s{skipped_summary}',
            color='#1f4e79'
        )
        return True

    def _apply_preview_reference_peak_pick(self, fig, preview_index, request_text=None):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return False
        if preview_index >= len(self.preview_modes):
            self._set_preview_search_status(fig, 'Invalid preview index', color='#8b0000')
            return False

        request_text = self._preview_pick_request_text(fig, request_text)
        marker_key, half_window, mode, error_message = self._parse_preview_curve_pick_request(
            request_text,
            default_half_window=self.preview_peak_half_window_default,
        )
        if error_message is not None:
            self._set_preview_search_status(fig, error_message, color='#8b0000')
            return False

        selected_indices, target_entries = self._preview_target_entries(preview_state)
        updated_wave_names = []
        skipped_wave_names = []
        for entry in target_entries:
            wave_name = entry['wave_name']
            reference_time = self._preview_alignment_reference_time(
                preview_state.get('tmarker'),
                wave_name,
                reference_times=preview_state.get('reference_times'),
            )
            if reference_time is None or not np.isfinite(reference_time):
                skipped_wave_names.append(entry['display_name'])
                continue
            if mode == 'pkm':
                peak_time = self._preview_line_peak_time_near_visual(
                    entry['line'],
                    entry['baseline_y'],
                    reference_time,
                    0.0,
                    half_window,
                )
            else:
                peak_time = self._preview_line_peak_time_near(
                    entry['line'],
                    entry['baseline_y'],
                    reference_time,
                    0.0,
                    half_window,
                )
            if peak_time is None:
                skipped_wave_names.append(entry['display_name'])
                continue
            rounded_peak_time = round(float(peak_time), 3)
            if self._set_wave_marker_time(
                    wave_name,
                    marker_key,
                    rounded_peak_time,
                    update_alignment_reference=False):
                updated_wave_names.append(wave_name)
            else:
                skipped_wave_names.append(entry['display_name'])

        if not updated_wave_names:
            skipped_summary = ''
            if skipped_wave_names:
                skipped_summary = f'; skipped {len(skipped_wave_names)}'
            self._set_preview_search_status(fig, f'Peak applied to 0 waveforms{skipped_summary}', color='#8b0000')
            return False

        self._finish_preview_marker_updates(fig, preview_index, updated_wave_names)

        skipped_summary = ''
        if skipped_wave_names:
            skipped_summary = f'; skipped {len(skipped_wave_names)}'
        scope_label = 'selected' if selected_indices else 'visible'
        self._set_preview_search_status(
            fig,
            f'Peak picked t{marker_key} via {mode} for {len(updated_wave_names)} {scope_label} waveform(s), window {half_window:g}s{skipped_summary}',
            color='#1f4e79'
        )
        return True

    def _apply_preview_peak_action(self, fig, preview_index, request_text=None):
        request_text = self._preview_pick_request_text(fig, request_text)
        curve_state = self._preview_curve_pick_state(fig)
        curve_points = list(curve_state.get('points', []))
        if curve_state.get('active', False) or (curve_points and not curve_state.get('finished', False)):
            self._set_preview_search_status(
                fig,
                'Right-click to finish the P curve before pk, or press P to cancel and use direct peak mode',
                color='#8b0000'
            )
            return False
        if curve_state.get('finished', False) and len(curve_points) >= 2:
            applied = self._apply_preview_curve_peak_pick(fig, preview_index, request_text=request_text)
            if applied:
                self._arm_preview_peak_submit_guard(fig, request_text)
            return applied
        if self._consume_preview_peak_submit_guard(fig, request_text):
            return False
        return self._apply_preview_reference_peak_pick(fig, preview_index, request_text=request_text)

    def _apply_preview_curve_alignment(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return False
        curve_state = self._preview_curve_pick_state(fig)
        points = list(curve_state.get('points', []))
        if not curve_state.get('finished', False) or len(points) < 2:
            self._set_preview_search_status(fig, 'Finish the curve before A', color='#8b0000')
            return False
        if preview_index >= len(self.preview_modes):
            self._set_preview_search_status(fig, 'Invalid preview index', color='#8b0000')
            return False

        marker_key = self._normalize_marker_key(preview_state.get('tmarker'))
        selected_indices, target_entries = self._preview_target_entries(preview_state)
        updated_wave_names = []
        skipped_wave_names = []

        for entry in target_entries:
            wave_name = entry['wave_name']
            reference_time = self._preview_alignment_reference_time(
                marker_key,
                wave_name,
                reference_times=preview_state.get('reference_times'),
            )
            if math.isnan(reference_time):
                skipped_wave_names.append(entry['display_name'])
                continue
            target_time = self._curve_marker_time_for_preview_line(
                entry['line'],
                entry['baseline_y'],
                reference_time,
                points,
                half_window=0.5,
            )
            if target_time is None:
                skipped_wave_names.append(entry['display_name'])
                continue
            if self._set_wave_marker_time(
                    wave_name,
                    marker_key,
                    round(float(target_time), 3),
                    update_alignment_reference=False):
                updated_wave_names.append(wave_name)
            else:
                skipped_wave_names.append(entry['display_name'])

        if not updated_wave_names:
            skipped_summary = f'; skipped {len(skipped_wave_names)}' if skipped_wave_names else ''
            self._set_preview_search_status(fig, f'A applied to 0 waveforms{skipped_summary}', color='#8b0000')
            return False

        self._finish_preview_marker_updates(fig, preview_index, updated_wave_names)
        skipped_summary = f'; skipped {len(skipped_wave_names)}' if skipped_wave_names else ''
        scope_label = 'selected' if selected_indices else 'visible'
        self._set_preview_search_status(
            fig,
            f'Aligned t{marker_key} for {len(updated_wave_names)} {scope_label} waveform(s){skipped_summary}',
            color='#1f4e79'
        )
        return True

    def _wave_name_index_from_pick_axes(self, event):
        if event.inaxes not in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            return None
        i_t_verify = event.inaxes.get_position().y0
        if i_t_verify > 0.74:
            i_t = 0
        elif 0.58 < i_t_verify < 0.74:
            i_t = 1
        elif 0.42 < i_t_verify < 0.58:
            i_t = 2
        elif 0.26 < i_t_verify < 0.42:
            i_t = 3
        elif 0.10 < i_t_verify < 0.26:
            i_t = 4
        else:
            return None
        return self._visible_wave_index_for_page_slot(i_t)

    def _remember_pick_wave(self, wave_name_index):
        if wave_name_index is None or wave_name_index >= len(self.wave):
            return
        self.current_pick_wave_name = self.ori_sacnames[wave_name_index]
        tr = self.wave[wave_name_index]
        self.current_pick_station_name = self._wave_display_name(
            self.current_pick_wave_name,
            f"{tr.stats.network}.{tr.stats.station}",
        )

    def _marker_artist_gid(self, wave_name, marker_key):
        return f"dephasekit-marker:{marker_key}:{wave_name}"

    def _crustal_text_artist_gid(self, wave_name):
        return f"dephasekit-crustal-text:{wave_name}"

    def _marker_affects_crustal_text(self, marker_key):
        return str(marker_key) in {'5', '6', '8', '9'}

    def _current_crustal_text(self, wave_name):
        crustal_text = self._stack_crustal_summary_text(wave_name)
        if not crustal_text:
            crustal_text = self._single_trace_crustal_text(wave_name)
        return crustal_text

    def _draw_crustal_text_artist(self, ax, wave_name):
        crustal_text = self._current_crustal_text(wave_name)
        if not crustal_text:
            return None
        text = ax.text(
            0.01, 0.52,
            crustal_text,
            color="#2f5d7a",
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes,
            fontsize=8.8,
        )
        try:
            text.set_gid(self._crustal_text_artist_gid(wave_name))
        except Exception:
            pass
        return text

    def _remove_crustal_text_artist(self, ax, wave_name):
        gid = self._crustal_text_artist_gid(wave_name)
        for text in list(getattr(ax, 'texts', [])):
            if getattr(text, 'get_gid', lambda: None)() == gid:
                try:
                    text.remove()
                except Exception:
                    pass

    def _refresh_crustal_text_artist(self, ax, wave_name):
        if ax is None or not wave_name:
            return None
        self._remove_crustal_text_artist(ax, wave_name)
        text = self._draw_crustal_text_artist(ax, wave_name)
        try:
            self.fig.canvas.draw_idle()
        except Exception:
            pass
        return text

    def _draw_marker_artists(self, ax, wave_name, marker_key, display_time):
        label, color = self.marker_styles[marker_key]
        gid = self._marker_artist_gid(wave_name, marker_key)
        line = ax.axvline(x=display_time, color=color, linewidth=0.5)
        text = ax.text(x=display_time, y=0, s=label, color=color, fontsize=10, clip_on=True)
        try:
            line.set_gid(gid)
            text.set_gid(gid)
        except Exception:
            pass
        return line, text

    def _remove_marker_artists(self, ax, wave_name, marker_key, display_time=math.nan):
        label, color = self.marker_styles.get(marker_key, ('', None))
        gid = self._marker_artist_gid(wave_name, marker_key)

        def same_display_time(value):
            try:
                return not math.isnan(display_time) and abs(float(value) - float(display_time)) <= 1e-6
            except Exception:
                return False

        for line in list(getattr(ax, 'lines', [])):
            remove_line = getattr(line, 'get_gid', lambda: None)() == gid
            if not remove_line and color is not None:
                try:
                    xdata = list(line.get_xdata())
                    remove_line = (
                        len(xdata) == 2
                        and same_display_time(xdata[0])
                        and same_display_time(xdata[1])
                        and line.get_color() == color
                    )
                except Exception:
                    remove_line = False
            if remove_line:
                try:
                    line.remove()
                except Exception:
                    pass

        for text in list(getattr(ax, 'texts', [])):
            remove_text = getattr(text, 'get_gid', lambda: None)() == gid
            if not remove_text and color is not None:
                try:
                    text_x, _text_y = text.get_position()
                    remove_text = (
                        text.get_text() == label
                        and same_display_time(text_x)
                        and text.get_color() == color
                    )
                except Exception:
                    remove_text = False
            if remove_text:
                try:
                    text.remove()
                except Exception:
                    pass

    def onclick(self, event):
        self._last_click_refresh_needed = True
        if event.inaxes not in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]:
            return False
        wave_name_index = self._wave_name_index_from_pick_axes(event)
        previous_wave_name = self.current_pick_wave_name
        self._remember_pick_wave(wave_name_index)
        # click_idx = int(np.round(event.ydata))
        if self.key in self.markers:
            click_time = round(event.xdata, 3)
            if click_time:
                if wave_name_index is None:
                    return False
                wave_name = self.ori_sacnames[wave_name_index]
                if (not self.pick_mode_armed
                        and previous_wave_name is not None
                        and wave_name != previous_wave_name):
                    self.clear_pick_mode()
                    return False
                absolute_click_time = round(self._event_x_to_absolute(click_time, wave_name_index), 3)
                old_marker_time = _safe_float(self.markers.get(self.key, {}).get(wave_name, math.nan))
                old_display_time = math.nan
                if not math.isnan(old_marker_time):
                    old_display_time = self._stack_marker_display_x_value(old_marker_time, wave_name_index)
                updates_alignment_reference = self.tmarker == f't{self.key}'
                if not self._set_wave_marker_time(
                        wave_name,
                        self.key,
                        absolute_click_time,
                        update_alignment_reference=updates_alignment_reference):
                    return False
                if updates_alignment_reference:
                    self.pick_mode_armed = False
                    self._last_click_refresh_needed = True
                    return True
                self._remove_marker_artists(event.inaxes, wave_name, self.key, old_display_time)
                self._draw_marker_artists(event.inaxes, wave_name, self.key, click_time)
                if self._marker_affects_crustal_text(self.key):
                    self._refresh_crustal_text_artist(event.inaxes, wave_name)
                self.pick_mode_armed = False
                self._last_click_refresh_needed = False
                return True

        # 删除部分
        elif self.key == 'd':
            click_time = round(event.xdata, 3)
            if click_time:
                if wave_name_index is None:
                    return False
                wave_name = self.ori_sacnames[wave_name_index]
                if (not self.pick_mode_armed
                        and previous_wave_name is not None
                        and wave_name != previous_wave_name):
                    self.clear_pick_mode()
                    return False
                # print(f"Trying to delete: {wave_name}")


                # 查找所有可能的标记
                all_markers = [(k, v[wave_name]) for k, v in self.markers.items() if
                               wave_name in v and not np.isnan(v[wave_name])]

                if all_markers:
                    closest_marker = min(
                        all_markers,
                        key=lambda x: abs(self._stack_marker_display_x_value(x[1], wave_name_index) - click_time)
                    )
                    closest_key, closest_time = closest_marker
                    # print(f"Closest marker found: {closest_marker}")

                    # 删除键值对
                    if wave_name in self.markers[closest_key]:
                        # del self.markers[closest_key][wave_name]
                        old_display_time = self._stack_marker_display_x_value(closest_time, wave_name_index)
                        updates_alignment_reference = self.tmarker == f't{closest_key}'
                        if not self._set_wave_marker_time(
                                wave_name,
                                closest_key,
                                math.nan,
                                update_alignment_reference=updates_alignment_reference):
                            return False
                        if updates_alignment_reference:
                            self.clear_pick_mode()
                            print(f"Deleted {wave_name} from markers[{closest_key}]")
                            self._last_click_refresh_needed = True
                            return True
                        self._remove_marker_artists(event.inaxes, wave_name, closest_key, old_display_time)
                        if self._marker_affects_crustal_text(closest_key):
                            self._refresh_crustal_text_artist(event.inaxes, wave_name)
                        self.clear_pick_mode()
                        print(f"Deleted {wave_name} from markers[{closest_key}]")
                        self._last_click_refresh_needed = False
                        return True
                    else:
                        print(f"{wave_name} not found in markers[{closest_key}]")
        elif self.key == 's':
            if wave_name_index is None:
                return False
            wave_name = self.ori_sacnames[wave_name_index]
            if (not self.pick_mode_armed
                    and previous_wave_name is not None
                    and wave_name != previous_wave_name):
                self.clear_pick_mode()
                return False
            changed = self._toggle_user1_marker(wave_name)
            self.clear_pick_mode()
            if changed:
                state_label = 'enabled' if self._is_user1_wave(wave_name) else 'cleared'
                print(f"User1 {state_label} for {wave_name}")
            return changed
        else:
            return False
        return False

    def refresh_current_page(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.plotwave()
        self.set_page()
        self.set_figure()

    def butprevious(self):
        self.clear_pick_mode()
        self.ipage -= 1
        if self.ipage < 0:
            self.ipage = 0
            return
        # self.set_ylabels()
        self.set_figure()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.plotwave()
        self.set_page()
        self.set_figure()

    def butnext(self):
        self.clear_pick_mode()
        self.ipage += 1
        if self.ipage >= self.axpages:
            self.ipage = self.axpages - 1
            return
        # self.set_ylabels()
        self.set_figure()
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.plotwave()
        self.set_page()
        self.set_figure()

    def plot_preview(self, preview_index=0):
        # WSLg/XWayland: 不要开 ion，否则 figure 一创建就异步弹空白窗，
        # 后续 _draw_preview_content / _maximize_preview_window 又反复
        # hide/resize/show，每次几何变化都合成一帧 → 闪多次黑屏。
        # 改为 ioff，窗口直到几何与内容都就绪后由 _maximize_preview_window 统一 show。
        plt.ioff()
        plt.rcParams['toolbar'] = 'None'
        if preview_index >= len(self.preview_modes):
            return False
        tmarker, x1, x2 = self.preview_modes[preview_index]
        waves, t_lst, reference_times = self._collect_preview_display_stream(tmarker)
        if len(waves) == 0:
            self._emit_compare_status(
                f'No preview waveforms available for t{tmarker}',
                timeout_ms=5000,
            )
            return False

        if self.preview_view_mode == 'tall':
            self.plotfig, axr, axb, axp = init_tall_preview_figure()
        else:
            self.plotfig, axr, axb, axp = init_tall_preview_figure()
        self.plotfig._preview_reference_times = reference_times
        self.plotfig._preview_reference_tmarker = self._normalize_marker_key(tmarker)
        if getattr(self, 'stack_mode', False):
            self.plotfig._stack_preview_wave_name = getattr(self, 'stack_preview_active_wave_name', None)
            window = self._apply_stack_preview_window(
                preview_index,
                self.plotfig._stack_preview_wave_name,
            )
            if window is not None:
                x1, x2 = window
        self._draw_preview_content(
            self.plotfig,
            axr,
            axb,
            axp,
            tmarker,
            x1,
            x2,
            waves=waves,
            t_lst=t_lst,
            reference_times=reference_times,
        )
        self._attach_azimuth_selectors(self.plotfig)
        self._attach_pierce_selectors(self.plotfig)
        self._attach_preview_qt_controls(self.plotfig, preview_index)
        self._maximize_preview_window(self.plotfig)
        self._sync_preview_to_current_pick(self.plotfig)
        # 几何与内容已在 _maximize_preview_window 内统一 show+draw，
        # 这里不再 draw_idle，避免异步重绘再次触发黑屏闪烁。
        return True

    def _collect_preview_stream(self, tmarker, reference_times=None):
        waves = obspy.Stream()
        t_lst = []
        active_reference_times = {}
        for wave_name in self.markers[tmarker].keys():
            click_time = self._preview_alignment_reference_time(
                tmarker,
                wave_name,
                reference_times=reference_times,
            )
            if math.isnan(click_time):
                continue
            if self._is_preview_hidden_wave(wave_name):
                continue
            tr = self._filtered_trace_for_preview(wave_name)
            waves += tr
            t_lst.append(click_time)
            active_reference_times[wave_name] = float(click_time)
        return waves, np.array(t_lst), active_reference_times

    def _draw_preview_content(self, fig, axr, axb, axp, tmarker, x1, x2, waves=None, t_lst=None, reference_times=None):
        if waves is None or t_lst is None:
            waves, t_lst, reference_times = self._collect_preview_display_stream(tmarker, fig=fig)
        else:
            reference_times = {
                getattr(tr.stats, 'dephasekit_wave_name', ''): float(reference_time)
                for tr, reference_time in zip(waves, t_lst)
                if getattr(tr.stats, 'dephasekit_wave_name', '')
            }
        fig._preview_reference_times = reference_times
        fig._preview_reference_tmarker = self._normalize_marker_key(tmarker)
        if len(waves) == 0:
            self.preview_selected_wave_names = set()
            if hasattr(fig, '_preview_state') and fig._preview_state is not None:
                old_info_text = fig._preview_state.get('info_text')
                if old_info_text is not None:
                    try:
                        old_info_text.remove()
                    except ValueError:
                        pass
                old_control_status_text = fig._preview_state.get('control_status_text')
                if old_control_status_text is not None:
                    try:
                        old_control_status_text.remove()
                    except ValueError:
                        pass
            axr.cla()
            axb.cla()
            axp.cla()
            fig.suptitle(f'No marked waveforms for t{tmarker}', fontsize=14)
            fig._preview_state = None
            fig.canvas.draw_idle()
            return

        previous_index = 0
        previous_selected_wave_names = []
        previous_active_wave_name = None
        previous_anchor_wave_name = None
        if hasattr(fig, '_preview_state') and fig._preview_state is not None:
            previous_index = fig._preview_state.get('active_index', 0)
            previous_selected_wave_names = [
                fig._preview_state['metadata'][i]['wave_name']
                for i in sorted(fig._preview_state.get('selected_indices', {0}))
                if i < len(fig._preview_state['metadata'])
            ]
            active_index = fig._preview_state.get('active_index', 0)
            anchor_index = fig._preview_state.get('anchor_index', active_index)
            if active_index < len(fig._preview_state['metadata']):
                previous_active_wave_name = fig._preview_state['metadata'][active_index]['wave_name']
            if anchor_index < len(fig._preview_state['metadata']):
                previous_anchor_wave_name = fig._preview_state['metadata'][anchor_index]['wave_name']
            old_info_text = fig._preview_state.get('info_text')
            if old_info_text is not None:
                try:
                    old_info_text.remove()
                except ValueError:
                    pass
            old_control_status_text = fig._preview_state.get('control_status_text')
            if old_control_status_text is not None:
                try:
                    old_control_status_text.remove()
                except ValueError:
                    pass
        forced_selected_wave_names = getattr(fig, '_preview_forced_selected_wave_names', None)
        if forced_selected_wave_names is not None:
            previous_selected_wave_names = list(forced_selected_wave_names)
            if forced_selected_wave_names:
                previous_active_wave_name = forced_selected_wave_names[0]
                previous_anchor_wave_name = forced_selected_wave_names[0]
            fig._preview_forced_selected_wave_names = None

        axr.cla()
        axb.cla()
        axp.cla()
        axr.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
        axb.grid(color='gray', linestyle='--', linewidth=0.4, axis='x')
        axp.grid(color='gray', linestyle='--', linewidth=0.3, axis='both')
        evtdata = EvtData(waves, t_lst, x1=x1, x2=x2, dt=self.dt)
        defer_azimuth_panel = self._preview_should_defer_side_panels(evtdata.sta_num)
        y_values, y_ticks, y_ticklabels, ylabel = self._preview_y_axis_config(evtdata, order='gcarc')
        stack_overlay_mode = getattr(self, 'stack_mode', False) and self._stack_preview_display_mode() == 'overlay'
        azimuth_y_values = np.asarray(evtdata.gcarc, dtype=float)
        if stack_overlay_mode:
            y_values = np.zeros(evtdata.sta_num, dtype=float)
            y_ticks = np.asarray([-1.0, 0.0, 1.0], dtype=float)
            y_ticklabels = ['-1', '0', '1']
            ylabel = 'Normalized amplitude'
        if defer_azimuth_panel:
            lines = plot_waves_only(axr, evtdata, enf=self.preview_amplitude_scale, y_values=y_values)
            scatter = None
        else:
            azimuth_mask = None
            if getattr(self, 'stack_mode', False):
                azimuth_mask = _stack_member_visible_mask(evtdata)
            lines, scatter = plot_waves_with_masked_azimuth(
                axr,
                axb,
                evtdata,
                enf=self.preview_amplitude_scale,
                y_values=y_values,
                azimuth_mask=azimuth_mask,
                azimuth_y_values=azimuth_y_values,
            )
        tick_interval, tick_mode = self._current_preview_tick_interval(x1, x2)
        if defer_azimuth_panel:
            set_wave_axis_only(
                axr,
                evtdata,
                tmarker,
                y_values=y_values,
                y_ticks=y_ticks,
                y_ticklabels=y_ticklabels,
                ylabel=ylabel,
            )
            deferred_message = (
                f'Deferred in large-event mode\n'
                f'Visible waveforms: {evtdata.sta_num}\n'
                f'Shown again when visible waveforms <= {self.preview_deferred_panel_threshold}'
            )
            self._draw_preview_deferred_panel(
                axb,
                'Azimuth',
                deferred_message,
                xlabel=r'Azimuth ($^\circ$)',
            )
            fig = axr.figure
            fig.suptitle(
                "{}:{}\n Latitude: {:.2f}\N{DEGREE SIGN}, Longitude: {:.2f}\N{DEGREE SIGN}, Depth:{:.1f} km".format(
                    _event_title_prefix(getattr(evtdata, 'is_stack_mode', False)),
                    evtdata.evtname, evtdata.evla, evtdata.evlo, evtdata.evdp),
                fontsize=16)
        elif stack_overlay_mode:
            set_stack_overlay_fig(
                axr,
                axb,
                evtdata,
                tmarker,
                amplitude_scale=self.preview_amplitude_scale,
                interval_x_override=tick_interval,
                azimuth_y_values=azimuth_y_values,
            )
        else:
            set_fig(
                axr, axb, evtdata, tmarker,
                interval_x_override=tick_interval,
                y_values=y_values,
                y_ticks=y_ticks,
                y_ticklabels=y_ticklabels,
                ylabel=ylabel,
            )
        phase_keys, _error_message = self._parse_standard_phase_tokens(self.standard_export_phase_tokens)
        phase_keys = self._phase_keys_with_alignment(phase_keys, tmarker)
        if stack_overlay_mode:
            preview_phase_count = self._draw_preview_phase_annotations(
                axr,
                evtdata,
                y_values,
                tmarker,
                [self._normalize_marker_key(tmarker)],
                reference_times=reference_times,
            )
        else:
            preview_phase_count = self._draw_preview_phase_annotations(
                axr,
                evtdata,
                y_values,
                tmarker,
                phase_keys,
                reference_times=reference_times,
            )
        selected_marker = None
        if not defer_azimuth_panel:
            selected_marker, = axb.plot([], [], 'o', color='red', markersize=6, zorder=5)
        overlay_positions = self._preview_overlay_positions()
        info_text = fig.text(
            overlay_positions['info'][0], overlay_positions['info'][1], '',
            va='center', ha='center',
            fontsize=overlay_positions['info_size'], color='red'
        )
        control_status_text = fig.text(
            overlay_positions['status'][0], overlay_positions['status'][1], '',
            va='center', ha='center',
            fontsize=overlay_positions['status_size'], color='#555555'
        )
        metadata = []
        for tr in evtdata.wave_ori:
            wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
            metadata.append({
                'name': f"{tr.stats.network}.{tr.stats.station}",
                'gcarc': _sac_float(tr, 'gcarc', 0.0),
                'az': _sac_float(tr, 'az', 0.0),
                'wave_name': wave_name,
                'stack_preview_role': getattr(tr.stats, 'dephasekit_stack_preview_role', ''),
                'stack_preview_wave_name': getattr(tr.stats, 'dephasekit_stack_wave_name', ''),
                'stack_summary': self._stack_wave_summary(wave_name),
                'is_marked_m': self._is_preview_purple_wave(wave_name),
                'is_user1_marked': self._is_user1_wave(wave_name),
                'is_user5_marked': self._is_user5_wave(wave_name),
                'is_user4_marked': self._is_user4_wave(wave_name),
            })
        pierce_state = None
        pending_pierce_render = None
        if getattr(self, 'stack_mode', False):
            active_stack_wave_name = getattr(fig, '_stack_preview_wave_name', None)
            pierce_records = self._stack_preview_pierce_points(
                active_stack_wave_name=active_stack_wave_name,
            )
        else:
            self._maybe_generate_current_preview_pierce_cache(metadata)
            pierce_records = self._preview_pierce_points({'metadata': metadata}, selected_only=False)
        if self._preview_should_async_pierce_panel(len(pierce_records)):
            self._draw_preview_deferred_panel(
                axp,
                f'{self.preview_pierce_phase} {self.preview_pierce_model}',
                f'Loading {len(pierce_records)} pierce points...',
                xlabel='Lon',
                ylabel='Lat',
            )
            axp.scatter(
                [float(evtdata.evlo)],
                [float(evtdata.evla)],
                marker='*',
                s=120,
                c='red',
                edgecolors='black',
                linewidths=0.5,
                zorder=5,
            )
            pierce_state = {
                'axes': axp,
                'base_scatter': None,
                'highlight_scatter': None,
                'records': [],
                'bounds': tuple(axp.get_xlim()) + tuple(axp.get_ylim()),
            }
            pending_pierce_render = {
                'token': object(),
                'axes': axp,
                'evtdata': evtdata,
                'records': list(pierce_records),
                'metadata': list(metadata),
            }
        else:
            pierce_state = self._draw_preview_pierce_panel(
                axp,
                evtdata,
                pierce_records,
            )
            pierce_status_message = self._preview_pierce_status_message(metadata, pierce_records)
            if pierce_status_message:
                self._draw_preview_deferred_panel(
                    axp,
                    f'{self.preview_pierce_phase} {self.preview_pierce_model}',
                    pierce_status_message,
                    xlabel='Lon',
                    ylabel='Lat',
                )
                axp.scatter(
                    [float(evtdata.evlo)],
                    [float(evtdata.evla)],
                    marker='*',
                    s=120,
                    c='red',
                    edgecolors='black',
                    linewidths=0.5,
                    zorder=5,
                )
                pierce_state = {
                    'axes': axp,
                    'base_scatter': None,
                    'highlight_scatter': None,
                    'records': [],
                    'bounds': tuple(axp.get_xlim()) + tuple(axp.get_ylim()),
                }
        fig._preview_state = {
            'tmarker': tmarker,
            'evtdata': evtdata,
            'lines': lines,
            'scatter': scatter,
            'selected_marker': selected_marker,
            'pierce_state': pierce_state,
            'pending_pierce_render': pending_pierce_render,
            'defer_side_panels': defer_azimuth_panel,
            'info_text': info_text,
            'control_status_text': control_status_text,
            'metadata': metadata,
            'selected_indices': {0},
            'active_index': 0,
            'anchor_index': 0,
            'amplitude_scale': self.preview_amplitude_scale,
            'phase_keys': phase_keys,
            'preview_phase_count': preview_phase_count,
            'tick_interval': tick_interval,
            'tick_mode': tick_mode,
            'window_width': x2 - x1,
            'y_values': y_values,
            'azimuth_y_values': azimuth_y_values,
            'reference_times': reference_times,
        }
        active_stack_wave_name = getattr(fig, '_stack_preview_wave_name', None)
        if getattr(self, 'stack_mode', False) and active_stack_wave_name:
            for idx, meta in enumerate(metadata):
                if meta.get('wave_name') == active_stack_wave_name:
                    fig._preview_state['selected_indices'] = {idx}
                    fig._preview_state['active_index'] = idx
                    fig._preview_state['anchor_index'] = idx
                    break
        if previous_selected_wave_names:
            restored_indices = {
                idx for idx, meta in enumerate(metadata)
                if meta['wave_name'] in previous_selected_wave_names
            }
            if restored_indices:
                fig._preview_state['selected_indices'] = restored_indices
        if previous_active_wave_name is not None:
            for idx, meta in enumerate(metadata):
                if meta['wave_name'] == previous_active_wave_name:
                    fig._preview_state['active_index'] = idx
                    break
        else:
            fig._preview_state['active_index'] = min(previous_index, evtdata.sta_num - 1)
        if previous_anchor_wave_name is not None:
            for idx, meta in enumerate(metadata):
                if meta['wave_name'] == previous_anchor_wave_name:
                    fig._preview_state['anchor_index'] = idx
                    break
        else:
            fig._preview_state['anchor_index'] = fig._preview_state['active_index']
        self._apply_preview_selection(fig)
        curve_state = self._preview_curve_pick_state(fig)
        if curve_state.get('points'):
            self._refresh_preview_curve_artist(fig, axr)
        if defer_azimuth_panel:
            self._set_preview_search_status(
                fig,
                f'Large-event mode: azimuth deferred until visible waveforms <= {self.preview_deferred_panel_threshold}',
                color='#8a5a00'
            )
        if pending_pierce_render is not None:
            self._set_preview_search_status(
                fig,
                f'Pierce panel loading in background: {len(pending_pierce_render.get("records", []))} points',
                color='#1f4e79'
            )
            self._schedule_pending_preview_pierce_render(fig)
        fig.canvas.draw_idle()

    def _close_preview_control_dock(self):
        dock = getattr(self, 'preview_control_dock', None)
        if dock is None:
            return
        try:
            dock.close()
        except Exception:
            pass
        self.preview_control_dock = None

    def _attach_preview_qt_controls(self, fig, preview_index):
        manager = getattr(fig.canvas, 'manager', None)
        window = getattr(manager, 'window', None)
        fig._preview_controls = {}
        if window is None:
            class _SimpleTextAdapter:
                def __init__(self, value=''):
                    self.text = str(value)

                def set_val(self, value):
                    self.text = str(value)

            class _SimpleComboAdapter:
                def __init__(self):
                    self.items = []
                    self.current_index = 0

                def blockSignals(self, _blocked):
                    return False

                def clear(self):
                    self.items = []
                    self.current_index = 0

                def addItem(self, text, data=None):
                    self.items.append((str(text), data))

                def setCurrentIndex(self, index):
                    self.current_index = max(0, min(int(index), max(len(self.items) - 1, 0)))

                def findText(self, text):
                    target = str(text)
                    for index, (item_text, _item_data) in enumerate(self.items):
                        if item_text == target:
                            return index
                    return -1

                def currentText(self):
                    if not self.items:
                        return ''
                    return self.items[self.current_index][0]

                def currentData(self):
                    if not self.items:
                        return None
                    return self.items[self.current_index][1]

            fig._preview_controls['amplitude'] = _SimpleTextAdapter(f'{self.preview_amplitude_scale:g}')
            fig._preview_controls['amplitude_preset_widget'] = _SimpleComboAdapter()
            fig._preview_controls['peak_half_window'] = _SimpleTextAdapter(f'{self.preview_peak_half_window_default:g}')
            self._refresh_preview_amplitude_preset_combo(fig, select_value=self.preview_amplitude_scale)
            return
        self._close_preview_control_dock()
        dock = QDockWidget('Preview Controls', window)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setTitleBarWidget(QWidget(dock))

        container = QWidget(dock)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(8, 2, 8, 2)
        outer.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(8)
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row3 = QHBoxLayout()
        row3.setSpacing(8)

        controls = {}

        def add_line_edit(row_layout, label_text, initial_text, width=68):
            group = QWidget()
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            group_layout.addWidget(QLabel(label_text))
            widget = QLineEdit()
            widget.setText(str(initial_text))
            widget.setFixedWidth(width)
            widget.setFixedHeight(24)
            group_layout.addWidget(widget)
            row_layout.addWidget(group)
            # After submitting a value with Enter, return keyboard focus to the
            # canvas so subsequent pick shortcuts (digits / p / d) aren't typed
            # into this edit and the user can keep picking without re-clicking.
            def _return_focus_to_canvas(*_args, _fig=fig, _w=widget):
                try:
                    _w.clearFocus()
                    _fig.canvas.setFocus()
                except Exception:
                    pass
            widget.returnPressed.connect(_return_focus_to_canvas)
            return widget

        def add_button(row_layout, label_text, handler, width=None):
            button = QPushButton(label_text)
            if width is not None:
                button.setFixedWidth(width)
            button.setFixedHeight(24)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setAutoDefault(False)
            button.setDefault(False)
            button.clicked.connect(handler)
            row_layout.addWidget(button)
            return button

        def add_combo_box(row_layout, label_text, options, current_text, width=84):
            group = QWidget()
            group_layout = QHBoxLayout(group)
            group_layout.setContentsMargins(0, 0, 0, 0)
            group_layout.setSpacing(4)
            group_layout.addWidget(QLabel(label_text))
            widget = QComboBox()
            widget.addItems(list(options))
            widget.setCurrentText(str(current_text))
            widget.setFixedWidth(width)
            widget.setFixedHeight(24)
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            group_layout.addWidget(widget)
            row_layout.addWidget(group)
            return widget

        tmarker, x1, x2 = self.preview_modes[preview_index]
        controls['x1_widget'] = add_line_edit(row1, 'x1', x1, width=70)
        controls['x2_widget'] = add_line_edit(row1, 'x2', x2, width=70)
        dx_initial = 'auto' if self.preview_x_tick_interval_override is None else f"{self.preview_x_tick_interval_override:g}"
        controls['dx_widget'] = add_line_edit(row1, 'dx', dx_initial, width=70)
        controls['search_widget'] = add_line_edit(row1, 'Station', '', width=180)
        controls['std_phases_widget'] = add_line_edit(row1, 'Ph', self.standard_export_phase_tokens, width=120)
        controls['pierce_phase_widget'] = add_combo_box(
            row1, 'Pierce', ['pP', 'sP'], self.preview_pierce_phase, width=76
        )
        controls['pierce_model_widget'] = add_combo_box(
            row1, 'Model', ['prem', 'iasp91'], self.preview_pierce_model, width=84
        )

        def apply_preview_window_qt():
            try:
                new_x1 = float(controls['x1_widget'].text().strip())
                new_x2 = float(controls['x2_widget'].text().strip())
            except ValueError:
                self._set_preview_search_status(fig, 'x1/x2 must be numeric', color='#8b0000')
                return
            if new_x2 <= new_x1:
                self._set_preview_search_status(fig, 'x2 must be greater than x1', color='#8b0000')
                return
            dx_text = controls['dx_widget'].text().strip().lower()
            if dx_text in ('', 'auto'):
                self.preview_x_tick_interval_override = None
            else:
                try:
                    new_dx = float(dx_text)
                except ValueError:
                    self._set_preview_search_status(fig, 'dx must be numeric or auto', color='#8b0000')
                    return
                if new_dx <= 0:
                    self._set_preview_search_status(fig, 'dx must be > 0', color='#8b0000')
                    return
                self.preview_x_tick_interval_override = new_dx
            canonical_tokens, error_message = self.set_standard_phase_tokens(
                controls['std_phases_widget'].text(),
                preview_index=preview_index,
                refresh=False,
                sync_controls=False,
            )
            if error_message is not None:
                self._set_preview_search_status(fig, error_message, color='#8b0000')
                return
            self.preview_modes[preview_index][1] = new_x1
            self.preview_modes[preview_index][2] = new_x2
            self._refresh_preview_figure(fig, preview_index)
            self._refresh_compare_for_preview_index(preview_index)
            applied_dx, applied_mode = self._current_preview_tick_interval(new_x1, new_x2)
            self._set_preview_search_status(
                fig,
                f"Applied window; dx={applied_dx:g} ({applied_mode}); phases={canonical_tokens or 'none'}",
                color='#1f4e79'
            )

        def select_preview_purple_waveforms_qt():
            selected_count = self._select_preview_purple_waveforms(fig)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No visible purple waveforms', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Selected {selected_count} purple waveform(s)', color='#1f4e79')

        def select_preview_user1_waveforms_qt():
            selected_count = self._select_preview_user1_waveforms(fig)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No visible user1 waveforms', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Selected {selected_count} user1 waveform(s)', color='#1f4e79')

        def select_preview_user5_waveforms_qt():
            selected_count = self._select_preview_user5_waveforms(fig)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No visible user5 waveforms', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Selected {selected_count} user5 waveform(s)', color='#1f4e79')

        def apply_selected_preview_user1_qt():
            selected_count = self._apply_selected_preview_user1(fig)
            self._set_preview_search_status(fig, f'U1+ applied to {selected_count} waveform(s)', color='#1f4e79')

        def clear_selected_preview_user1_qt():
            selected_count = self._clear_selected_preview_user1(fig)
            self._set_preview_search_status(fig, f'U1- cleared for {selected_count} waveform(s)', color='#1f4e79')

        def apply_selected_preview_user5_qt():
            selected_count = self._apply_selected_preview_user5(fig)
            self._set_preview_search_status(fig, f'U5+ applied to {selected_count} waveform(s)', color='#1f4e79')

        def clear_selected_preview_user5_qt():
            selected_count = self._clear_selected_preview_user5(fig)
            self._set_preview_search_status(fig, f'U5- cleared for {selected_count} waveform(s)', color='#1f4e79')

        def select_preview_user4_waveforms_qt():
            selected_count = self._select_preview_user4_waveforms(fig)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No visible flipped waveforms', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Selected {selected_count} flipped waveform(s)', color='#1f4e79')

        def select_preview_group_qt():
            parent_window = getattr(getattr(fig.canvas, 'manager', None), 'window', None)
            dialog = QDialog(parent_window)
            dialog.setWindowTitle('Select Preview Group')
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(10, 10, 10, 10)
            layout.setSpacing(8)
            option_map = [
                ('Purple', 'purple'),
                ('User1', 'user1'),
                ('User4 / Flip', 'user4'),
                ('User5', 'user5'),
                ('t0', 't0'),
                ('t1', 't1'),
                ('t2', 't2'),
                ('t3', 't3'),
                ('t5', 't5'),
                ('t6', 't6'),
                ('t7', 't7'),
                ('t8', 't8'),
                ('t9', 't9'),
            ]
            checkboxes = []
            for label, value in option_map:
                checkbox = QCheckBox(label, dialog)
                checkbox.setProperty('group_value', value)
                layout.addWidget(checkbox)
                checkboxes.append(checkbox)

            button_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
                parent=dialog,
            )
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            layout.addWidget(button_box)

            if dialog.exec() != int(QDialog.DialogCode.Accepted):
                return
            selected_values = [
                checkbox.property('group_value')
                for checkbox in checkboxes
                if checkbox.isChecked()
            ]
            if not selected_values:
                self._set_preview_search_status(fig, 'Choose at least one group', color='#8b0000')
                return
            selected_count, selected_label = self._select_preview_waveforms_by_group(fig, selected_values)
            if selected_count == 0:
                self._set_preview_search_status(fig, f'No visible waveforms matched {selected_label}', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Selected {selected_count} {selected_label} waveform(s)', color='#1f4e79')

        def toggle_selected_preview_user4_qt():
            selected_count = self._toggle_selected_preview_user4(fig, preview_index)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No selected waveforms to flip', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Flipped polarity for {selected_count} waveform(s)', color='#1f4e79')

        def clear_selected_preview_user4_qt():
            selected_count = self._clear_selected_preview_user4(fig, preview_index)
            if selected_count == 0:
                self._set_preview_search_status(fig, 'No flipped waveforms to clear', color='#8b0000')
                return
            self._set_preview_search_status(fig, f'Cleared flip for {selected_count} waveform(s)', color='#1f4e79')

        def run_preview_stack_qt():
            parent_window = getattr(getattr(fig.canvas, 'manager', None), 'window', None)
            options = self._prompt_preview_stack_options(preview_index, parent_window=parent_window)
            if options is None:
                self._set_preview_search_status(fig, 'Stack cancelled', color='#8b0000')
                return
            success, message, metadata = self._run_preview_stack(fig, preview_index, options)
            self._set_preview_search_status(fig, message, color='#1f4e79' if success else '#8b0000')
            if success and metadata is not None:
                print(f"Saved preview stack figure: {metadata['outputs']['png']}")
                print(f"Saved preview stack SAC: {metadata['outputs']['sac']}")
                print(f"Saved preview stack text: {metadata['outputs']['txt']}")
                print(f"Saved preview stack metadata: {metadata['outputs']['json']}")

        controls['amplitude_widget'] = add_line_edit(row1, 'Amp', f'{self.preview_amplitude_scale:g}', width=64)
        controls['amplitude_preset_widget'] = add_combo_box(
            row1,
            'AmpP',
            ['AmpP'],
            'AmpP',
            width=88,
        )

        def apply_preview_amplitude_qt():
            amp_text = controls['amplitude_widget'].text().strip()
            try:
                new_scale = float(amp_text)
            except ValueError:
                self._set_preview_search_status(fig, 'Amp must be numeric', color='#8b0000')
                return
            self._set_preview_amplitude(fig, preview_index, new_scale)

        def apply_preview_amplitude_preset_qt(index=None):
            combo = controls['amplitude_preset_widget']
            preset_value = combo.currentData()
            if preset_value is None:
                return
            self._set_preview_amplitude(fig, preview_index, preset_value)

        def add_preview_amplitude_preset_qt():
            amp_text = controls['amplitude_widget'].text().strip()
            normalized = self._normalize_preview_amplitude_preset(amp_text)
            if normalized is None:
                self._set_preview_search_status(fig, 'Amp preset must be numeric', color='#8b0000')
                return
            if normalized not in self.preview_amplitude_presets:
                self.preview_amplitude_presets.append(normalized)
                self.preview_amplitude_presets = sorted(set(self.preview_amplitude_presets))
                self._save_preview_amplitude_presets()
            self._refresh_preview_amplitude_preset_combo(fig, select_value=normalized)
            self._set_preview_search_status(fig, f'Added amp preset {normalized:g}', color='#1f4e79')

        def remove_preview_amplitude_preset_qt():
            combo = controls['amplitude_preset_widget']
            preset_value = combo.currentData()
            if preset_value is None:
                self._set_preview_search_status(fig, 'Choose an amp preset to remove', color='#8b0000')
                return
            self.preview_amplitude_presets = [
                value for value in self.preview_amplitude_presets
                if abs(float(value) - float(preset_value)) >= 1e-9
            ]
            if not self.preview_amplitude_presets:
                self.preview_amplitude_presets = self._default_preview_amplitude_presets()
            self._save_preview_amplitude_presets()
            self._refresh_preview_amplitude_preset_combo(fig)
            self._set_preview_search_status(fig, f'Removed amp preset {float(preset_value):g}', color='#1f4e79')

        add_button(row1, 'A-', lambda: self._adjust_preview_amplitude(fig, preview_index, -self.preview_amplitude_step), width=44)
        add_button(row1, 'A+', lambda: self._adjust_preview_amplitude(fig, preview_index, self.preview_amplitude_step), width=44)
        add_button(row1, 'A=', apply_preview_amplitude_qt, width=44)
        add_button(row1, '+', add_preview_amplitude_preset_qt, width=28)
        add_button(row1, '-', remove_preview_amplitude_preset_qt, width=28)
        add_button(row1, 'Sel*', select_preview_group_qt, width=52)
        add_button(row1, 'U1+', apply_selected_preview_user1_qt, width=52)
        add_button(row1, 'U1-', clear_selected_preview_user1_qt, width=52)
        add_button(row1, 'U5+', apply_selected_preview_user5_qt, width=52)
        add_button(row1, 'U5-', clear_selected_preview_user5_qt, width=52)
        add_button(row1, 'Flip', toggle_selected_preview_user4_qt, width=54)
        add_button(row1, 'F-', clear_selected_preview_user4_qt, width=44)
        row1.addStretch(1)

        controls['curve_pick_request_widget'] = add_line_edit(row2, 'pk', '', width=90)
        controls['peak_half_window_widget'] = add_line_edit(row2, 'pkW', f'{self.preview_peak_half_window_default:g}', width=54)

        def apply_preview_peak_half_window_qt():
            self._set_preview_peak_half_window_default(fig, controls['peak_half_window_widget'].text())

        peak_pick_mode_button = add_button(
            row2,
            self._preview_peak_pick_mode_label(),
            lambda: self._toggle_preview_peak_pick_mode(fig),
            width=68,
        )

        add_button(row2, 'R', lambda: self._restore_last_preview_m(fig, preview_index), width=40)
        add_button(row2, 'RA', lambda: self._restore_all_preview_m(fig, preview_index), width=48)
        layout_button = add_button(row2, 'Even', lambda: self._toggle_preview_layout_qt(fig, preview_index), width=62)
        viewmode_button = add_button(row2, self._preview_view_mode_label(), lambda: self._toggle_preview_view_mode_qt(fig, preview_index), width=62)
        add_button(row2, 'C', lambda: self._save_preview_snapshot(fig, preview_index), width=40)
        add_button(row2, 'Std', lambda: self._save_standard_preview_exports(fig, preview_index), width=48)
        add_button(row2, 'Stack', run_preview_stack_qt, width=58)
        stack_member_pierce_button = None
        stack_preview_display_button = None
        if getattr(self, 'stack_mode', False):
            stack_preview_display_button = add_button(
                row2,
                self._stack_preview_display_button_label(),
                lambda: self._toggle_stack_preview_display_mode(fig, preview_index),
                width=72,
            )
            add_button(row2, 'Stack-', lambda: self._switch_stack_preview_wave(fig, preview_index, -1), width=62)
            add_button(row2, 'Stack+', lambda: self._switch_stack_preview_wave(fig, preview_index, 1), width=62)
            stack_member_pierce_button = add_button(
                row2,
                self._stack_member_pierce_button_label(),
                lambda: self._toggle_stack_member_pierce_points(fig, preview_index),
                width=82,
            )
        add_button(row2, 'V', lambda: self.plot_compare_preview(preview_index, use_default_profiles=True), width=40)
        curve_pick_button = add_button(
            row2,
            'P',
            lambda: self._start_preview_curve_pick(fig, self._preview_axes(fig)[0]),
            width=40
        )
        add_button(row2, 'A', lambda: self._apply_preview_curve_alignment(fig, preview_index), width=40)
        keepmode_button = add_button(row2, self._preview_keep_mode_label(), lambda: self._toggle_preview_keep_mode(fig), width=64)
        add_button(row2, 'K', lambda: self._keep_preview_waveforms_by_mode(fig, preview_index), width=40)
        rect_button = add_button(row2, 'Rect', lambda: self._toggle_pierce_selector_mode(fig, 'rect'), width=56)
        circle_button = add_button(row2, 'Circle', lambda: self._toggle_pierce_selector_mode(fig, 'ellipse'), width=60)
        fixrange_button = add_button(row2, 'FixRange', lambda: self._toggle_preview_pierce_range_lock(fig), width=72)
        group_overlay_button = add_button(row2, 'Grp#', lambda: self._toggle_preview_group_overlay(fig, preview_index), width=54)
        ungrouped_button = add_button(row2, 'Ungrp', lambda: self._toggle_preview_ungrouped_only(fig, preview_index), width=60)
        add_button(row2, 'D', lambda: self._drop_preview_marked_waveforms(fig, preview_index), width=40)
        row2.addStretch(1)

        controls['group_save_widget'] = add_line_edit(row3, 'Group', '', width=80)
        controls['group_combo_widget'] = add_combo_box(row3, 'Load', [''], '', width=110)
        controls['delete_marker_widget'] = add_line_edit(row3, 'DelT', '', width=70)

        def save_preview_group_qt():
            success, message = self._save_preview_group(fig, preview_index, controls['group_save_widget'].text())
            self._set_preview_search_status(fig, message, color='#1f4e79' if success else '#8b0000')

        def restore_preview_group_qt():
            success, message = self._restore_preview_group(fig, controls['group_combo_widget'].currentData())
            self._set_preview_search_status(fig, message, color='#1f4e79' if success else '#8b0000')

        def clear_selected_preview_marker_qt():
            cleared_count, marker_keys, error_message = self._clear_selected_preview_marker(
                fig,
                preview_index,
                controls['delete_marker_widget'].text(),
            )
            if error_message is not None:
                self._set_preview_search_status(fig, error_message, color='#8b0000')
                return
            marker_label = ','.join(f't{marker_key}' for marker_key in marker_keys)
            self._set_preview_search_status(
                fig,
                f'Cleared {marker_label} for {cleared_count} marker value(s)',
                color='#1f4e79'
            )

        def delete_preview_group_qt():
            success, message = self._delete_preview_group(fig, controls['group_combo_widget'].currentData())
            self._set_preview_search_status(fig, message, color='#1f4e79' if success else '#8b0000')

        add_button(row3, 'SaveG', save_preview_group_qt, width=62)
        add_button(row3, 'LoadG', restore_preview_group_qt, width=62)
        add_button(row3, 'ClrT', clear_selected_preview_marker_qt, width=56)
        add_button(row3, 'DelG', delete_preview_group_qt, width=56)
        row3.addStretch(1)

        outer.addLayout(row1)
        outer.addLayout(row2)
        outer.addLayout(row3)
        dock.setWidget(container)
        dock.setMaximumHeight(94)
        window.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.preview_control_dock = dock

        status_label = QLabel('')
        status_label.setStyleSheet('color: #444444;')

        fig._preview_controls['x1'] = QtLineEditAdapter(controls['x1_widget'])
        fig._preview_controls['x2'] = QtLineEditAdapter(controls['x2_widget'])
        fig._preview_controls['dx'] = QtLineEditAdapter(controls['dx_widget'])
        fig._preview_controls['amplitude'] = QtLineEditAdapter(controls['amplitude_widget'])
        fig._preview_controls['amplitude_preset_widget'] = controls['amplitude_preset_widget']
        fig._preview_controls['search'] = QtLineEditAdapter(controls['search_widget'])
        fig._preview_controls['std_phases'] = QtLineEditAdapter(controls['std_phases_widget'])
        fig._preview_controls['curve_pick_request'] = QtLineEditAdapter(controls['curve_pick_request_widget'])
        fig._preview_controls['peak_half_window'] = QtLineEditAdapter(controls['peak_half_window_widget'])
        fig._preview_controls['peak_pick_mode_button'] = peak_pick_mode_button
        fig._preview_controls['search_status'] = QtLabelAdapter(status_label)
        fig._preview_controls['preview_index'] = preview_index
        fig._preview_controls['selector_rect_button'] = rect_button
        fig._preview_controls['selector_circle_button'] = circle_button
        fig._preview_controls['fixrange_button'] = fixrange_button
        fig._preview_controls['group_overlay_button'] = group_overlay_button
        fig._preview_controls['ungrouped_button'] = ungrouped_button
        fig._preview_controls['layout_button'] = layout_button
        fig._preview_controls['viewmode_button'] = viewmode_button
        fig._preview_controls['curve_pick_button'] = curve_pick_button
        fig._preview_controls['keepmode_button'] = keepmode_button
        fig._preview_controls['stack_preview_display_button'] = stack_preview_display_button
        fig._preview_controls['stack_member_pierce_button'] = stack_member_pierce_button
        fig._preview_controls['group_combo_widget'] = controls['group_combo_widget']

        axr, axb, axp = self._preview_axes(fig)
        self._refresh_preview_group_combo(fig)
        self._refresh_preview_amplitude_preset_combo(fig, select_value=self.preview_amplitude_scale)
        self._sync_preview_amplitude_control(fig)
        self._sync_preview_peak_half_window_control(fig)

        controls['x1_widget'].returnPressed.connect(apply_preview_window_qt)
        controls['x2_widget'].returnPressed.connect(apply_preview_window_qt)
        controls['dx_widget'].returnPressed.connect(apply_preview_window_qt)
        controls['amplitude_widget'].returnPressed.connect(apply_preview_amplitude_qt)
        controls['amplitude_preset_widget'].currentIndexChanged.connect(apply_preview_amplitude_preset_qt)
        controls['std_phases_widget'].returnPressed.connect(apply_preview_window_qt)
        controls['group_save_widget'].returnPressed.connect(save_preview_group_qt)
        controls['delete_marker_widget'].returnPressed.connect(clear_selected_preview_marker_qt)
        controls['peak_half_window_widget'].returnPressed.connect(apply_preview_peak_half_window_qt)
        controls['pierce_phase_widget'].currentTextChanged.connect(
            lambda text: self._set_preview_pierce_view(
                fig,
                preview_index,
                phase=text,
                model=controls['pierce_model_widget'].currentText(),
            )
        )
        controls['pierce_model_widget'].currentTextChanged.connect(
            lambda text: self._set_preview_pierce_view(
                fig,
                preview_index,
                phase=controls['pierce_phase_widget'].currentText(),
                model=text,
            )
        )
        controls['search_widget'].returnPressed.connect(
            lambda: self._focus_preview_search_match(fig, controls['search_widget'].text())
        )
        controls['curve_pick_request_widget'].returnPressed.connect(
            lambda: self._apply_preview_peak_action(
                fig,
                preview_index,
                request_text=controls['curve_pick_request_widget'].text(),
            )
        )

        def on_preview_key(event):
            raw_key = str(event.key or '')
            key = raw_key.lower()
            def is_shift_letter(letter):
                lower_letter = str(letter).lower()
                return raw_key == lower_letter.upper() or key == f'shift+{lower_letter}'

            if key == ' ' or key == 'space':
                if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
                    self.close_compare_window()
                    self._set_preview_search_status(fig, 'Closed compare window', color='#1f4e79')
                else:
                    self.close_preview_window()
                return
            if getattr(self, 'stack_mode', False) and key in ('w', 's'):
                direction = -1 if key == 'w' else 1
                self._switch_stack_preview_wave(fig, preview_index, direction)
                return
            if key.endswith('up'):
                if 'shift' in key:
                    self._step_preview_selection(fig, 1, mode='range')
                elif 'ctrl' in key or 'control' in key:
                    self._step_preview_selection(fig, 1, mode='add')
                else:
                    self._step_preview_selection(fig, 1, mode='single')
            elif key.endswith('down'):
                if 'shift' in key:
                    self._step_preview_selection(fig, -1, mode='range')
                elif 'ctrl' in key or 'control' in key:
                    self._step_preview_selection(fig, -1, mode='add')
                else:
                    self._step_preview_selection(fig, -1, mode='single')
            elif key.endswith('left') or key.endswith('right'):
                direction = 1 if key.endswith('left') else -1
                if 'shift' in key:
                    step_samples = self.preview_alignment_nudge_steps['large']
                elif 'ctrl' in key or 'control' in key:
                    step_samples = self.preview_alignment_nudge_steps['fine']
                else:
                    step_samples = self.preview_alignment_nudge_steps['normal']
                self._nudge_preview_reference_times(
                    fig,
                    preview_index,
                    direction * step_samples,
                    use_selected_set=True
                )
            elif key == 'm':
                self._mark_selected_preview_as_m(fig)
            elif key == 'n':
                selected_count, enabled = self._toggle_selected_preview_user5(fig)
                if selected_count == 0:
                    self._set_preview_search_status(fig, 'No selected waveforms for U5 toggle', color='#8b0000')
                else:
                    action_label = 'U5+ applied to' if enabled else 'U5- cleared for'
                    self._set_preview_search_status(fig, f'{action_label} {selected_count} waveform(s)', color='#1f4e79')
            elif key == 'd':
                self._drop_preview_marked_waveforms(fig, preview_index)
            elif key == 'r':
                self._restore_last_preview_m(fig, preview_index)
            elif key == 'c':
                self._save_preview_snapshot(fig, preview_index)
            elif key == 'v':
                self.plot_compare_preview(preview_index, use_default_profiles=True)
            elif key == 'p':
                if axr is not None:
                    self._start_preview_curve_pick(fig, axr)
            elif key == 'a':
                self._apply_preview_curve_alignment(fig, preview_index)
            elif key == 'k' or is_shift_letter('k'):
                self._keep_preview_waveforms_by_mode(fig, preview_index)
            elif key == 'h':
                selected_count, enabled = self._toggle_selected_preview_user1(fig)
                if selected_count == 0:
                    self._set_preview_search_status(fig, 'No selected waveforms for U1 toggle', color='#8b0000')
                else:
                    action_label = 'U1+ applied to' if enabled else 'U1- cleared for'
                    self._set_preview_search_status(fig, f'{action_label} {selected_count} waveform(s)', color='#1f4e79')
            elif key == 'l' or is_shift_letter('l'):
                selected_count, state_label, target_scope, error_message = self._toggle_preview_selection_by_selected_user_states(fig)
                if error_message is not None:
                    self._set_preview_search_status(fig, error_message, color='#8b0000')
                else:
                    scope_label = 'complement' if target_scope == 'complement' else 'all'
                    self._set_preview_search_status(
                        fig,
                        f'Selected {selected_count} {scope_label} {state_label} waveform(s)',
                        color='#1f4e79'
                    )
            elif key == 'x':
                cleared_count, marker_keys, error_message = self._clear_selected_preview_marker(
                    fig,
                    preview_index,
                    controls['delete_marker_widget'].text(),
                )
                if error_message is not None:
                    self._set_preview_search_status(fig, error_message, color='#8b0000')
                else:
                    marker_label = ','.join(f't{marker_key}' for marker_key in marker_keys)
                    self._set_preview_search_status(
                        fig,
                        f'Cleared {marker_label} for {cleared_count} marker value(s)',
                        color='#1f4e79'
                    )
            elif key == 'f':
                self._focus_preview_search_match(fig, controls['search_widget'].text())
            elif key == 'j':
                wave_name = self.active_preview_wave_name(fig)
                if not wave_name:
                    self._set_preview_search_status(fig, 'No active preview waveform', color='#8b0000')
                    return
                jumped = self.jump_from_preview_to_wave_name(wave_name, refresh=True)
                if jumped:
                    station_name = self.current_pick_station_name or wave_name
                    self._set_preview_search_status(fig, f'Jumped to {station_name}', color='#1f4e79')
                    if callable(self.jump_status_callback):
                        self.jump_status_callback(wave_name)

        def on_preview_click(event):
            curve_state = self._preview_curve_pick_state(fig)
            if curve_state.get('active', False):
                if event.inaxes != axr:
                    return
                if self._is_left_click(event) and event.xdata is not None and event.ydata is not None:
                    curve_state['points'].append((float(event.xdata), float(event.ydata)))
                    self._refresh_preview_curve_artist(fig, axr)
                    fig.canvas.draw_idle()
                    self._set_preview_search_status(
                        fig,
                        f'Preview pick points: {len(curve_state["points"])}',
                        color='#1f4e79'
                    )
                    return
                if self._is_curve_finish_click(event):
                    self._finish_preview_curve_pick(fig)
                    return
            if self.preview_pierce_selection_mode != 'point':
                if event.inaxes in [axb, axp]:
                    return
                if event.inaxes == axr and self._is_left_click(event):
                    mode_label = {
                        'rect': 'Rect',
                        'ellipse': 'Circle',
                    }.get(self.preview_pierce_selection_mode, self.preview_pierce_selection_mode)
                    self._set_preview_search_status(
                        fig,
                        f'{mode_label} mode active; switch back to point first',
                        color='#8b0000'
                    )
                    return
            if not self._is_left_click(event):
                return
            preview_state = getattr(fig, '_preview_state', None)
            if preview_state is None or event.ydata is None:
                return
            if event.inaxes == axp:
                selected_index = self._preview_index_from_pierce_click(preview_state, event)
                if selected_index is None:
                    return
            else:
                if event.inaxes not in [axr, axb]:
                    return
                if event.inaxes == axb:
                    y_values = np.asarray(
                        preview_state.get('azimuth_y_values', preview_state['evtdata'].gcarc),
                        dtype=float,
                    )
                else:
                    y_values = np.asarray(preview_state.get('y_values', preview_state['evtdata'].gcarc), dtype=float)
                selected_index = int(np.argmin(np.abs(y_values - event.ydata)))
            if self._event_has_modifier(event, 'shift'):
                self._update_preview_selection(fig, selected_index, mode='range')
            elif self._event_has_modifier(event, 'ctrl'):
                self._update_preview_selection(fig, selected_index, mode='toggle')
            else:
                self._update_preview_selection(fig, selected_index, mode='single')

        fig.canvas.mpl_connect('key_press_event', on_preview_key)
        fig.canvas.mpl_connect('button_press_event', on_preview_click)
        try:
            fig.canvas.setFocus()
        except Exception:
            pass
        self._update_preview_mode_button_styles(fig)

    def _set_preview_amplitude(self, fig, preview_index, new_scale):
        clamped_scale = min(
            self.preview_amplitude_max,
            max(self.preview_amplitude_min, float(new_scale))
        )
        if abs(clamped_scale - self.preview_amplitude_scale) < 1e-9:
            self._sync_preview_amplitude_control(fig)
            self._set_preview_search_status(fig, f'Amp x{self.preview_amplitude_scale:.1f}', color='#1f4e79')
            return
        self.preview_amplitude_scale = clamped_scale
        self._update_preview_hidden_history_state()
        self._refresh_preview_figure(fig, preview_index)
        self._sync_preview_amplitude_control(fig)
        self._set_preview_search_status(fig, f'Amp x{self.preview_amplitude_scale:.1f}', color='#1f4e79')

    def _adjust_preview_amplitude(self, fig, preview_index, delta):
        self._set_preview_amplitude(fig, preview_index, self.preview_amplitude_scale + delta)

    def _toggle_preview_layout_qt(self, fig, preview_index):
        self.preview_trace_layout_mode = 'even' if self.preview_trace_layout_mode == 'real' else 'real'
        self._refresh_preview_figure(fig, preview_index)
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            self.plot_compare_preview(preview_index)
        self._set_preview_search_status(fig, f'Layout: {self._preview_layout_summary()}', color='#1f4e79')

    def _toggle_preview_view_mode_qt(self, fig, preview_index):
        self.preview_view_mode = 'tall' if self.preview_view_mode == 'wide' else 'wide'
        self._close_preview_control_dock()
        try:
            plt.close(fig)
        except Exception:
            pass
        self.plot_preview(preview_index)

    def _event_has_modifier(self, event, modifier):
        key = str(getattr(event, 'key', '') or '').lower()
        if modifier == 'ctrl':
            return 'ctrl' in key or 'control' in key
        return modifier in key

    def _set_preview_search_status(self, fig, text, color='#444444'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is not None:
            control_status_text = preview_state.get('control_status_text')
            if control_status_text is not None:
                control_status_text.set_text(str(text))
                control_status_text.set_color(color)
                fig.canvas.draw_idle()
                return
        controls = getattr(fig, '_preview_controls', {})
        status_text = controls.get('search_status')
        if status_text is None:
            return
        status_text.set_text(text)
        status_text.set_color(color)
        fig.canvas.draw_idle()

    def _preview_match_text(self, meta):
        parts = [
            str(meta.get('name', '')),
            str(meta.get('wave_name', '')),
        ]
        return ' '.join(parts).lower()

    def _focus_preview_wave_name(self, fig, wave_name):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None or not wave_name:
            return False
        metadata = preview_state.get('metadata', [])
        for idx, meta in enumerate(metadata):
            if meta.get('wave_name') == wave_name:
                self._update_preview_selection(fig, idx, mode='single')
                return True
        return False

    def _sync_preview_to_current_pick(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        search_box = controls.get('search')
        station_name = self.current_pick_station_name or ''
        if search_box is not None and station_name:
            search_box.set_val(station_name)
        if self._focus_preview_wave_name(fig, self.current_pick_wave_name):
            self._set_preview_search_status(fig, f'Picked: {station_name}', color='#1f4e79')
            return
        if station_name:
            self._focus_preview_search_match(fig, station_name)

    def _focus_preview_search_match(self, fig, query):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        query = str(query or '').strip().lower()
        if query == '':
            self._set_preview_search_status(fig, 'Search cleared')
            return
        metadata = preview_state.get('metadata', [])
        matches = [
            idx for idx, meta in enumerate(metadata)
            if query in self._preview_match_text(meta)
        ]
        if not matches:
            fig._preview_search = {
                'query': query,
                'matches': [],
                'position': -1,
            }
            self._set_preview_search_status(fig, f'No match: {query}', color='#b22222')
            return

        previous_search = getattr(fig, '_preview_search', None) or {}
        if previous_search.get('query') == query and previous_search.get('matches') == matches:
            next_position = (previous_search.get('position', -1) + 1) % len(matches)
        else:
            next_position = 0
        target_index = matches[next_position]
        fig._preview_search = {
            'query': query,
            'matches': matches,
            'position': next_position,
        }
        self._update_preview_selection(fig, target_index, mode='single')
        target_name = metadata[target_index].get('name', '')
        self._set_preview_search_status(
            fig,
            f'{next_position + 1}/{len(matches)}  {target_name}',
            color='#1f4e79'
        )

    def _apply_preview_selection(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        total = len(preview_state['lines'])
        if total == 0:
            return
        selected_indices = {
            idx for idx in preview_state.get('selected_indices', {0})
            if 0 <= idx < total
        }
        if not selected_indices:
            selected_indices = {0}
        active_index = max(0, min(preview_state.get('active_index', 0), total - 1))
        anchor_index = max(0, min(preview_state.get('anchor_index', active_index), total - 1))
        preview_state['selected_indices'] = selected_indices
        preview_state['active_index'] = active_index
        preview_state['anchor_index'] = anchor_index
        self._sync_pick_highlight_from_preview_selection(fig)
        scatter_colors = []
        for index, line in enumerate(preview_state['lines']):
            meta = preview_state['metadata'][index]
            is_selected = index in selected_indices
            line_color, line_width = self._preview_wave_colors(meta, is_selected)
            self._apply_preview_line_style(line, meta, is_selected, line_color, line_width)
            scatter_colors.append(
                line_color if (
                    is_selected
                    or meta.get('stack_preview_role') == 'stack'
                    or meta.get('is_marked_m', False)
                    or meta.get('is_user1_marked', False)
                    or meta.get('is_user5_marked', False)
                    or meta.get('is_user4_marked', False)
                ) else '#1f77b4'
            )
        scatter_artist = preview_state.get('scatter')
        if scatter_artist is not None:
            full_indices = getattr(scatter_artist, '_dephasekit_preview_full_indices', None)
            if full_indices is None:
                scatter_artist.set_facecolors([mcolors.to_rgba(color) for color in scatter_colors])
                scatter_artist.set_edgecolors([mcolors.to_rgba(color) for color in scatter_colors])
            else:
                masked_colors = [
                    scatter_colors[int(index)]
                    for index in np.asarray(full_indices, dtype=int)
                    if 0 <= int(index) < len(scatter_colors)
                ]
                scatter_artist.set_facecolors([mcolors.to_rgba(color) for color in masked_colors])
                scatter_artist.set_edgecolors([mcolors.to_rgba(color) for color in masked_colors])
        evtdata = preview_state['evtdata']
        meta = preview_state['metadata'][active_index]
        marker_color, _marker_width = self._preview_wave_colors(meta, True)
        selected_marker = preview_state.get('selected_marker')
        if selected_marker is not None:
            if getattr(self, 'stack_mode', False) and meta.get('stack_preview_role') == 'stack':
                selected_marker.set_data([], [])
            else:
                selected_marker.set_color(marker_color)
                selected_marker.set_data(
                    [evtdata.az[active_index]],
                    [preview_state.get('azimuth_y_values', preview_state.get('y_values', evtdata.gcarc))[active_index]]
                )
        pierce_state = preview_state.get('pierce_state')
        if pierce_state is not None:
            base_scatter = pierce_state.get('base_scatter')
            if base_scatter is not None:
                base_colors = [
                    self._pierce_record_style(record.wave_name, selected=False)[0]
                    for record in pierce_state.get('records', [])
                ]
                base_scatter.set_facecolors([mcolors.to_rgba(color) for color in base_colors])
                base_scatter.set_edgecolors([mcolors.to_rgba(color) for color in base_colors])
            selected_wave_names = {
                preview_state['metadata'][idx].get('wave_name')
                for idx in selected_indices
                if 0 <= idx < len(preview_state['metadata'])
            }
            selected_record_wave_names = self._preview_selected_record_wave_names(selected_wave_names)
            selected_records = [
                record for record in pierce_state.get('records', [])
                if record.wave_name in selected_record_wave_names
            ]
            highlight_scatter = pierce_state.get('highlight_scatter')
            if highlight_scatter is not None:
                if selected_records:
                    selected_colors = [
                        self._pierce_record_style(record.wave_name, selected=True)[1]
                        for record in selected_records
                    ]
                    highlight_scatter.set_offsets(
                        np.column_stack((
                            [record.longitude for record in selected_records],
                            [record.latitude for record in selected_records],
                        ))
                    )
                    highlight_scatter.set_facecolors([mcolors.to_rgba(color) for color in selected_colors])
                    highlight_scatter.set_edgecolors([mcolors.to_rgba(color) for color in selected_colors])
                else:
                    highlight_scatter.set_offsets(np.empty((0, 2)))
            axes = pierce_state.get('axes')
            label_artists = getattr(axes, '_dephasekit_group_label_artists', {}) if axes is not None else {}
            selected_group_numbers = {
                self._group_number_from_record(record)
                for record in selected_records
            }
            if not selected_group_numbers:
                selected_group_numbers = {
                    self._group_number_from_wave_name(preview_state['metadata'][idx].get('wave_name'))
                    for idx in selected_indices
                    if 0 <= idx < len(preview_state['metadata'])
                }
            for group_number, text_artist in label_artists.items():
                is_selected_group = group_number in selected_group_numbers
                group_color = self._preview_group_color(f'group{group_number}')
                text_artist.set_color(group_color)
                text_artist.set_fontweight('bold' if is_selected_group else 'normal')
                text_artist.set_bbox({
                    'boxstyle': 'round,pad=0.18',
                    'facecolor': group_color if is_selected_group else 'white',
                    'edgecolor': group_color,
                    'linewidth': 1.2 if is_selected_group else 0.9,
                    'alpha': 0.95 if is_selected_group else 0.92,
                })
        label_suffix = ''
        if meta.get('is_marked_m', False):
            label_suffix += '    [purple]'
        if meta.get('is_user1_marked', False):
            label_suffix += '    [user1]'
        if meta.get('is_user5_marked', False):
            label_suffix += '    [user5]'
        stack_summary = meta.get('stack_summary', '')
        stack_suffix = f"    Stack: {stack_summary}" if stack_summary else ''
        stack_crustal_text = self._stack_crustal_summary_text(meta.get('wave_name'))
        stack_crustal_suffix = f"    Thickness: {stack_crustal_text}" if stack_crustal_text else ''
        if getattr(self, 'preview_view_mode', 'wide') == 'tall':
            info_message = (
                f"{meta['name']}    Dist: {meta['gcarc']:.2f}°    Az: {meta['az']:.2f}°"
                f"    {self.current_wave_theory_delta_text(meta.get('wave_name'))}"
                f"    {self._preview_layout_summary()}/{self._preview_view_mode_label()}{label_suffix}"
                f"    Sel: {len(selected_indices)}{stack_suffix}{stack_crustal_suffix}"
            )
        else:
            info_message = (
                f"{meta['name']}    Epicenter distance: {meta['gcarc']:.2f}°    Azimuth: {meta['az']:.2f}°"
                f"    {self.current_wave_theory_delta_text(meta.get('wave_name'))}"
                f"    Layout: {self._preview_layout_summary()}    View: {self._preview_view_mode_label()}{label_suffix}"
                f"    Selected: {len(selected_indices)}{stack_suffix}{stack_crustal_suffix}"
            )
        preview_state['info_text'].set_color(marker_color)
        preview_state['info_text'].set_text(info_message)
        control_status_text = preview_state.get('control_status_text')
        if control_status_text is not None:
            hidden_rounds, hidden_count = self._preview_hidden_summary()
            even_gap_text = f"    Even gap: {self.preview_even_spacing_step:.1f}" if self.preview_trace_layout_mode == 'even' else ''
            reference_mode_text = f"    Ref: {self._preview_reference_mode_label(preview_state.get('tmarker'))}"
            if getattr(self, 'preview_view_mode', 'wide') == 'tall':
                status_message = (
                    f"W: {preview_state.get('window_width', 0):g}s    dx: {preview_state.get('tick_interval', 0):g}"
                    f" ({preview_state.get('tick_mode', 'auto')})    ph: {self.standard_export_phase_tokens or 'none'}"
                    f"    Amp x{self.preview_amplitude_scale:.1f}    {self._preview_layout_summary()}/{self._preview_view_mode_label()}"
                    f"{even_gap_text}{reference_mode_text}    Hidden: {hidden_count}/{hidden_rounds}"
                )
            else:
                status_message = (
                    f"Window width: {preview_state.get('window_width', 0):g}s    dx: {preview_state.get('tick_interval', 0):g}"
                    f" ({preview_state.get('tick_mode', 'auto')})    ph: {self.standard_export_phase_tokens or 'none'}"
                    f"    Amp: x{self.preview_amplitude_scale:.1f}    Layout: {self._preview_layout_summary()}"
                    f"    View: {self._preview_view_mode_label()}{even_gap_text}{reference_mode_text}"
                    f"    Hidden: {hidden_count} in {hidden_rounds} round(s)"
                )
            control_status_text.set_text(status_message)
        fig.canvas.draw_idle()

    def _update_preview_selection(self, fig, target_index, mode='single'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        total = len(preview_state['lines'])
        if total == 0:
            return
        target_index = max(0, min(target_index, total - 1))
        anchor_index = preview_state.get('anchor_index', preview_state.get('active_index', target_index))
        selected_indices = set(preview_state.get('selected_indices', {target_index}))
        if mode == 'single':
            selected_indices = {target_index}
            anchor_index = target_index
        elif mode == 'range':
            start = min(anchor_index, target_index)
            end = max(anchor_index, target_index)
            selected_indices = set(range(start, end + 1))
        elif mode == 'toggle':
            if target_index in selected_indices and len(selected_indices) > 1:
                selected_indices.remove(target_index)
            else:
                selected_indices.add(target_index)
            anchor_index = target_index
        elif mode == 'add':
            selected_indices.add(target_index)
        preview_state['selected_indices'] = selected_indices
        preview_state['active_index'] = target_index
        preview_state['anchor_index'] = anchor_index
        self._apply_preview_selection(fig)

    def _step_preview_selection(self, fig, step, mode='single'):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        current_index = preview_state.get('active_index', 0)
        self._update_preview_selection(fig, current_index + step, mode=mode)

    def _set_preview_selected_indices(self, fig, indices):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        metadata = preview_state.get('metadata', [])
        valid_indices = sorted({
            int(index) for index in indices
            if 0 <= int(index) < len(metadata) and metadata[int(index)].get('wave_name')
        })
        if not valid_indices:
            return 0
        preview_state['selected_indices'] = set(valid_indices)
        preview_state['active_index'] = valid_indices[0]
        preview_state['anchor_index'] = valid_indices[0]
        self._apply_preview_selection(fig)
        return len(valid_indices)

    def _update_preview_selection_by_wave_names(self, fig, wave_names):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        indices = []
        for idx, meta in enumerate(preview_state.get('metadata', [])):
            if meta.get('wave_name') in wave_names:
                indices.append(idx)
        if not indices:
            return
        preview_state['selected_indices'] = set(indices)
        preview_state['active_index'] = indices[0]
        preview_state['anchor_index'] = indices[0]

    def _preview_primary_user_state_for_meta(self, meta):
        if meta.get('is_user1_marked', False):
            return 'user1'
        if meta.get('is_user4_marked', False):
            return 'user4'
        if meta.get('is_marked_m', False):
            return 'user2'
        if meta.get('is_user5_marked', False):
            return 'user5'
        return None

    def _preview_indices_matching_user_states(self, preview_state, state_keys):
        wanted = set(state_keys or [])
        if not wanted:
            return set()
        return {
            idx for idx, meta in enumerate(preview_state.get('metadata', []))
            if (
                meta.get('wave_name')
                and self._preview_primary_user_state_for_meta(meta) in wanted
            )
        }

    def _preview_user_state_label(self, state_keys):
        ordered_keys = [
            state_key for state_key in ('user1', 'user4', 'user2', 'user5')
            if state_key in set(state_keys or [])
        ]
        return ' + '.join(ordered_keys)

    def _toggle_preview_selection_by_selected_user_states(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0, None, None, 'Preview state unavailable'
        metadata = preview_state.get('metadata', [])
        all_indices = {
            idx for idx, meta in enumerate(metadata)
            if meta.get('wave_name')
        }
        selected_indices = {
            idx for idx in preview_state.get('selected_indices', set())
            if idx in all_indices
        }
        if not selected_indices:
            return 0, None, None, 'No selected waveforms'

        target_state_keys = None
        target_indices = set()
        target_scope = 'matching'
        previous_state_keys = tuple(preview_state.get('_user_state_toggle_keys') or ())
        if previous_state_keys:
            matching_indices = self._preview_indices_matching_user_states(preview_state, previous_state_keys)
            complement_indices = all_indices - matching_indices
            if selected_indices == matching_indices:
                target_state_keys = previous_state_keys
                target_indices = complement_indices
                target_scope = 'complement'
            elif selected_indices == complement_indices:
                target_state_keys = previous_state_keys
                target_indices = matching_indices
                target_scope = 'matching'

        if target_state_keys is None:
            selected_state_set = set()
            for idx in sorted(selected_indices):
                state_key = self._preview_primary_user_state_for_meta(metadata[idx])
                if state_key is not None:
                    selected_state_set.add(state_key)
            target_state_keys = tuple(
                state_key for state_key in ('user1', 'user4', 'user2', 'user5')
                if state_key in selected_state_set
            )
            if not target_state_keys:
                return 0, None, None, 'Selected waveforms have no user state'
            target_indices = self._preview_indices_matching_user_states(preview_state, target_state_keys)
            target_scope = 'matching'

        state_label = self._preview_user_state_label(target_state_keys)
        if not target_indices:
            return 0, state_label, target_scope, f'No waveform complement for {state_label}'
        selected_count = self._set_preview_selected_indices(fig, target_indices)
        preview_state['_user_state_toggle_keys'] = tuple(target_state_keys)
        preview_state['_user_state_toggle_scope'] = target_scope
        return selected_count, state_label, target_scope, None

    def _select_preview_purple_waveforms(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        purple_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name') and meta.get('is_marked_m', False)
        ]
        if not purple_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, purple_wave_names)
        self._apply_preview_selection(fig)
        return len(purple_wave_names)

    def _select_preview_user1_waveforms(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        user1_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name') and meta.get('is_user1_marked', False)
        ]
        if not user1_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, user1_wave_names)
        self._apply_preview_selection(fig)
        return len(user1_wave_names)

    def _select_preview_user5_waveforms(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        user5_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name') and meta.get('is_user5_marked', False)
        ]
        if not user5_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, user5_wave_names)
        self._apply_preview_selection(fig)
        return len(user5_wave_names)

    def _select_preview_user4_waveforms(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        user4_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if meta.get('wave_name') and meta.get('is_user4_marked', False)
        ]
        if not user4_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, user4_wave_names)
        self._apply_preview_selection(fig)
        return len(user4_wave_names)

    def _select_preview_waveforms_by_marker(self, fig, marker_key):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        marker_key = self._normalize_marker_key(marker_key)
        selected_wave_names = [
            meta.get('wave_name')
            for meta in preview_state.get('metadata', [])
            if (
                meta.get('wave_name')
                and not math.isnan(self.markers.get(marker_key, {}).get(meta.get('wave_name'), math.nan))
            )
        ]
        if not selected_wave_names:
            return 0
        self._update_preview_selection_by_wave_names(fig, selected_wave_names)
        self._apply_preview_selection(fig)
        return len(selected_wave_names)

    def _preview_group_wave_names(self, preview_state, group_key):
        group = str(group_key or '').strip().lower()
        if group in ('purple', 'm', 'preview_m'):
            return [
                meta.get('wave_name')
                for meta in preview_state.get('metadata', [])
                if meta.get('wave_name') and meta.get('is_marked_m', False) and not meta.get('is_user1_marked', False)
            ], 'purple'
        if group in ('user1', 'u1', 'g'):
            return [
                meta.get('wave_name')
                for meta in preview_state.get('metadata', [])
                if meta.get('wave_name') and meta.get('is_user1_marked', False)
            ], 'user1'
        if group in ('user4', 'u4', 'flip', 'f'):
            return [
                meta.get('wave_name')
                for meta in preview_state.get('metadata', [])
                if meta.get('wave_name') and meta.get('is_user4_marked', False)
            ], 'user4'
        if group in ('user5', 'u5', 'c'):
            return [
                meta.get('wave_name')
                for meta in preview_state.get('metadata', [])
                if meta.get('wave_name') and meta.get('is_user5_marked', False) and not meta.get('is_user1_marked', False)
            ], 'user5'
        marker_key = self._normalize_marker_key(group)
        if marker_key in self.marker_styles:
            return [
                meta.get('wave_name')
                for meta in preview_state.get('metadata', [])
                if (
                    meta.get('wave_name')
                    and not math.isnan(self.markers.get(marker_key, {}).get(meta.get('wave_name'), math.nan))
                )
            ], f't{marker_key}'
        return [], None

    def _select_preview_waveforms_by_group(self, fig, group_key):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0, None
        group_keys = group_key if isinstance(group_key, (list, tuple, set)) else [group_key]
        selected_wave_names = []
        selected_labels = []
        seen_wave_names = set()
        for raw_group_key in group_keys:
            wave_names, label = self._preview_group_wave_names(preview_state, raw_group_key)
            if label is None:
                continue
            selected_labels.append(label)
            for wave_name in wave_names:
                if wave_name and wave_name not in seen_wave_names:
                    seen_wave_names.add(wave_name)
                    selected_wave_names.append(wave_name)
        if not selected_wave_names:
            return 0, ', '.join(selected_labels) if selected_labels else None
        self._update_preview_selection_by_wave_names(fig, selected_wave_names)
        self._apply_preview_selection(fig)
        return len(selected_wave_names), ' + '.join(selected_labels)

    def _mark_selected_preview_as_m(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        selected_indices = sorted(preview_state.get('selected_indices', []))
        changed = False
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            next_enabled = not self._is_preview_purple_wave(wave_name)
            changed = self._set_user_marker(wave_name, 'user2', next_enabled) or changed
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if changed:
            self._refresh_pick_window_if_available()
            self._apply_preview_selection(fig)

    def _unmark_selected_preview_m(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return
        selected_indices = sorted(preview_state.get('selected_indices', []))
        changed = False
        removed_wave_names = set()
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            changed = self._set_user_marker(wave_name, 'user2', False) or changed
            meta['is_marked_m'] = False
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if changed:
            self._refresh_pick_window_if_available()
            self._apply_preview_selection(fig)

    def _all_current_hidden_wave_names(self):
        return [
            wave_name for wave_name in self.ori_sacnames
            if self._is_preview_hidden_wave(wave_name)
        ]

    def _preview_hidden_summary(self):
        hidden_wave_names = self._all_current_hidden_wave_names()
        hidden_rounds = max(0, len(self.preview_hidden_batches) - 1)
        return hidden_rounds, len(hidden_wave_names)

    def _ensure_preview_hidden_history(self):
        if self.preview_hidden_batches:
            return
        self.preview_hidden_batches = [{
            'hidden_wave_names': list(self._all_current_hidden_wave_names()),
            'amplitude_scale': self.preview_amplitude_scale,
            'selected_wave_names': [],
        }]

    def _update_preview_hidden_history_state(self):
        if not self.preview_hidden_batches:
            return
        self.preview_hidden_batches[-1] = {
            'hidden_wave_names': list(self._all_current_hidden_wave_names()),
            'amplitude_scale': self.preview_amplitude_scale,
            'selected_wave_names': list(self.preview_hidden_batches[-1].get('selected_wave_names', [])),
        }

    def _restore_preview_hidden_set(self, target_hidden_wave_names):
        target_hidden = set(target_hidden_wave_names or [])
        current_hidden = set(self._all_current_hidden_wave_names())
        changed = current_hidden != target_hidden
        if changed:
            self.preview_hidden_wave_names = set(target_hidden)
        return changed, current_hidden, target_hidden

    def _finalize_preview_restore(
            self, fig, preview_index, restored_count, selected_wave_names,
            restored_all, restore_amplitude_scale=None):
        remaining_hidden = self._all_current_hidden_wave_names()
        if restore_amplitude_scale is not None:
            self.preview_amplitude_scale = restore_amplitude_scale
        elif not remaining_hidden and self.preview_amplitude_hidden_mode:
            self.preview_amplitude_scale = self.preview_amplitude_restore_scale
        self.preview_amplitude_hidden_mode = bool(remaining_hidden)
        if selected_wave_names:
            fig._preview_forced_selected_wave_names = list(selected_wave_names)
        self._refresh_pick_window_if_available()
        self._refresh_preview_figure(fig, preview_index)
        restore_scope = 'all hidden' if restored_all else 'last hidden batch'
        self._set_preview_search_status(
            fig,
            f"Restored {restored_count} waveform(s) from {restore_scope}; Amp x{self.preview_amplitude_scale:.1f}",
            color='#1f4e79'
        )

    def _restore_last_preview_m(self, fig, preview_index):
        if len(self.preview_hidden_batches) <= 1:
            self._set_preview_search_status(fig, 'No hidden rounds to restore', color='#8b0000')
            return
        current_hidden = set(self._all_current_hidden_wave_names())
        current_state = self.preview_hidden_batches.pop()
        target_state = self.preview_hidden_batches[-1]
        target_hidden = set(target_state.get('hidden_wave_names', []))
        selected_wave_names = list(current_state.get('selected_wave_names', []))
        if not selected_wave_names:
            selected_wave_names = list(current_hidden - target_hidden)
        restore_amplitude_scale = target_state.get('amplitude_scale', self.preview_amplitude_restore_scale)
        restored_any, current_hidden, target_hidden = self._restore_preview_hidden_set(target_hidden)
        restored_count = len(current_hidden - target_hidden)
        if restored_any:
            self._finalize_preview_restore(
                fig,
                preview_index,
                restored_count,
                selected_wave_names or list(current_hidden - target_hidden),
                restored_all=False,
                restore_amplitude_scale=restore_amplitude_scale,
            )

    def _restore_all_preview_m(self, fig, preview_index):
        current_hidden = set(self._all_current_hidden_wave_names())
        if not current_hidden:
            self._set_preview_search_status(fig, 'No hidden waveforms to restore', color='#8b0000')
            return
        selected_wave_names = []
        if self.preview_hidden_batches:
            selected_wave_names = list(self.preview_hidden_batches[-1].get('selected_wave_names', []))
        if not selected_wave_names:
            selected_wave_names = list(current_hidden)
        restored_any, current_hidden, target_hidden = self._restore_preview_hidden_set([])
        self.preview_hidden_batches = []
        if restored_any:
            self._finalize_preview_restore(
                fig,
                preview_index,
                len(current_hidden - target_hidden),
                selected_wave_names,
                restored_all=True,
                restore_amplitude_scale=self.preview_amplitude_restore_scale,
            )

    def _drop_preview_marked_waveforms(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        hidden_wave_names = []
        selected_wave_names = []
        if preview_state is not None:
            for selected_index in sorted(preview_state.get('selected_indices', [])):
                if selected_index >= len(preview_state['metadata']):
                    continue
                meta = preview_state['metadata'][selected_index]
                wave_name = meta.get('wave_name')
                if not wave_name:
                    continue
                if not self._is_preview_hidden_wave(wave_name):
                    hidden_wave_names.append(wave_name)
                    selected_wave_names.append(wave_name)
        if not hidden_wave_names:
            self._set_preview_search_status(fig, 'No selected waveforms to hide', color='#8b0000')
            return
        self._ensure_preview_hidden_history()
        changed = False
        for wave_name in hidden_wave_names:
            if wave_name not in self.preview_hidden_wave_names:
                self.preview_hidden_wave_names.add(wave_name)
                changed = True
        if changed:
            if not self.preview_amplitude_hidden_mode:
                self.preview_amplitude_restore_scale = self.preview_amplitude_scale
            self.preview_amplitude_scale = 1.0
            self.preview_amplitude_hidden_mode = True
            self.preview_hidden_batches.append({
                'hidden_wave_names': list(self._all_current_hidden_wave_names()),
                'amplitude_scale': self.preview_amplitude_scale,
                'selected_wave_names': list(selected_wave_names),
            })
            self._refresh_pick_window_if_available()
        self._refresh_preview_figure(fig, preview_index)
        self._set_preview_search_status(
            fig,
            f'Hidden {len(hidden_wave_names)} selected waveform(s); Amp x{self.preview_amplitude_scale:.1f}',
            color='#1f4e79'
        )

    def _keep_preview_waveforms_by_mode(self, fig, preview_index):
        if self.preview_keep_selection_mode == 'unselected':
            self._keep_only_unselected_preview_waveforms(fig, preview_index)
            return
        self._keep_only_selected_preview_waveforms(fig, preview_index)

    def _keep_only_selected_preview_waveforms(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return
        metadata = preview_state.get('metadata', [])
        selected_indices = sorted(preview_state.get('selected_indices', []))
        if not selected_indices:
            self._set_preview_search_status(fig, 'No selected waveforms to keep', color='#8b0000')
            return
        selected_wave_names = []
        selected_wave_name_set = set()
        for selected_index in selected_indices:
            if selected_index >= len(metadata):
                continue
            wave_name = metadata[selected_index].get('wave_name')
            if not wave_name:
                continue
            if wave_name not in selected_wave_name_set:
                selected_wave_names.append(wave_name)
                selected_wave_name_set.add(wave_name)
        if not selected_wave_names:
            self._set_preview_search_status(fig, 'No selected waveforms to keep', color='#8b0000')
            return
        hidden_wave_names = []
        for meta in metadata:
            wave_name = meta.get('wave_name')
            if not wave_name or wave_name in selected_wave_name_set:
                continue
            if not self._is_preview_hidden_wave(wave_name):
                hidden_wave_names.append(wave_name)
        if not hidden_wave_names:
            self._set_preview_search_status(fig, 'Selected waveforms already isolated', color='#1f4e79')
            return
        self._ensure_preview_hidden_history()
        changed = False
        for wave_name in hidden_wave_names:
            if wave_name not in self.preview_hidden_wave_names:
                self.preview_hidden_wave_names.add(wave_name)
                changed = True
        if changed:
            if not self.preview_amplitude_hidden_mode:
                self.preview_amplitude_restore_scale = self.preview_amplitude_scale
            self.preview_amplitude_scale = 1.0
            self.preview_amplitude_hidden_mode = True
            self.preview_hidden_batches.append({
                'hidden_wave_names': list(self._all_current_hidden_wave_names()),
                'amplitude_scale': self.preview_amplitude_scale,
                'selected_wave_names': list(selected_wave_names),
            })
            self.current_pick_wave_name = selected_wave_names[0]
            first_index = self._wave_index_by_name(selected_wave_names[0])
            if first_index is not None and first_index < len(self.filenames):
                self.current_pick_station_name = self._wave_display_name(selected_wave_names[0], self.filenames[first_index])
            fig._preview_forced_selected_wave_names = list(selected_wave_names)
            self._refresh_pick_window_if_available(focus_current_wave=True)
        self._refresh_preview_figure(fig, preview_index)
        self._set_preview_search_status(
            fig,
            f'Kept {len(selected_wave_names)} waveform(s); hid {len(hidden_wave_names)} others; Amp x{self.preview_amplitude_scale:.1f}',
            color='#1f4e79'
        )

    def _keep_only_unselected_preview_waveforms(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            self._set_preview_search_status(fig, 'Preview state unavailable', color='#8b0000')
            return
        metadata = preview_state.get('metadata', [])
        selected_indices = {
            idx for idx in preview_state.get('selected_indices', set())
            if 0 <= idx < len(metadata)
        }
        if not metadata:
            self._set_preview_search_status(fig, 'No preview waveforms available', color='#8b0000')
            return
        kept_wave_names = []
        kept_wave_name_set = set()
        hidden_wave_names = []
        for idx, meta in enumerate(metadata):
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            if idx in selected_indices:
                if not self._is_preview_hidden_wave(wave_name):
                    hidden_wave_names.append(wave_name)
                continue
            if wave_name not in kept_wave_name_set:
                kept_wave_names.append(wave_name)
                kept_wave_name_set.add(wave_name)
        if not kept_wave_names:
            self._set_preview_search_status(fig, 'No unselected waveforms to keep', color='#8b0000')
            return
        if not hidden_wave_names:
            self._set_preview_search_status(fig, 'Unselected waveforms already isolated', color='#1f4e79')
            return
        self._ensure_preview_hidden_history()
        changed = False
        for wave_name in hidden_wave_names:
            if wave_name not in self.preview_hidden_wave_names:
                self.preview_hidden_wave_names.add(wave_name)
                changed = True
        if changed:
            if not self.preview_amplitude_hidden_mode:
                self.preview_amplitude_restore_scale = self.preview_amplitude_scale
            self.preview_amplitude_scale = 1.0
            self.preview_amplitude_hidden_mode = True
            self.preview_hidden_batches.append({
                'hidden_wave_names': list(self._all_current_hidden_wave_names()),
                'amplitude_scale': self.preview_amplitude_scale,
                'selected_wave_names': list(kept_wave_names),
            })
            self.current_pick_wave_name = kept_wave_names[0]
            first_index = self._wave_index_by_name(kept_wave_names[0])
            if first_index is not None and first_index < len(self.filenames):
                self.current_pick_station_name = self._wave_display_name(kept_wave_names[0], self.filenames[first_index])
            fig._preview_forced_selected_wave_names = list(kept_wave_names)
            self._refresh_pick_window_if_available(focus_current_wave=True)
        self._refresh_preview_figure(fig, preview_index)
        self._set_preview_search_status(
            fig,
            f'Kept {len(kept_wave_names)} unselected waveform(s); hid {len(hidden_wave_names)} selected; Amp x{self.preview_amplitude_scale:.1f}',
            color='#1f4e79'
        )

    def _apply_selected_preview_user1(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        target_wave_names = []
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            target_wave_names.append(wave_name)
        applied_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user1', True):
                self._set_user_marker(wave_name, 'user2', False)
                self._set_user_marker(wave_name, 'user5', False)
                applied_count += 1
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if target_wave_names:
            self._update_preview_selection_by_wave_names(fig, target_wave_names)
            self._apply_preview_selection(fig)
        if applied_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names)

    def _toggle_selected_preview_user1(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0, None
        target_wave_names = []
        seen_wave_names = set()
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name or wave_name in seen_wave_names:
                continue
            target_wave_names.append(wave_name)
            seen_wave_names.add(wave_name)
        if not target_wave_names:
            return 0, None
        should_enable = not all(self._is_user1_wave(wave_name) for wave_name in target_wave_names)
        changed_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user1', should_enable):
                changed_count += 1
            if should_enable:
                self._set_user_marker(wave_name, 'user2', False)
                self._set_user_marker(wave_name, 'user5', False)
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        self._update_preview_selection_by_wave_names(fig, target_wave_names)
        self._apply_preview_selection(fig)
        if changed_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names), should_enable

    def _apply_selected_preview_user5(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        target_wave_names = []
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            target_wave_names.append(wave_name)
        applied_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user5', True):
                self._set_user_marker(wave_name, 'user2', False)
                applied_count += 1
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if target_wave_names:
            self._update_preview_selection_by_wave_names(fig, target_wave_names)
            self._apply_preview_selection(fig)
        if applied_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names)

    def _toggle_selected_preview_user5(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0, None
        target_wave_names = []
        seen_wave_names = set()
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name or wave_name in seen_wave_names:
                continue
            target_wave_names.append(wave_name)
            seen_wave_names.add(wave_name)
        if not target_wave_names:
            return 0, None
        should_enable = not all(self._is_user5_wave(wave_name) for wave_name in target_wave_names)
        changed_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user5', should_enable):
                changed_count += 1
            if should_enable:
                self._set_user_marker(wave_name, 'user2', False)
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        self._update_preview_selection_by_wave_names(fig, target_wave_names)
        self._apply_preview_selection(fig)
        if changed_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names), should_enable

    def _toggle_selected_preview_user4(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        target_wave_names = []
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            wave_name = preview_state['metadata'][selected_index].get('wave_name')
            if wave_name:
                target_wave_names.append(wave_name)
        if not target_wave_names:
            return 0
        for wave_name in target_wave_names:
            self._toggle_user4_marker(wave_name)
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        self._update_preview_selection_by_wave_names(fig, target_wave_names)
        self._apply_preview_selection(fig)
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_pick_window_if_available()
        self._refresh_compare_for_preview_index(preview_index)
        return len(target_wave_names)

    def _clear_selected_preview_user4(self, fig, preview_index):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        target_wave_names = []
        selected_indices = sorted(preview_state.get('selected_indices', []))
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if wave_name and meta.get('is_user4_marked', False):
                target_wave_names.append(wave_name)
        if not target_wave_names:
            target_wave_names = [
                meta.get('wave_name') for meta in preview_state['metadata']
                if meta.get('wave_name') and meta.get('is_user4_marked', False)
            ]
        if not target_wave_names:
            return 0
        for wave_name in target_wave_names:
            self._set_user_marker(wave_name, 'user4', False)
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        self._update_preview_selection_by_wave_names(fig, target_wave_names)
        self._apply_preview_selection(fig)
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_pick_window_if_available()
        self._refresh_compare_for_preview_index(preview_index)
        return len(target_wave_names)

    def _clear_selected_preview_user1(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        selected_indices = sorted(preview_state.get('selected_indices', []))
        target_wave_names = []
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if wave_name and meta.get('is_user1_marked', False):
                target_wave_names.append(wave_name)
        if not target_wave_names:
            target_wave_names = [
                meta.get('wave_name') for meta in preview_state['metadata']
                if meta.get('wave_name') and meta.get('is_user1_marked', False)
            ]
        cleared_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user1', False):
                cleared_count += 1
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if target_wave_names:
            self._update_preview_selection_by_wave_names(fig, target_wave_names)
            self._apply_preview_selection(fig)
        if cleared_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names)

    def _clear_selected_preview_user5(self, fig):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0
        selected_indices = sorted(preview_state.get('selected_indices', []))
        target_wave_names = []
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if wave_name and meta.get('is_user5_marked', False):
                target_wave_names.append(wave_name)
        if not target_wave_names:
            target_wave_names = [
                meta.get('wave_name') for meta in preview_state['metadata']
                if meta.get('wave_name') and meta.get('is_user5_marked', False)
            ]
        cleared_count = 0
        for wave_name in target_wave_names:
            if self._set_user_marker(wave_name, 'user5', False):
                cleared_count += 1
        for meta in preview_state['metadata']:
            wave_name = meta.get('wave_name')
            meta['is_marked_m'] = self._is_preview_purple_wave(wave_name)
            meta['is_user1_marked'] = self._is_user1_wave(wave_name)
            meta['is_user5_marked'] = self._is_user5_wave(wave_name)
            meta['is_user4_marked'] = self._is_user4_wave(wave_name)
        if target_wave_names:
            self._update_preview_selection_by_wave_names(fig, target_wave_names)
            self._apply_preview_selection(fig)
        if cleared_count:
            self._refresh_pick_window_if_available()
        return len(target_wave_names)

    def _parse_preview_delete_marker_keys(self, marker_text):
        raw_tokens = [
            token.strip()
            for token in str(marker_text or '').replace('，', ',').split(',')
            if token.strip()
        ]
        if not raw_tokens:
            return None, 'Delete marker must be t0-t9 or 0-9; multiple markers can use commas'
        marker_keys = []
        seen = set()
        for raw_token in raw_tokens:
            marker_key = self._normalize_marker_key(raw_token)
            if marker_key not in self.marker_styles:
                return None, 'Delete marker must be t0-t9 or 0-9; multiple markers can use commas'
            if marker_key in seen:
                continue
            seen.add(marker_key)
            marker_keys.append(marker_key)
        return marker_keys, None

    def _clear_selected_preview_marker(self, fig, preview_index, marker_text):
        preview_state = getattr(fig, '_preview_state', None)
        if preview_state is None:
            return 0, None, 'Preview state unavailable'
        marker_keys, error_message = self._parse_preview_delete_marker_keys(marker_text)
        if error_message is not None:
            return 0, None, error_message
        selected_indices = sorted(preview_state.get('selected_indices', []))
        if not selected_indices:
            return 0, marker_keys, 'No selected waveforms to clear'
        target_wave_names = []
        cleared_count = 0
        for selected_index in selected_indices:
            if selected_index >= len(preview_state['metadata']):
                continue
            meta = preview_state['metadata'][selected_index]
            wave_name = meta.get('wave_name')
            if not wave_name:
                continue
            target_wave_names.append(wave_name)
            for marker_key in marker_keys:
                current_value = self.markers.get(marker_key, {}).get(wave_name, math.nan)
                if math.isnan(current_value):
                    continue
                if self._set_wave_marker_time(wave_name, marker_key, math.nan):
                    cleared_count += 1
        if not target_wave_names:
            return 0, marker_keys, 'No selected waveforms to clear'
        active_align_marker = None
        if 0 <= preview_index < len(self.preview_modes):
            active_align_marker = self._normalize_marker_key(self.preview_modes[preview_index][0])
        if active_align_marker is not None and active_align_marker in marker_keys:
            if hasattr(fig, '_preview_reference_times'):
                fig._preview_reference_times = None
            if hasattr(fig, '_preview_reference_tmarker'):
                fig._preview_reference_tmarker = None
            if preview_state is not None:
                preview_state['reference_times'] = None
        self._update_preview_selection_by_wave_names(fig, target_wave_names)
        self._apply_preview_selection(fig)
        self._refresh_preview_figure(fig, preview_index)
        self._refresh_pick_window_if_available()
        self._refresh_compare_for_preview_index(preview_index)
        return cleared_count, marker_keys, None

    def _sync_preview_amplitude_control(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        amp_box = controls.get('amplitude')
        if amp_box is None:
            return
        amp_box.set_val(f'{self.preview_amplitude_scale:g}')
        amp_preset = controls.get('amplitude_preset_widget')
        if amp_preset is not None:
            target_text = f'{self.preview_amplitude_scale:g}'
            index = amp_preset.findText(target_text)
            previous = amp_preset.blockSignals(True)
            try:
                amp_preset.setCurrentIndex(index if index >= 0 else 0)
            finally:
                amp_preset.blockSignals(previous)

    def _sync_preview_peak_half_window_control(self, fig):
        controls = getattr(fig, '_preview_controls', {})
        half_window_box = controls.get('peak_half_window')
        if half_window_box is None:
            return
        half_window_box.set_val(f'{self.preview_peak_half_window_default:g}')

    def _preview_peak_pick_mode_label(self):
        return 'Pk:Near' if getattr(self, 'preview_peak_pick_mode', 'pk') == 'pkm' else 'Pk:Std'

    def _toggle_preview_peak_pick_mode(self, fig):
        self.preview_peak_pick_mode = 'pkm' if self.preview_peak_pick_mode == 'pk' else 'pk'
        controls = getattr(fig, '_preview_controls', {})
        mode_button = controls.get('peak_pick_mode_button')
        if mode_button is not None:
            mode_button.setText(self._preview_peak_pick_mode_label())
        self._set_preview_search_status(
            fig,
            f'pk mode: {self.preview_peak_pick_mode}',
            color='#1f4e79'
        )

    def _set_preview_peak_half_window_default(self, fig, value_text):
        try:
            half_window = float(str(value_text or '').strip())
        except ValueError:
            self._set_preview_search_status(fig, 'pk half-window must be numeric', color='#8b0000')
            return False
        if half_window <= 0:
            self._set_preview_search_status(fig, 'pk half-window must be > 0', color='#8b0000')
            return False
        self.preview_peak_half_window_default = float(half_window)
        self._sync_preview_peak_half_window_control(fig)
        self._set_preview_search_status(
            fig,
            f'pk default half-window = {self.preview_peak_half_window_default:g}s',
            color='#1f4e79'
        )
        return True

    def _default_preview_amplitude_presets(self):
        return [0.2, 0.5, 1.0, 1.5, 2.0]

    def _normalize_preview_amplitude_preset(self, value):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        numeric = min(self.preview_amplitude_max, max(self.preview_amplitude_min, numeric))
        return round(numeric, 3)

    def _load_preview_amplitude_presets(self):
        if not os.path.exists(PREVIEW_AMPLITUDE_PRESET_PATH):
            return self._default_preview_amplitude_presets()
        try:
            with open(PREVIEW_AMPLITUDE_PRESET_PATH, 'r', encoding='utf-8') as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return self._default_preview_amplitude_presets()
        if not isinstance(loaded, list):
            return self._default_preview_amplitude_presets()
        presets = []
        seen = set()
        for item in loaded:
            normalized = self._normalize_preview_amplitude_preset(item)
            if normalized is None or normalized in seen:
                continue
            presets.append(normalized)
            seen.add(normalized)
        return presets or self._default_preview_amplitude_presets()

    def _save_preview_amplitude_presets(self):
        try:
            with open(PREVIEW_AMPLITUDE_PRESET_PATH, 'w', encoding='utf-8') as handle:
                json.dump(self.preview_amplitude_presets, handle, ensure_ascii=True, indent=2)
        except OSError:
            pass

    def _refresh_preview_amplitude_preset_combo(self, fig, select_value=None):
        controls = getattr(fig, '_preview_controls', {})
        combo = controls.get('amplitude_preset_widget')
        if combo is None:
            return
        previous = combo.blockSignals(True)
        try:
            combo.clear()
            combo.addItem('AmpP', None)
            selected_index = 0
            for index, preset_value in enumerate(self.preview_amplitude_presets, start=1):
                label = f'{preset_value:g}'
                combo.addItem(label, preset_value)
                if select_value is not None and abs(float(preset_value) - float(select_value)) < 1e-9:
                    selected_index = index
            combo.setCurrentIndex(selected_index)
        finally:
            combo.blockSignals(previous)

    def _save_preview_snapshot(self, fig, preview_index):
        if preview_index >= len(self.preview_modes):
            return
        tmarker, _, _ = self.preview_modes[preview_index]
        output_dir = self._lowq_preview_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = obspy.UTCDateTime().strftime("%Y%m%d_%H%M%S")
        latest_path = os.path.join(output_dir, f"lowq_preview_t{tmarker}_latest.png")
        history_path = os.path.join(output_dir, f"lowq_preview_t{tmarker}_{timestamp}.png")
        summary_artist = self._add_snapshot_station_summary(fig)
        fig.savefig(latest_path, dpi=300, bbox_inches='tight')
        fig.savefig(history_path, dpi=300, bbox_inches='tight')
        if summary_artist is not None:
            summary_artist.remove()
        print(f"Saved preview snapshot: {latest_path}")
        print(f"Saved preview snapshot history: {history_path}")

    def _collect_preview_wave_names_and_times(self, tmarker, reference_times=None):
        wave_names = []
        t_lst = []
        active_reference_times = {}
        for wave_name in self.markers[tmarker].keys():
            click_time = self._preview_alignment_reference_time(
                tmarker,
                wave_name,
                reference_times=reference_times,
            )
            if math.isnan(click_time):
                continue
            if self._is_preview_hidden_wave(wave_name):
                continue
            wave_names.append(wave_name)
            t_lst.append(click_time)
            active_reference_times[wave_name] = float(click_time)
        return wave_names, np.array(t_lst), active_reference_times

    def _build_standard_preview_evtdata(self, tmarker, x1, x2, order='gcarc', reference_times=None):
        if getattr(self, 'stack_mode', False):
            stack_wave_name = getattr(self.plotfig, '_stack_preview_wave_name', None) if self.plotfig is not None else None
            waves, t_lst, _active_reference_times = self._collect_stack_preview_stream(
                tmarker,
                stack_wave_name=stack_wave_name,
            )
            if len(waves) == 0:
                return None
            return EvtData(
                waves,
                t_lst,
                x1=x1,
                x2=x2,
                dt=self.dt,
                order=order,
                event_name_override=self._semantic_event_name(),
            )
        wave_names, t_lst, _active_reference_times = self._collect_preview_wave_names_and_times(
            tmarker,
            reference_times=reference_times,
        )
        if len(wave_names) == 0:
            return None
        waves = self._build_preview_stream_for_profiles(
            wave_names,
            profile=self._current_bandpass_profile(),
        )
        if len(waves) == 0:
            return None
        return EvtData(
            waves,
            t_lst,
            x1=x1,
            x2=x2,
            dt=self.dt,
            order=order,
            event_name_override=self._semantic_event_name(),
        )

    def _draw_standard_phase_annotations(self, ax, evtdata, axis_values, align_marker_key, phase_keys, reference_times=None, wave_lines=None):
        if not phase_keys:
            return 0
        axis_values = np.asarray(axis_values, dtype=float)
        if axis_values.size == 0:
            return 0
        visible_phase_count = 0
        label_specs = []
        xmin = evtdata.x1
        xmax = evtdata.x2
        sorted_unique = np.unique(np.sort(axis_values))
        if sorted_unique.size >= 2:
            spacing = np.min(np.diff(sorted_unique))
            marker_half_height = max(0.22, 0.28 * spacing)
        else:
            ymin, ymax = ax.get_ylim()
            marker_half_height = max(0.35, 0.015 * (ymax - ymin))
        selected_display_labels = [self.phase_display_labels.get(key, f"t{key}") for key in phase_keys]
        duplicate_labels = {label for label, count in Counter(selected_display_labels).items() if count > 1}
        ymin, ymax = ax.get_ylim()

        for marker_key in phase_keys:
            _marker_name, color = self.marker_styles[marker_key]
            phase_times = []
            phase_y = []
            for wave_index, tr in enumerate(evtdata.wave_ori):
                wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
                relative_time = self._preview_relative_phase_time(
                    align_marker_key,
                    marker_key,
                    wave_name,
                    reference_times=reference_times,
                    trace=tr,
                )
                if math.isnan(relative_time):
                    continue
                if relative_time < xmin or relative_time > xmax:
                    continue
                if wave_lines is not None and wave_index < len(wave_lines):
                    line_x = np.asarray(wave_lines[wave_index].get_xdata(), dtype=float)
                    line_y = np.asarray(wave_lines[wave_index].get_ydata(), dtype=float)
                    curve_y = float(np.interp(relative_time, line_x, line_y))
                else:
                    wave_data = np.asarray(evtdata.data[wave_index], dtype=float)
                    max_amp = np.max(np.abs(wave_data))
                    if max_amp == 0:
                        normalized = np.zeros_like(wave_data)
                    else:
                        normalized = wave_data / max_amp
                    wave_curve = normalized * self.preview_amplitude_scale + axis_values[wave_index]
                    curve_y = float(np.interp(relative_time, evtdata.time_axis, wave_curve))
                phase_times.append(relative_time)
                phase_y.append(curve_y)
            if not phase_times:
                continue
            visible_phase_count += 1
            phase_times = np.asarray(phase_times, dtype=float)
            phase_y = np.asarray(phase_y, dtype=float)
            marker_area = max(12.0, 22.0 * marker_half_height)
            ax.scatter(
                phase_times,
                phase_y,
                s=marker_area,
                c=color,
                edgecolors='white',
                linewidths=0.35,
                alpha=1.0,
                zorder=5,
            )
            top_index = int(np.argmax(phase_y))
            label_text = self._standard_phase_label(marker_key, duplicate_labels=duplicate_labels)
            label_specs.append({
                'text': label_text,
                'x': float(phase_times[top_index]),
                'y': float(phase_y[top_index]),
                'color': color,
            })

        label_placements = self._standard_phase_label_placements(ax, label_specs, xmin, xmax)
        for placement in label_placements:
            ax.annotate(
                placement['text'],
                xy=(placement['x'], placement['y']),
                xytext=(placement['label_x'], placement['label_y']),
                textcoords='data',
                color=placement['color'],
                fontsize=9,
                ha='center',
                va='bottom',
                clip_on=True,
                annotation_clip=True,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.65),
                zorder=4,
            )
        return visible_phase_count

    def _standard_phase_label_offset_to_data(self, ax, dx_points=0.0, dy_points=0.0):
        base_data = (0.0, 0.0)
        base_display = ax.transData.transform(base_data)
        point_to_pixel = ax.figure.dpi / 72.0
        shifted_display = (
            base_display[0] + float(dx_points) * point_to_pixel,
            base_display[1] + float(dy_points) * point_to_pixel,
        )
        shifted_data = ax.transData.inverted().transform(shifted_display)
        return float(shifted_data[0] - base_data[0]), float(shifted_data[1] - base_data[1])

    def _standard_phase_label_placements(self, ax, label_specs, xmin, xmax):
        if not label_specs:
            return []
        x_span = max(float(xmax) - float(xmin), 1.0)
        close_threshold = max(4.0, 0.055 * x_span)
        max_cluster_width = close_threshold * 1.25
        sorted_specs = sorted(label_specs, key=lambda item: item['x'])
        clusters = []
        current_cluster = []
        cluster_left = None
        cluster_right = None
        for spec in sorted_specs:
            spec_x = float(spec['x'])
            too_far_from_previous = (
                current_cluster
                and spec_x - float(cluster_right) >= close_threshold
            )
            too_wide_for_cluster = (
                current_cluster
                and spec_x - float(cluster_left) > max_cluster_width
            )
            if too_far_from_previous or too_wide_for_cluster:
                clusters.append(current_cluster)
                current_cluster = []
                cluster_left = None
                cluster_right = None
            current_cluster.append(spec)
            cluster_left = spec_x if cluster_left is None else min(cluster_left, spec_x)
            cluster_right = spec_x if cluster_right is None else max(cluster_right, spec_x)
        if current_cluster:
            clusters.append(current_cluster)

        x_offsets = (0, -12, 12, -22, 22, -34, 34, -46, 46)
        placements = []
        for cluster in clusters:
            cluster_top_y = max(float(spec['y']) for spec in cluster)
            multi_label_cluster = len(cluster) > 1
            for lane, spec in enumerate(cluster):
                x_offset = x_offsets[min(lane, len(x_offsets) - 1)]
                y_offset = 7 + min(lane, 8) * 10
                dx_data, dy_data = self._standard_phase_label_offset_to_data(ax, x_offset, y_offset)
                label_x = min(max(float(spec['x']) + dx_data, float(xmin)), float(xmax))
                label_base_y = cluster_top_y if multi_label_cluster else float(spec['y'])
                placement = dict(spec)
                placement.update({
                    'label_x': label_x,
                    'label_y': label_base_y + dy_data,
                    'lane': lane,
                })
                placements.append(placement)
        return placements

    def _draw_preview_phase_annotations(self, ax, evtdata, axis_values, align_marker_key, phase_keys, reference_times=None):
        if not phase_keys:
            return 0
        axis_values = np.asarray(axis_values, dtype=float)
        if axis_values.size == 0:
            return 0
        xmin = evtdata.x1
        xmax = evtdata.x2
        sorted_unique = np.unique(np.sort(axis_values))
        if sorted_unique.size >= 2:
            spacing = np.min(np.diff(sorted_unique))
            marker_half_height = max(0.18, 0.22 * spacing)
        else:
            marker_half_height = 0.35
        visible_phase_count = 0

        for marker_key in phase_keys:
            _marker_name, color = self.marker_styles[marker_key]
            phase_times = []
            phase_y = []
            for wave_index, tr in enumerate(evtdata.wave_ori):
                wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
                relative_time = self._preview_relative_phase_time(
                    align_marker_key,
                    marker_key,
                    wave_name,
                    reference_times=reference_times,
                    trace=tr,
                )
                if math.isnan(relative_time):
                    continue
                if relative_time < xmin or relative_time > xmax:
                    continue
                phase_times.append(relative_time)
                phase_y.append(axis_values[wave_index])
            if not phase_times:
                continue
            visible_phase_count += 1
            ax.vlines(
                np.asarray(phase_times, dtype=float),
                np.asarray(phase_y, dtype=float) - marker_half_height,
                np.asarray(phase_y, dtype=float) + marker_half_height,
                colors=color,
                linewidth=0.55,
                alpha=0.95,
                zorder=3,
            )
        return visible_phase_count

    def _save_standard_preview_plot(self, evtdata, tmarker, axis_mode, output_dir, timestamp_tag, phase_keys, export_options=None):
        if evtdata is None or evtdata.sta_num == 0:
            return None
        fig, ax = init_standard_wave_figure()
        if axis_mode == 'az':
            order = 'az'
            filename_tag = 'az'
        else:
            order = 'gcarc'
            filename_tag = 'gcarc'
        axis_values, y_ticks, y_ticklabels, ylabel = self._preview_y_axis_config(evtdata, order=order)
        colors = []
        linewidths = []
        for tr in evtdata.wave_ori:
            wave_name = getattr(tr.stats, 'dephasekit_wave_name', '')
            color, linewidth = self._preview_standard_wave_style(tr, wave_name)
            colors.append(color)
            linewidths.append(linewidth)
        wave_lines = plot_standard_waves(ax, evtdata, axis_values, colors, enf=self.preview_amplitude_scale, linewidths=linewidths)
        set_standard_wave_axis(
            ax,
            evtdata,
            axis_values,
            xlabel=f"Time after {self._phase_display_label(tmarker)} (s)",
            ylabel=ylabel,
            y_mode=axis_mode,
            y_ticks=y_ticks,
            y_ticklabels=y_ticklabels,
        )
        visible_phase_count = self._draw_standard_phase_annotations(
            ax,
            evtdata,
            axis_values,
            tmarker,
            phase_keys,
            reference_times=self._preview_reference_times_from_evtdata(evtdata),
            wave_lines=wave_lines,
        )
        header_lines = self._standard_export_header_lines(evtdata, export_options)
        if header_lines:
            fig.subplots_adjust(top=0.93)
            axes_box = ax.get_position()
            title_x = axes_box.x0 + axes_box.width * 0.5
            fig.text(
                title_x,
                0.975,
                '\n'.join(header_lines),
                fontsize=14,
                ha='center',
                va='top',
                linespacing=1.18,
            )
        else:
            fig.subplots_adjust(top=0.965)
        output_path = os.path.join(output_dir, f"{filename_tag}_{timestamp_tag}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return output_path, visible_phase_count

    def _save_standard_pierce_plot(self, evtdata, output_dir, timestamp_tag, export_options=None):
        if evtdata is None or evtdata.sta_num == 0:
            return None
        pierce_records = self._standard_export_pierce_records_from_evtdata(evtdata)
        if not pierce_records:
            return None, 0
        os.makedirs(output_dir, exist_ok=True)
        signature = self._standard_pierce_cache_signature(evtdata, pierce_records, export_options=export_options)
        cached_output_path, cached_zoom_path = self._try_reuse_standard_pierce_cache(output_dir, signature)
        if cached_output_path is not None:
            output_path, zoom_output_path = self._copy_standard_pierce_cache_to_timestamped_outputs(
                cached_output_path,
                cached_zoom_path,
                output_dir,
                timestamp_tag,
            )
            return output_path, len(pierce_records), zoom_output_path
        script_path = self._standard_pierce_plot_script_path()
        cpt_path = self._standard_pierce_cpt_path()
        try:
            output_path = None
            if os.path.isfile(script_path) and os.path.isfile(cpt_path):
                output_prefix = os.path.join(output_dir, f"pierce_{timestamp_tag}")
                title_lines = self._standard_export_header_lines(evtdata, export_options)
                with tempfile.NamedTemporaryFile('w', suffix='_std_pierce_xy.txt', delete=False, encoding='utf-8') as handle:
                    temp_input_path = handle.name
                    for record in pierce_records:
                        point_color = self._standard_pierce_record_color(record.wave_name)
                        is_flip = 1 if self._is_user4_wave(record.wave_name) else 0
                        handle.write(
                            f"{float(record.longitude):.6f} {float(record.latitude):.6f} {point_color} {is_flip}\n"
                        )
                try:
                    env = os.environ.copy()
                    env['INPUT_FILE'] = temp_input_path
                    env['OUTPUT_PREFIX'] = output_prefix
                    env['CUSTOM_CPT'] = cpt_path
                    env['EVENT_LON'] = f"{float(evtdata.evlo):.6f}"
                    env['EVENT_LAT'] = f"{float(evtdata.evla):.6f}"
                    result = subprocess.run(
                        ['bash', script_path],
                        check=True,
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=os.path.dirname(script_path),
                        timeout=120,
                    )
                    output_path = f"{output_prefix}.png"
                    if result.stdout.strip():
                        last_line = result.stdout.strip().splitlines()[-1].strip()
                        if last_line.endswith('.png'):
                            output_path = last_line
                    self._apply_standard_header_to_image(output_path, title_lines)
                finally:
                    try:
                        os.remove(temp_input_path)
                    except OSError:
                        pass
            if output_path is None:
                output_path = self._save_standard_pierce_main_plot(
                    evtdata,
                    pierce_records,
                    output_dir,
                    timestamp_tag,
                    export_options=export_options,
                )
            zoom_output_path = self._save_standard_pierce_zoom_plot(
                evtdata,
                pierce_records,
                output_dir,
                timestamp_tag,
                export_options=export_options,
            )
            self._store_standard_pierce_cache(output_path, zoom_output_path, output_dir, signature)
            return output_path, len(pierce_records), zoom_output_path
        except subprocess.TimeoutExpired:
            print('Standard pierce plot timed out after 120s; falling back to Python renderer')
            output_path = self._save_standard_pierce_main_plot(
                evtdata,
                pierce_records,
                output_dir,
                timestamp_tag,
                export_options=export_options,
            )
            zoom_output_path = self._save_standard_pierce_zoom_plot(
                evtdata,
                pierce_records,
                output_dir,
                timestamp_tag,
                export_options=export_options,
            )
            self._store_standard_pierce_cache(output_path, zoom_output_path, output_dir, signature)
            return output_path, len(pierce_records), zoom_output_path
        except Exception as exc:
            print(f'Standard pierce plot generation failed: {exc}')
            return None, 0, None

    def _save_standard_preview_exports(self, fig, preview_index):
        if preview_index >= len(self.preview_modes):
            return
        parent_window = getattr(getattr(fig.canvas, 'manager', None), 'window', None)
        export_options = self._prompt_standard_export_options(parent_window=parent_window)
        if export_options is None:
            self._set_preview_search_status(fig, 'Std export cancelled', color='#8b0000')
            return
        phase_keys = self._standard_export_selected_phase_keys(export_options)
        axis_options = self._standard_export_selected_axes(export_options)
        tmarker, x1, x2 = self.preview_modes[preview_index]
        standard_phase_keys = list(phase_keys)
        reference_times = self._preview_reference_times_from_figure(fig, expected_tmarker=tmarker)
        if not any(axis_options.values()):
            self._set_preview_search_status(fig, 'Select at least one Std panel', color='#8b0000')
            return
        evtdata_gcarc = None
        evtdata_az = None
        if axis_options['gcarc'] or axis_options['pierce'] or axis_options['pierce_group']:
            evtdata_gcarc = self._build_standard_preview_evtdata(
                tmarker,
                x1,
                x2,
                order='gcarc',
                reference_times=reference_times,
            )
        if axis_options['az']:
            evtdata_az = self._build_standard_preview_evtdata(
                tmarker,
                x1,
                x2,
                order='az',
                reference_times=reference_times,
            )
        if ((axis_options['gcarc'] or axis_options['pierce'] or axis_options['pierce_group']) and evtdata_gcarc is None) or (axis_options['az'] and evtdata_az is None):
            self._set_preview_search_status(fig, 'No preview waveforms to export', color='#8b0000')
            return
        output_dir = self._analysis_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        legend_phase_keys = []
        if evtdata_gcarc is not None:
            legend_phase_keys = self._visible_phase_keys_in_evtdata(
                evtdata_gcarc,
                tmarker,
                standard_phase_keys,
                reference_times=self._preview_reference_times_from_evtdata(evtdata_gcarc),
            )
        elif evtdata_az is not None:
            legend_phase_keys = self._visible_phase_keys_in_evtdata(
                evtdata_az,
                tmarker,
                standard_phase_keys,
                reference_times=self._preview_reference_times_from_evtdata(evtdata_az),
            )
        if not legend_phase_keys:
            legend_phase_keys = list(standard_phase_keys)
        saved_parts = []
        if axis_options['gcarc']:
            gcarc_path, gcarc_phase_count = self._save_standard_preview_plot(
                evtdata_gcarc, tmarker, 'gcarc', output_dir, timestamp_tag, standard_phase_keys, export_options=export_options
            )
            print(f"Saved standard distance plot: {gcarc_path}")
            saved_parts.append(f"Dist:{gcarc_phase_count}")
        if axis_options['az']:
            az_path, az_phase_count = self._save_standard_preview_plot(
                evtdata_az, tmarker, 'az', output_dir, timestamp_tag, standard_phase_keys, export_options=export_options
            )
            print(f"Saved standard azimuth plot: {az_path}")
            saved_parts.append(f"Az:{az_phase_count}")
        if axis_options['pierce']:
            pierce_path, pierce_count, pierce_zoom_path = self._save_standard_pierce_plot(
                evtdata_gcarc, output_dir, timestamp_tag, export_options=export_options
            )
            if pierce_path is not None:
                print(f"Saved standard pierce plot: {pierce_path}")
                if pierce_zoom_path is not None:
                    print(f"Saved standard pierce zoom plot: {pierce_zoom_path}")
                saved_parts.append(f"Pierce:{pierce_count}")
            else:
                saved_parts.append("Pierce:0")
        if axis_options['pierce_group']:
            group_pierce_records = self._standard_export_pierce_records_from_evtdata(evtdata_gcarc)
            group_path = self._save_standard_pierce_group_plot(
                evtdata_gcarc,
                group_pierce_records,
                output_dir,
                timestamp_tag,
                export_options=export_options,
            )
            if group_path is not None:
                print(f"Saved standard pierce-by-group plot: {group_path}")
                saved_parts.append(f"PierceGrp:{len(group_pierce_records)}")
            else:
                saved_parts.append("PierceGrp:0")
        if self._std_export_option_value(export_options, 'phase_legend'):
            legend_path = self._save_standard_phase_legend_image(
                legend_phase_keys,
                output_dir,
                timestamp_tag,
            )
            if legend_path is not None:
                print(f"Saved standard phase legend: {legend_path}")
                saved_parts.append("Legend:1")
        print(f"Standard plot phases: {','.join(f't{key}' for key in standard_phase_keys)}")
        self._set_preview_search_status(
            fig,
            f"Saved Std {' | '.join(saved_parts)}; Amp x{self.preview_amplitude_scale:.1f}",
            color='#1f4e79'
        )

    def _build_preview_stream_for_profiles(self, wave_names, profile=None):
        waves = obspy.Stream()
        target_fs = 1.0 / self.dt
        for wave_name in wave_names:
            tr = self._trace_from_runtime_dir(wave_name)
            if abs(tr.stats.sampling_rate - target_fs) > 1e-3:
                tr.resample(target_fs, window="hann")
            if profile is not None:
                self._apply_bandpass_to_trace(tr, profile)
            tr.data = np.asarray(tr.data, dtype=float) * self._wave_polarity_factor(wave_name)
            waves += tr
        return waves

    def _current_preview_metadata_state(self):
        if self.plotfig is None or not hasattr(self.plotfig, '_preview_state'):
            return {}, None, None
        preview_state = self.plotfig._preview_state
        if preview_state is None:
            return {}, None, None
        selected_wave_names = {
            preview_state['metadata'][idx]['wave_name']
            for idx in preview_state.get('selected_indices', set())
            if idx < len(preview_state['metadata'])
        }
        active_wave_name = None
        anchor_wave_name = None
        active_index = preview_state.get('active_index', 0)
        anchor_index = preview_state.get('anchor_index', active_index)
        if active_index < len(preview_state['metadata']):
            active_wave_name = preview_state['metadata'][active_index]['wave_name']
        if anchor_index < len(preview_state['metadata']):
            anchor_wave_name = preview_state['metadata'][anchor_index]['wave_name']
        return selected_wave_names, active_wave_name, anchor_wave_name

    def _compare_profiles_with_raw(self):
        profiles = [None] + list(self._ensure_compare_profiles())
        return profiles[:self.max_compare_columns]

    def _compare_selected_slot(self):
        profiles = self._compare_profiles_with_raw()
        default_slot = 1 if len(profiles) > 1 else 0
        if self.comparefig is None or not hasattr(self.comparefig, '_compare_state'):
            return default_slot
        selected_slot = self.comparefig._compare_state.get('selected_slot', default_slot)
        return max(0, min(selected_slot, len(profiles) - 1))

    def _compare_input_profile(self, fig):
        if fig is None or not hasattr(fig, '_compare_controls'):
            return self._current_bandpass_profile()
        controls = fig._compare_controls
        try:
            freqmin = float(controls['freqmin'].text.strip())
            freqmax = float(controls['freqmax'].text.strip())
            corners = int(float(controls['corners'].text.strip()))
            passes = int(float(controls['passes'].text.strip()))
        except (ValueError, AttributeError):
            return None
        if freqmin <= 0 or freqmax <= freqmin:
            return None
        return {
            'freqmin': freqmin,
            'freqmax': freqmax,
            'corners': max(1, corners),
            'passes': max(1, passes),
        }

    def _populate_compare_inputs(self, fig, profile):
        if fig is None or not hasattr(fig, '_compare_controls') or profile is None:
            return
        fig._compare_controls['freqmin'].set_val(f"{profile['freqmin']:g}")
        fig._compare_controls['freqmax'].set_val(f"{profile['freqmax']:g}")
        fig._compare_controls['corners'].set_val(str(int(profile['corners'])))
        fig._compare_controls['passes'].set_val(str(int(profile['passes'])))

    def _replace_selected_compare_profile(self, fig, preview_index):
        profile = self._compare_input_profile(fig)
        if profile is None:
            print("Compare BP parameters are invalid; nothing changed.")
            return
        profiles = list(self._ensure_compare_profiles())
        selected_slot = self._compare_selected_slot()
        if selected_slot == 0:
            if not profiles:
                profiles.append(profile)
                selected_slot = 1
            else:
                profiles[0] = profile
                selected_slot = 1
        else:
            profile_index = selected_slot - 1
            if profile_index >= len(profiles):
                profiles.append(profile)
                selected_slot = len(profiles)
            else:
                profiles[profile_index] = profile
        self.compare_bandpass_profiles = profiles[:max(0, self.max_compare_columns - 1)]
        self.plot_compare_preview(preview_index, selected_slot=selected_slot, input_profile=profile)

    def _add_compare_profile_from_inputs(self, fig, preview_index):
        profile = self._compare_input_profile(fig)
        if profile is None:
            print("Compare BP parameters are invalid; nothing added.")
            return
        profiles = list(self._ensure_compare_profiles())
        new_key = self._bandpass_profile_key(profile)
        for existing in profiles:
            if self._bandpass_profile_key(existing) == new_key:
                print(f"Compare BP already exists: {self._format_bandpass_label(profile)}")
                self.plot_compare_preview(preview_index, input_profile=profile)
                return
        if len(profiles) >= self.max_compare_columns - 1:
            selected_slot = max(1, self._compare_selected_slot())
            profiles[selected_slot - 1] = profile
        else:
            profiles.append(profile)
            selected_slot = len(profiles)
        self.compare_bandpass_profiles = profiles[:max(0, self.max_compare_columns - 1)]
        self.plot_compare_preview(preview_index, selected_slot=selected_slot, input_profile=profile)

    def _clear_compare_profiles_from_window(self, preview_index):
        self.compare_bandpass_profiles = self._default_compare_profiles()
        selected_slot = 1 if self.compare_bandpass_profiles else 0
        self.plot_compare_preview(preview_index, selected_slot=selected_slot)

    def _toggle_compare_preset_profile(self, preview_index, profile):
        normalized = self._normalize_bandpass_profile(profile)
        if normalized is None:
            self._emit_compare_status('Selected compare BP preset is invalid')
            return
        key = self._bandpass_profile_key(normalized)
        profiles = list(self.compare_bandpass_profiles)
        existing_index = next(
            (idx for idx, existing in enumerate(profiles) if self._bandpass_profile_key(existing) == key),
            None,
        )
        if existing_index is not None:
            profiles.pop(existing_index)
            self.compare_bandpass_profiles = profiles
            selected_slot = min(existing_index + 1, len(profiles))
            self.plot_compare_preview(preview_index, selected_slot=selected_slot)
            return
        if len(profiles) >= self.max_compare_columns - 1:
            self._emit_compare_status(
                f'Compare preset limit reached: {self.max_compare_columns - 1} filtered columns (+ Raw)'
            )
            return
        profiles.append(normalized)
        self.compare_bandpass_profiles = profiles
        self.plot_compare_preview(preview_index, selected_slot=len(profiles))

    def _save_compare_defaults_from_selection(self, preview_index):
        preset_keys = {
            self._bandpass_profile_key(profile): profile
            for profile in self.compare_preset_profiles
        }
        default_profiles = []
        seen = set()
        for profile in self.compare_bandpass_profiles:
            key = self._bandpass_profile_key(profile)
            if key not in preset_keys or key in seen:
                continue
            default_profiles.append(preset_keys[key])
            seen.add(key)
        self.compare_default_bandpass_profiles = default_profiles[:max(0, self.max_compare_columns - 1)]
        if self.compare_defaults_update_callback is not None:
            self.compare_defaults_update_callback(self.compare_default_bandpass_profiles)
        else:
            self._emit_compare_status('Compare default presets updated')
        self.plot_compare_preview(preview_index)

    def _save_compare_snapshot(self, fig, preview_index):
        if preview_index >= len(self.preview_modes):
            return
        tmarker, _, _ = self.preview_modes[preview_index]
        output_dir = self._analysis_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = obspy.UTCDateTime().strftime("%Y%m%d_%H%M%S")
        latest_path = os.path.join(output_dir, f"compare_preview_t{tmarker}_latest.png")
        history_path = os.path.join(output_dir, f"compare_preview_t{tmarker}_{timestamp}.png")
        fig.savefig(latest_path, dpi=300, bbox_inches='tight')
        fig.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"Saved compare snapshot: {latest_path}")
        print(f"Saved compare snapshot history: {history_path}")

    def _capture_compare_window_state(self):
        if self.comparefig is None:
            return None
        manager = getattr(self.comparefig.canvas, 'manager', None)
        window = getattr(manager, 'window', None)
        if window is None:
            return None
        state = {'maximized': False, 'geometry': None}
        try:
            state['maximized'] = bool(window.isMaximized())
        except Exception:
            state['maximized'] = False
        try:
            geometry = window.geometry()
            state['geometry'] = (
                int(geometry.x()),
                int(geometry.y()),
                int(geometry.width()),
                int(geometry.height()),
            )
        except Exception:
            state['geometry'] = None
        return state

    def _restore_compare_window_state(self, fig, state):
        manager = getattr(fig.canvas, 'manager', None)
        window = getattr(manager, 'window', None)
        if window is None:
            return
        if state is None:
            # Newly created compare window: center it on the workarea. WSLg/XWayland
            # ignores move() called before show/map, so retry after show + async.
            def _center_compare():
                center_widget_keep_size(window)
            try:
                _center_compare()
                QTimer.singleShot(0, _center_compare)
            except Exception:
                pass
            return
        geometry = state.get('geometry')
        if geometry is not None:
            try:
                window.setGeometry(*geometry)
            except Exception:
                pass
        if state.get('maximized') and not os.environ.get('WAYLAND_DISPLAY'):
            try:
                window.showMaximized()
            except Exception:
                pass

    def _maximize_preview_window(self, fig):
        manager = getattr(fig.canvas, 'manager', None)
        window = getattr(manager, 'window', None)
        if window is None:
            # 当 window 为 None 时（从厚度审阅窗打开 stack 子窗口的场景），
            # plt.ioff() 创建的 figure 没有 Qt window。用 plt.show() 强制显示。
            try:
                plt.show(block=False)
            except Exception:
                pass
            return
        try:
            workarea = screen_workarea_rect(widget=window, screen=window.screen())
            if workarea is None or not workarea.isValid():
                # workarea probe failed (e.g. WSLg): fall back to a centered
                # near-full-size window on the detected screen so the preview
                # still pops up centered instead of at the default top-left.
                center_widget_on_workarea(window, frac=0.92)
                window.show()
                return

            size_hint = self._preview_window_size_hint()
            target_frame_width = int(workarea.width())
            target_frame_height = int(workarea.height())
            if size_hint is not None:
                width, height = size_hint
                size_scale = 0.985
                target_frame_width = min(int(width * size_scale), target_frame_width)
                target_frame_height = min(int(height * size_scale), target_frame_height)

            frame = None
            frame_dx = 0
            frame_dy = 0
            try:
                handle = window.windowHandle()
                if handle is not None:
                    frame = handle.frameGeometry()
                    if frame is not None and frame.isValid():
                        frame_dx = max(0, int(frame.width()) - int(window.width()))
                        frame_dy = max(0, int(frame.height()) - int(window.height()))
            except Exception:
                frame_dx = 0
                frame_dy = 0

            client_width = max(200, target_frame_width - frame_dx)
            client_height = max(200, target_frame_height - frame_dy)
            target_x = int(workarea.x() + max(0, (workarea.width() - target_frame_width) / 2))
            target_y = int(workarea.y() + max(0, (workarea.height() - target_frame_height) / 2))

            def _center_after_show():
                # WSLg/XWayland 的窗口管理器在窗口 map 时会重置位置，show 之前
                # 调用的 move() 常被忽略，导致预览窗弹出在默认位置(左上)而非居中。
                # show 之后再 move 一次(同步+异步各一次)才能真正落位。
                try:
                    window.move(target_x, target_y)
                except Exception:
                    pass

            try:
                # WSLg/XWayland: 先 resize 好尺寸再 show，避免 hide/resize/show
                # 多次几何变化各合成一帧导致黑屏闪烁。位置在 show 后校正。
                window.resize(client_width, client_height)
                window.show()
                _center_after_show()
                # 异步兜底：WM 可能在 show 返回后才完成放置，延迟再校正一次。
                QTimer.singleShot(0, _center_after_show)
            except Exception:
                try:
                    window.show()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # 用同步 draw 一次性出图，取代 draw_idle 的异步多次重绘。
            fig.canvas.draw()
        except Exception:
            try:
                fig.canvas.draw_idle()
            except Exception:
                pass

    def close_compare_window(self):
        if self.comparefig is None or not plt.fignum_exists(self.comparefig.number):
            self.comparefig = None
            return False
        try:
            plt.close(self.comparefig)
        except Exception:
            return False
        self.comparefig = None
        return True

    def close_preview_window(self):
        if self.plotfig is None or not plt.fignum_exists(self.plotfig.number):
            self.plotfig = None
            self._close_preview_control_dock()
            return False
        self._close_preview_control_dock()
        try:
            plt.close(self.plotfig)
        except Exception:
            return False
        self.plotfig = None
        return True

    def _add_current_bp_and_refresh_compare(self, preview_index):
        added = self.add_current_bandpass_to_compare()
        if added and self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            self.plot_compare_preview(preview_index)

    def _clear_compare_and_refresh(self):
        self.clear_compare_bandpasses()
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            plt.close(self.comparefig)
        self.comparefig = None

    def plot_compare_preview(self, preview_index=0, selected_slot=None, input_profile=None, use_default_profiles=False):
        if preview_index >= len(self.preview_modes):
            return
        tmarker, x1, x2 = self.preview_modes[preview_index]
        reference_times = None
        if self.comparefig is not None and hasattr(self.comparefig, '_compare_state'):
            compare_state = self.comparefig._compare_state
            if (compare_state is not None
                    and compare_state.get('preview_index') == preview_index
                    and compare_state.get('tmarker') == self._normalize_marker_key(tmarker)):
                reference_times = compare_state.get('reference_times')
        if reference_times is None and self.plotfig is not None:
            reference_times = self._preview_reference_times_from_figure(self.plotfig, expected_tmarker=tmarker)
        wave_names, t_lst, active_reference_times = self._collect_preview_wave_names_and_times(
            tmarker,
            reference_times=reference_times,
        )
        if len(wave_names) == 0:
            return
        phase_keys, _error_message = self._parse_standard_phase_tokens(self.standard_export_phase_tokens)
        if use_default_profiles:
            self.compare_bandpass_profiles = self._default_compare_profiles()
        self._ensure_compare_profiles()
        profiles = self._compare_profiles_with_raw()
        window_state = self._capture_compare_window_state()
        if self.comparefig is not None and plt.fignum_exists(self.comparefig.number):
            plt.close(self.comparefig)
        if selected_slot is None:
            selected_slot = 1 if len(profiles) > 1 else 0
        selected_slot = max(0, min(selected_slot, len(profiles) - 1))
        selected_wave_names, active_wave_name, _ = self._current_preview_metadata_state()
        preset_rows = max(1, int(math.ceil(max(1, len(self.compare_preset_profiles)) / 4)))
        bottom_margin = 0.18 + max(0, preset_rows - 1) * 0.06 + (0.06 if self.compare_preset_profiles else 0.0)
        comparefig = plt.figure(figsize=(4.6 * len(profiles), 10))
        _force_qt_arrow_cursor_for_figure(comparefig)
        comparefig.subplots_adjust(bottom=bottom_margin, wspace=0.22)
        gs = GridSpec(1, len(profiles), figure=comparefig)
        axes = [comparefig.add_subplot(gs[0, idx]) for idx in range(len(profiles))]
        compare_evtdata = None
        compare_lines = []
        compare_metadata = []
        for idx, profile in enumerate(profiles):
            waves = self._build_preview_stream_for_profiles(wave_names, profile=profile)
            evtdata = EvtData(
                waves,
                t_lst,
                x1=x1,
                x2=x2,
                dt=self.dt,
                event_name_override=self._semantic_event_name(),
            )
            if compare_evtdata is None:
                compare_evtdata = evtdata
                compare_metadata = [{
                    'name': f"{tr.stats.network}.{tr.stats.station}",
                    'gcarc': _sac_float(tr, 'gcarc', 0.0),
                    'baz': _sac_float(tr, 'baz', 0.0),
                    'wave_name': getattr(tr.stats, 'dephasekit_wave_name', ''),
                    'stack_summary': self._stack_wave_summary(getattr(tr.stats, 'dephasekit_wave_name', '')),
                    'is_marked_m': self._is_preview_purple_wave(getattr(tr.stats, 'dephasekit_wave_name', '')),
                    'is_user1_marked': self._is_user1_wave(getattr(tr.stats, 'dephasekit_wave_name', '')),
                    'is_user5_marked': self._is_user5_wave(getattr(tr.stats, 'dephasekit_wave_name', '')),
                    'is_user4_marked': self._is_user4_wave(getattr(tr.stats, 'dephasekit_wave_name', '')),
                } for tr in evtdata.wave_ori]
            y_values, y_ticks, y_ticklabels, ylabel = self._preview_y_axis_config(evtdata, order='gcarc')
            lines = plot_waves_only(axes[idx], evtdata, enf=1, y_values=y_values)
            set_wave_axis_only(
                axes[idx], evtdata, tmarker,
                title=self._format_bandpass_label(profile),
                show_ylabel=(idx == 0),
                y_values=y_values,
                y_ticks=y_ticks,
                y_ticklabels=y_ticklabels,
                ylabel=ylabel,
            )
            self._draw_preview_phase_annotations(
                axes[idx],
                evtdata,
                y_values,
                tmarker,
                phase_keys,
                reference_times=active_reference_times,
            )
            compare_lines.append(lines)
        comparefig.suptitle(
            "{}:{}\n Latitude: {:.2f}\N{DEGREE SIGN}, Longitude: {:.2f}\N{DEGREE SIGN}, Depth:{:.1f} km".format(
                _event_title_prefix(getattr(compare_evtdata, 'is_stack_mode', False)),
                compare_evtdata.evtname, compare_evtdata.evla, compare_evtdata.evlo, compare_evtdata.evdp),
            fontsize=15
        )
        for idx in range(1, len(profiles)):
            axes[idx].set_ylim(axes[0].get_ylim())
            axes[idx].set_yticks(axes[0].get_yticks())
        active_index = 0
        for idx, meta in enumerate(compare_metadata):
            if meta['wave_name'] == active_wave_name:
                active_index = idx
                break
        for idx, meta in enumerate(compare_metadata):
            is_selected = meta['wave_name'] in selected_wave_names
            color, width = self._preview_wave_colors(meta, is_selected)
            for lines in compare_lines:
                self._apply_preview_line_style(lines[idx], meta, is_selected, color, width)
        for idx, ax in enumerate(axes):
            if idx == selected_slot:
                for spine in ax.spines.values():
                    spine.set_color('#d62728')
                    spine.set_linewidth(1.6)
                ax.set_title(self._format_bandpass_label(profiles[idx]), fontsize=11, color='#d62728')
        comparefig._compare_state = {
            'preview_index': preview_index,
            'tmarker': tmarker,
            'selected_slot': selected_slot,
            'profiles': profiles,
            'preset_profiles': list(self.compare_preset_profiles),
            'reference_times': active_reference_times,
            'metadata': compare_metadata,
        }
        comparefig._compare_axes = axes
        comparefig._compare_controls = {}
        controls_y = 0.035
        preset_row_y0 = controls_y + 0.075
        compare_stack_text = self._compare_stack_summary_text(compare_metadata, active_wave_name)
        if compare_stack_text:
            comparefig.text(0.58, controls_y + 0.025, compare_stack_text, fontsize=10, va='center')
        comparefig.text(0.10, controls_y + 0.025, 'Compare BP', fontsize=10, va='center')
        comparefig.text(0.17, controls_y + 0.025, 'c', fontsize=10, va='center')
        comparefig.text(0.36, controls_y + 0.025, 'n', fontsize=10, va='center')
        comparefig.text(0.46, controls_y + 0.025, 'p', fontsize=10, va='center')
        if self.compare_preset_profiles:
            comparefig.text(0.10, preset_row_y0 + 0.018, 'Presets', fontsize=10, va='center')
        preset_buttons = []
        for preset_index, preset_profile in enumerate(self.compare_preset_profiles):
            row = preset_index // 4
            col = preset_index % 4
            ax_left = 0.18 + col * 0.18
            ax_bottom = preset_row_y0 + (preset_rows - 1 - row) * 0.06
            ax_preset = comparefig.add_axes([ax_left, ax_bottom, 0.14, 0.045])
            label = self._short_bandpass_label(preset_profile)
            button = Button(ax_preset, label)
            if any(self._bandpass_profile_key(existing) == self._bandpass_profile_key(preset_profile)
                   for existing in self.compare_bandpass_profiles):
                ax_preset.set_facecolor('#ffe7a8')
            elif any(self._bandpass_profile_key(existing) == self._bandpass_profile_key(preset_profile)
                     for existing in self.compare_default_bandpass_profiles):
                ax_preset.set_facecolor('#e6f1ff')
            else:
                ax_preset.set_facecolor('#f2f2f2')
            button.on_clicked(
                lambda _event, profile=preset_profile: self._toggle_compare_preset_profile(preview_index, profile)
            )
            preset_buttons.append(button)

        ax_freqmin = comparefig.add_axes([0.18, controls_y, 0.07, 0.05])
        ax_freqmax = comparefig.add_axes([0.27, controls_y, 0.07, 0.05])
        ax_corners = comparefig.add_axes([0.39, controls_y, 0.05, 0.05])
        ax_passes = comparefig.add_axes([0.48, controls_y, 0.05, 0.05])
        ax_apply = comparefig.add_axes([0.57, controls_y, 0.07, 0.05])
        ax_add = comparefig.add_axes([0.66, controls_y, 0.07, 0.05])
        ax_clear = comparefig.add_axes([0.75, controls_y, 0.07, 0.05])
        ax_capture = comparefig.add_axes([0.84, controls_y, 0.07, 0.05])
        ax_defaults = comparefig.add_axes([0.84, preset_row_y0, 0.07, 0.045])
        box_freqmin = TextBox(ax_freqmin, '', initial='')
        box_freqmax = TextBox(ax_freqmax, '', initial='')
        box_corners = TextBox(ax_corners, '', initial='')
        box_passes = TextBox(ax_passes, '', initial='')
        button_apply = Button(ax_apply, 'Apply')
        button_add = Button(ax_add, 'Add')
        button_clear = Button(ax_clear, 'Clear')
        button_capture = Button(ax_capture, 'C')
        button_defaults = Button(ax_defaults, 'Def')
        comparefig._compare_controls['freqmin'] = box_freqmin
        comparefig._compare_controls['freqmax'] = box_freqmax
        comparefig._compare_controls['corners'] = box_corners
        comparefig._compare_controls['passes'] = box_passes
        comparefig._compare_controls['apply'] = button_apply
        comparefig._compare_controls['add'] = button_add
        comparefig._compare_controls['clear'] = button_clear
        comparefig._compare_controls['capture'] = button_capture
        comparefig._compare_controls['defaults'] = button_defaults
        comparefig._compare_controls['preset_buttons'] = preset_buttons
        shown_profile = input_profile
        if shown_profile is None:
            shown_profile = profiles[selected_slot] if selected_slot > 0 else self._current_bandpass_profile()
        if shown_profile is not None:
            self._populate_compare_inputs(comparefig, shown_profile)

        def on_compare_click(event):
            if event.inaxes in axes:
                selected = axes.index(event.inaxes)
                profile = profiles[selected] if selected < len(profiles) else None
                self.plot_compare_preview(preview_index, selected_slot=selected, input_profile=profile)

        def on_compare_key(event):
            key = str(event.key).lower()
            if key == ' ' or key == 'space':
                self.close_compare_window()
            elif key == 'a':
                self._add_compare_profile_from_inputs(comparefig, preview_index)
            elif key == 'c':
                self._save_compare_snapshot(comparefig, preview_index)
            elif key == 'd':
                self._save_compare_defaults_from_selection(preview_index)
            elif key == 'x':
                self._clear_compare_profiles_from_window(preview_index)

        button_apply.on_clicked(lambda _event: self._replace_selected_compare_profile(comparefig, preview_index))
        button_add.on_clicked(lambda _event: self._add_compare_profile_from_inputs(comparefig, preview_index))
        button_clear.on_clicked(lambda _event: self._clear_compare_profiles_from_window(preview_index))
        button_capture.on_clicked(lambda _event: self._save_compare_snapshot(comparefig, preview_index))
        button_defaults.on_clicked(lambda _event: self._save_compare_defaults_from_selection(preview_index))
        comparefig.canvas.mpl_connect('button_press_event', on_compare_click)
        comparefig.canvas.mpl_connect('key_press_event', on_compare_key)
        comparefig.canvas.draw_idle()
        self._restore_compare_window_state(comparefig, window_state)
        self.comparefig = comparefig

    def _refresh_preview_figure(self, fig, preview_index):
        if preview_index >= len(self.preview_modes):
            return
        tmarker, x1, x2 = self.preview_modes[preview_index]
        reference_times = self._preview_reference_times_from_figure(fig, expected_tmarker=tmarker)
        waves, t_lst, active_reference_times = self._collect_preview_display_stream(
            tmarker,
            fig=fig,
            reference_times=reference_times,
        )
        if getattr(self, 'stack_mode', False):
            stack_wave_name = getattr(fig, '_stack_preview_wave_name', None) or getattr(self, 'stack_preview_active_wave_name', None)
            window = self._apply_stack_preview_window(preview_index, stack_wave_name, fig=fig)
            if window is not None:
                x1, x2 = window
        axr, axb, axp = self._preview_axes(fig)
        if axr is None or axb is None or axp is None:
            return
        if len(waves) == 0:
            self._draw_preview_content(
                fig,
                axr,
                axb,
                axp,
                tmarker,
                x1,
                x2,
                waves=waves,
                t_lst=t_lst,
                reference_times=active_reference_times,
            )
            self._attach_azimuth_selectors(fig)
            self._attach_pierce_selectors(fig)
            self._activate_pierce_selector(fig, self.preview_pierce_selection_mode)
            return
        self._draw_preview_content(
            fig,
            axr,
            axb,
            axp,
            tmarker,
            x1,
            x2,
            waves=waves,
            t_lst=t_lst,
            reference_times=active_reference_times,
        )
        self._attach_azimuth_selectors(fig)
        self._attach_pierce_selectors(fig)
        self._activate_pierce_selector(fig, self.preview_pierce_selection_mode)

    def plot(self):
        self.plot_preview(0)

    def plot_2(self):
        self.plot_preview(1)

    def Change_time_window(self):
        self.ax1.cla()
        self.ax2.cla()
        self.ax3.cla()
        self.ax4.cla()
        self.ax5.cla()
        self.plotwave()
        # self._set_gray()
        self.set_page()
        self.set_figure()

    def _persist_markers_to_disk(self):
        """Flush in-memory markers + user markers to the on-disk SAC files.

        Idempotent and safe to call mid-session (does NOT move LowQ files or
        close anything). This is the sync medium between the original-event
        window and the stack-subsystem window, which are separate WaveFigure
        instances sharing the same source SAC files on disk.
        """
        self._flush_pending_source_marker_writes(notify_review=False)
        wave_path = os.path.abspath(self.wavepath)
        marker_updates = {}
        for t_marker in range(10):
            t_str = str(t_marker)
            for sac_file, arrival_time in self.markers[t_str].items():
                marker_updates.setdefault(sac_file, {})[t_str] = arrival_time

        marker_files = set(marker_updates.keys())
        original_files = set(self.ori_sacnames)
        if getattr(self, 'stack_mode', False):
            files_to_persist = marker_files | original_files
        else:
            dirty_files = getattr(self, 'dirty_marker_wave_names', None)
            if dirty_files is None:
                files_to_persist = marker_files | original_files
            else:
                files_to_persist = set(dirty_files) & (marker_files | original_files)

        persisted_files = []
        for sac_file in sorted(files_to_persist):
            ab_sac_path = join(wave_path, sac_file)
            try:
                st = obspy.read(ab_sac_path)
            except FileNotFoundError:
                continue
            except Exception:
                if getattr(self, 'stack_mode', False):
                    continue
                raise

            sac_headers = st[0].stats.sac
            should_write = False
            stack_relative_markers = self._stack_window_relative_markers_for_wave(sac_file) if getattr(self, 'stack_mode', False) else None
            if stack_relative_markers is not None:
                relative_markers, window_length = stack_relative_markers
                sac_headers['b'] = 0.0
                sac_headers['e'] = float(window_length)
                should_write = True
                for t_str, arrival_time in relative_markers.items():
                    sac_headers[t_str] = arrival_time
            else:
                for t_str, arrival_time in marker_updates.get(sac_file, {}).items():
                    header_key = 't' + t_str
                    if math.isnan(arrival_time):
                        if hasattr(sac_headers, header_key):
                            sac_headers[header_key] = arrival_time
                            should_write = True
                    else:
                        sac_headers[header_key] = arrival_time
                        should_write = True

            if sac_file in original_files:
                sac_headers['user1'] = self._user_marker_value(sac_file, 'user1')
                sac_headers['user2'] = self._user_marker_value(sac_file, 'user2')
                sac_headers['user3'] = math.nan
                sac_headers['user4'] = self._user_marker_value(sac_file, 'user4')
                sac_headers['user5'] = self._user_marker_value(sac_file, 'user5')
                should_write = True

            if should_write:
                st.write(ab_sac_path, format='SAC')
                persisted_files.append(sac_file)
            if getattr(self, 'stack_mode', False):
                self._sync_stack_sidecar_from_markers(sac_file)
                self._sync_stack_package_sac_from_markers(sac_file)

        if getattr(self, 'stack_mode', False):
            write_stack_workspace_index(wave_path)
        if callable(getattr(self, 'stack_review_refresh_callback', None)):
            try:
                self.stack_review_refresh_callback()
            except Exception:
                pass
        dirty_files = getattr(self, 'dirty_marker_wave_names', None)
        if isinstance(dirty_files, set):
            dirty_files.difference_update(persisted_files)
        return list(sorted(persisted_files))

    def finish(self):
        """
        Write travel Time
        :return:
        """
        wave_path = os.path.abspath(self.wavepath)  # Convert relative path to absolute
        persisted_files = self._persist_markers_to_disk()
        original_files = set(self.ori_sacnames)
        # LowQ bucket move is a quit-only side effect (not done on mid-session flush).
        if not getattr(self, 'stack_mode', False):
            lowq_candidates = set(persisted_files) | {
                sac_file for sac_file in original_files
                if not math.isnan(self._user_marker_value(sac_file, 'user1'))
            }
            for sac_file in sorted(lowq_candidates):
                if sac_file not in original_files:
                    continue
                user1_value = self._user_marker_value(sac_file, 'user1')
                if not math.isnan(user1_value):
                    ab_sac_path = join(wave_path, sac_file)
                    if os.path.exists(ab_sac_path):
                        self._move_file_to_bucket(ab_sac_path, "LowQ_sac")
