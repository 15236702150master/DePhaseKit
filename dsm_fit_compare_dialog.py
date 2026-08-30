#!/usr/bin/env python3
"""DSM 正演拟合对比子系统 —— 拟合窗 + 组总览窗。

照搬 DePhaseKit 拾取窗 / 预览窗 的交互范式，做两个大窗：

* **DSM 拟合窗 (DSMFitCompareWindow)** —— 拾取窗范式。5 台站/页，每行把观测(黑)与
  DSM 理论(红)叠绘在同一 axes 上按震相对齐比较，``n``/``b`` 翻页，参数放左侧 dock。
* **DSM 组总览窗 (DSMGroupOverviewWindow)** —— 预览窗范式。把当前 group 全部台站的
  观测+理论叠绘在一张按震中距排列的大剖面上，一眼看整组拟合度。

数据准备（配对/对齐/滤波/归一）由同包的 ``dsm_fit_compare_core.build_pairs(args)`` 负责，
一次性产出已对齐/滤波/归一的 ``WaveformPair`` 列表（``observed_t/y``、``synthetic_t/y`` 已
相对对齐震相，x=0 即震相到时）。**不再调用 ``build_section_figure``**，改由本模块自绘，
以支持分页与叠绘布局。

分页算法 ``_paginate`` 与 ``WaveFigure.indexpags`` (WaveFigure.py:584) 完全一致。
"""

from __future__ import annotations

import os
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from dsm_fit_compare_core import (
    MISFIT_MODE_LINEAR,
    MISFIT_MODE_SHAPE,
    SYNTHETIC_PATTERNS,
    build_pairs as _build_pairs,
    build_station_map as _build_station_map,
    write_pair_csv,
)
from pierce_point_cache import PROJECT_ROOT
from window_geometry import maximize_on_workarea


# ---------------------------------------------------------------------------
# 上下文 —— 由 ppk 从主窗状态组装，告诉拟合窗如何自动配置 obs/synth
# ---------------------------------------------------------------------------


@dataclass
class DSMFitContext:
    """DePhaseKit 主窗传给 DSM 拟合对比窗的上下文。

    scenario:
      - 'A'      打开的是原始事件目录，且预览/拾取窗有可见集合 → 自动按
                  Jaccard 选最匹配的 dsm group，obs 限定为可见集合台站。
      - 'B'      打开的是 dsm 波形目录(-s .bhz) → synth=wavepath，obs=原始事件目录。
      - 'manual' 无可见集合且非 dsm 树 → 不自动选，保留旧手动流程。
    """

    wavepath: str
    runtime_event_dir: str
    suffix: str = ".sac"
    is_dsm_tree: bool = False
    event_id: str = ""
    obs_dir: str | None = None
    synth_dir: str | None = None
    visible_station_keys: set[str] = field(default_factory=set)
    scenario: str = "manual"


def _count_bhz(group_dir: Path) -> int:
    return sum(1 for f in group_dir.iterdir() if f.is_file() and f.suffix.lower() == ".bhz")


def scan_dsm_groups(event_id: str) -> list[tuple[Path, str]]:
    """Find all non-empty DSM group dirs for an event across both DSM trees.

    Returns (path, label) pairs; label distinguishes the tree, e.g.
    "dsm/group1_11 (7)" or "24.4/group3 (12)".
    """
    candidates = [
        PROJECT_ROOT / "data" / "dsm" / event_id,
        PROJECT_ROOT / "data" / "dsm" / "24.4" / event_id,
    ]
    groups: list[tuple[Path, str]] = []
    for root in candidates:
        if not root.is_dir():
            continue
        tree_label = root.parent.name  # "dsm" for the top tree, "24.4" for the other
        for d in sorted(root.iterdir(), key=lambda p: p.name):
            if not d.is_dir() or not d.name.startswith("group"):
                continue
            n = _count_bhz(d)
            if n > 0:
                groups.append((d, f"{tree_label}/{d.name} ({n})"))
    return groups


def _paginate(sta_num: int, maxidx: int = 5) -> tuple[int, list[np.ndarray]]:
    """与 WaveFigure.indexpags 等价的分页：返回 (总页数, 每页全局索引数组列表)。"""
    full_pages = sta_num // maxidx
    axpages = full_pages if sta_num % maxidx == 0 else full_pages + 1
    if axpages == 0:
        return 0, []
    waveidx: list[np.ndarray] = []
    for i in range(axpages - 1):
        waveidx.append(np.arange(maxidx * i, maxidx * (i + 1)))
    waveidx.append(np.arange(maxidx * (axpages - 1), sta_num))
    return axpages, waveidx


def _misfit_label(args) -> str:
    mode = getattr(args, "misfit_mode", MISFIT_MODE_SHAPE) if args else MISFIT_MODE_SHAPE
    return "Mlin" if mode == MISFIT_MODE_LINEAR else "Mshape"


def _target_delta_label(pair, args) -> str:
    residual = getattr(pair, "target_delta_residual_s", np.nan)
    target = getattr(pair, "target_phase", "")
    if not target or not np.isfinite(residual):
        return ""
    anchor = getattr(args, "align_phase", "?") if args else "?"
    return f"Δ{target}-{anchor}={residual:+.2f}s"


# ---------------------------------------------------------------------------
# 参数面板（拟合窗与旧 dialog 共用的全部控件）
# ---------------------------------------------------------------------------


class _ParamPanel(QWidget):
    """左侧参数面板：观测目录 / DSM group / 对齐 / 时窗 / 归一 / 排序 / 筛选。"""

    ALIGN_PHASES = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9",
                    "P", "pP", "sP", "ScS", "SKS", "PcP"]
    SORT_OPTIONS = [("distance", "震中距"), ("azimuth", "方位角"), ("station", "台站")]
    NORMALIZE_OPTIONS = [("separate", "各自归一"), ("pair", "成对归一")]
    TARGET_PHASE_OPTIONS = [
        ("", "不显示"),
        ("t5", "t5 / sP"),
        ("t6", "t6 / pP"),
        ("t8", "t8 / pmP"),
        ("t9", "t9 / smP"),
    ]
    MISFIT_OPTIONS = [
        (MISFIT_MODE_SHAPE, "论文 1-CC²"),
        (MISFIT_MODE_LINEAR, "线性 1-CC"),
    ]
    ALIGN_SOURCE_OPTIONS = [("header_then_taup", "头段优先→taup兜底"),
                            ("header", "仅头段"), ("taup", "仅taup现算")]
    # 与 WaveFigure._default_xlim_for_marker 一致：对齐震相 → 默认时窗 (pre, post)。
    PHASE_TIME_DEFAULTS = {
        "t0": (-10, 70), "t7": (-10, 70),
        "t2": (-40, 30), "t6": (-40, 30),
        "t3": (-50, 20), "t5": (-50, 20),
    }
    NORMALIZE_TOOLTIP = (
        "各自归一(separate): 观测/理论各自除自己的最大振幅，两条线都归一到 [-1,1]，\n"
        "只比波形形态，丢失振幅相对关系。\n"
        "成对归一(pair): 观测/理论同除两者中最大的那个，用同一比例，\n"
        "保留振幅相对大小，能看出真实振幅差异。"
    )

    def __init__(self, context: "DSMFitContext", on_groups_changed, parent=None):
        super().__init__(parent)
        self.context = context
        self.event_id = context.event_id or Path(context.obs_dir or context.wavepath).name
        self.event_dir = str(context.obs_dir or context.runtime_event_dir or context.wavepath)
        self._on_groups_changed_cb = on_groups_changed
        self._group_paths: list[Path] = []
        self._tw_user_modified = False
        self._build()

    @staticmethod
    def _wrap(layout) -> QWidget:
        w = QWidget()
        w.setLayout(layout)
        w.setMinimumWidth(0)
        w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        return w

    @classmethod
    def _phase_default_window(cls, phase: str) -> tuple[float, float]:
        return cls.PHASE_TIME_DEFAULTS.get(phase, (-10.0, 10.0))

    def _build(self):
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(6)
        form.setVerticalSpacing(8)

        # 观测目录
        obs_row = QHBoxLayout()
        self.obs_edit = QLineEdit(self.event_dir)
        self.obs_edit.setReadOnly(True)
        self.obs_edit.setMinimumWidth(0)
        obs_row.addWidget(self.obs_edit, 1)
        obs_btn = QPushButton("换…")
        obs_btn.clicked.connect(self._choose_observed_dir)
        obs_row.addWidget(obs_btn)
        form.addRow("观测目录:", self._wrap(obs_row))

        # DSM group
        self.group_combo = QComboBox()
        self.group_combo.setMinimumWidth(0)
        self.group_combo.currentIndexChanged.connect(lambda _i: None)
        grp_row = QHBoxLayout()
        grp_row.addWidget(self.group_combo, 1)
        grp_btn = QPushButton("手动…")
        grp_btn.clicked.connect(self._choose_synthetic_dir)
        grp_row.addWidget(grp_btn)
        form.addRow("DSM group:", self._wrap(grp_row))

        # 对齐震相 / 对齐来源
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(self.ALIGN_PHASES)
        self.phase_combo.setCurrentText("t0")
        self.phase_combo.currentIndexChanged.connect(self._on_phase_changed)
        form.addRow("对齐震相:", self.phase_combo)

        self.align_src_combo = QComboBox()
        for key, label in self.ALIGN_SOURCE_OPTIONS:
            self.align_src_combo.addItem(label, key)
        form.addRow("对齐来源:", self.align_src_combo)

        # 时窗（默认随对齐震相，与 DePhaseKit 主程序 _default_xlim_for_marker 一致）
        pre, post = self._phase_default_window("t0")
        win_row = QHBoxLayout()
        self.tmin_edit = QLineEdit(str(int(pre)))
        self.tmin_edit.setValidator(QDoubleValidator())
        self.tmax_edit = QLineEdit(str(int(post)))
        self.tmax_edit.setValidator(QDoubleValidator())
        self.tmin_edit.setFixedWidth(58)
        self.tmax_edit.setFixedWidth(58)
        self.tmin_edit.textEdited.connect(lambda _t: setattr(self, "_tw_user_modified", True))
        self.tmax_edit.textEdited.connect(lambda _t: setattr(self, "_tw_user_modified", True))
        win_box = QVBoxLayout()
        win_box.setContentsMargins(0, 0, 0, 0)
        win_box.setSpacing(3)
        win_row.addWidget(QLabel("pre"))
        win_row.addWidget(self.tmin_edit)
        win_row.addWidget(QLabel("post"))
        win_row.addWidget(self.tmax_edit)
        tw_reset_btn = QPushButton("重置")
        tw_reset_btn.clicked.connect(self._reset_time_window)
        win_row.addWidget(tw_reset_btn)
        win_row.addStretch(1)
        win_note = QLabel("相对对齐震相秒")
        win_note.setStyleSheet("color: #666;")
        win_box.addLayout(win_row)
        win_box.addWidget(win_note)
        form.addRow("时窗:", self._wrap(win_box))

        # 归一 / 排序
        self.norm_combo = QComboBox()
        self.norm_combo.setToolTip(self.NORMALIZE_TOOLTIP)
        for key, label in self.NORMALIZE_OPTIONS:
            self.norm_combo.addItem(label, key)
            self.norm_combo.setItemData(self.norm_combo.count() - 1, self.NORMALIZE_TOOLTIP, Qt.ToolTipRole)
        form.addRow("归一:", self.norm_combo)

        self.manual_pick_chk = QCheckBox("指标用实际到时 t7/t6/t5")
        self.manual_pick_chk.setChecked(True)
        self.manual_pick_chk.setToolTip(
            "绘图/基础对比按当前对齐震相读取，通常用理论到时 t0=P, t2=pP, t3=sP。\n"
            "互相关和残差另按实际拾取对齐：t7=P, t6=pP, t5=sP。\n"
            "如果某个实际拾取缺失，该台站会跳过，不用理论到时代替实际到时。"
        )
        form.addRow("实际拾取:", self.manual_pick_chk)

        self.target_phase_combo = QComboBox()
        self.target_phase_combo.setToolTip(
            "固定当前对齐震相为锚点，显示目标相位相对锚点的间隔残差。\n"
            "例如对齐震相=t7、目标相位=t5 时，图上显示 Δt5-t7，"
            "也就是 (观测t5-观测t7) - (合成t5-合成t7)。"
        )
        for key, label in self.TARGET_PHASE_OPTIONS:
            self.target_phase_combo.addItem(label, key)
        form.addRow("目标相位:", self.target_phase_combo)

        self.xcorr_align_chk = QCheckBox("计算互相关指标")
        self.xcorr_align_chk.setToolTip(
            "假设观测与合成已经预处理到同一采样率、同一窗口、同一单位/分量后，"
            "直接在原生采样点上计算 CC、Misfit、最优时移 τ、振幅因子 A 和 VR。\n"
            "论文模式 Misfit=1-CC²，适合只比较波形形态、弱化绝对振幅；线性模式 Misfit=1-CC，保留作对照。\n"
            "这里不做插值或重采样；指标窗口只把拾取到时吸附到最近原生采样点，采样率不同会跳过该台站。\n"
            "勾选实际到时时，指标窗口使用 t7/t6/t5，不使用 t0/t2/t3 的理论到时。"
        )
        xcorr_box = QVBoxLayout()
        xcorr_box.setContentsMargins(0, 0, 0, 0)
        xcorr_box.setSpacing(3)
        xcorr_box.addWidget(self.xcorr_align_chk)
        xcorr_row = QHBoxLayout()
        self.xcorr_tau_edit = QLineEdit("10")
        self.xcorr_tau_edit.setFixedWidth(48)
        self.xcorr_tau_edit.setValidator(QDoubleValidator(0.0, 999.0, 2))
        xcorr_row.addWidget(QLabel("τ±"))
        xcorr_row.addWidget(self.xcorr_tau_edit)
        xcorr_row.addWidget(QLabel("s"))
        self.misfit_mode_combo = QComboBox()
        self.misfit_mode_combo.setToolTip(
            "论文 1-CC²: 水层混响论文的 shape-fitting misfit，默认推荐。\n"
            "线性 1-CC: 旧演示口径，便于和过去结果对比。"
        )
        for key, label in self.MISFIT_OPTIONS:
            self.misfit_mode_combo.addItem(label, key)
        self.misfit_mode_combo.setMinimumWidth(0)
        xcorr_row.addWidget(QLabel("M"))
        xcorr_row.addWidget(self.misfit_mode_combo, 1)
        xcorr_row.addStretch(1)
        xcorr_box.addLayout(xcorr_row)
        xcorr_note = QLabel("原生采样")
        xcorr_note.setStyleSheet("color: #666;")
        xcorr_box.addWidget(xcorr_note)
        form.addRow("指标:", self._wrap(xcorr_box))

        sort_row = QHBoxLayout()
        self.sort_combo = QComboBox()
        for key, label in self.SORT_OPTIONS:
            self.sort_combo.addItem(label, key)
        self.sort_combo.setMinimumWidth(0)
        self.reverse_chk = QCheckBox("反转")
        sort_row.addWidget(self.sort_combo, 1)
        sort_row.addWidget(self.reverse_chk)
        form.addRow("排序:", self._wrap(sort_row))

        # 震中距 / 最多条数
        dist_box = QVBoxLayout()
        dist_box.setContentsMargins(0, 0, 0, 0)
        dist_box.setSpacing(3)
        dist_row = QHBoxLayout()
        self.dmin_edit = QLineEdit(); self.dmin_edit.setPlaceholderText("min")
        self.dmax_edit = QLineEdit(); self.dmax_edit.setPlaceholderText("max")
        self.dmin_edit.setValidator(QDoubleValidator()); self.dmax_edit.setValidator(QDoubleValidator())
        self.maxtraces_edit = QLineEdit(); self.maxtraces_edit.setPlaceholderText("不限")
        self.maxtraces_edit.setValidator(QDoubleValidator(1, 99999, 0))
        self.dmin_edit.setFixedWidth(62)
        self.dmax_edit.setFixedWidth(62)
        self.maxtraces_edit.setFixedWidth(74)
        dist_row.addWidget(QLabel("gcarc"))
        dist_row.addWidget(self.dmin_edit)
        dist_row.addWidget(self.dmax_edit)
        dist_row.addStretch(1)
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("最多"))
        max_row.addWidget(self.maxtraces_edit)
        max_row.addWidget(QLabel("条"))
        max_row.addStretch(1)
        dist_box.addLayout(dist_row)
        dist_box.addLayout(max_row)
        form.addRow("筛选:", self._wrap(dist_box))

        self.populate_groups()

    # ---- group scanning ----

    def populate_groups(self):
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        self._group_paths = []
        groups = scan_dsm_groups(self.event_id)
        for path, label in groups:
            self._group_paths.append(path)
            self.group_combo.addItem(label)
        if not groups:
            self.group_combo.addItem("(未找到 DSM group，请手动指目录)")
        self.group_combo.blockSignals(False)

    def group_paths(self) -> list[Path]:
        return list(self._group_paths)

    def select_synthetic_dir(self, path: Path) -> bool:
        """在 group_combo 里选中指定 path；若不在已扫描列表中则返回 False。"""
        path = Path(path)
        for idx, gp in enumerate(self._group_paths):
            if gp == path:
                self.group_combo.blockSignals(True)
                self.group_combo.setCurrentIndex(idx)
                self.group_combo.blockSignals(False)
                return True
        return False

    def set_manual_synthetic_dir(self, path: Path):
        """场景B：当前 synth 目录不在扫描结果里时，手动塞进下拉框。"""
        path = Path(path)
        self._group_paths = [path]
        self.group_combo.blockSignals(True)
        self.group_combo.clear()
        n = _count_bhz(path)
        self.group_combo.addItem(f"{path.parent.name}/{path.name} ({n})")
        self.group_combo.blockSignals(False)

    def set_observed_dir(self, path: str):
        self.obs_edit.setText(path)
        self.event_dir = path

    def _choose_observed_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择观测波形目录", self.event_dir)
        if d:
            self.obs_edit.setText(d)
            self.event_dir = d
            self.event_id = Path(d).name
            self.populate_groups()
            if self._on_groups_changed_cb:
                self._on_groups_changed_cb()

    def _choose_synthetic_dir(self):
        cur = str(self.current_synthetic_dir() or (PROJECT_ROOT / "data" / "dsm"))
        d = QFileDialog.getExistingDirectory(self, "选择 DSM group 目录", cur)
        if d:
            self.set_manual_synthetic_dir(Path(d))

    def current_synthetic_dir(self) -> Path | None:
        idx = self.group_combo.currentIndex()
        if 0 <= idx < len(self._group_paths):
            return self._group_paths[idx]
        return None

    def observed_dir(self) -> str:
        return self.obs_edit.text().strip()

    # ---- 时窗随对齐震相 ----

    def _on_phase_changed(self):
        if self._tw_user_modified:
            return
        self._apply_phase_default_window()

    def _apply_phase_default_window(self):
        pre, post = self._phase_default_window(self.phase_combo.currentText())
        self.tmin_edit.setText(str(int(pre)))
        self.tmax_edit.setText(str(int(post)))

    def _reset_time_window(self):
        self._tw_user_modified = False
        self._apply_phase_default_window()

    # ---- args ----

    def build_args(self, amplitude_scale: float = 1.0,
                   observed_station_keys: set[str] | None = None) -> Namespace | None:
        syn = self.current_synthetic_dir()
        obs = Path(self.observed_dir())
        if syn is None or not syn.is_dir():
            return None
        if not obs.is_dir():
            return None

        def fnum(edit, default=None):
            t = edit.text().strip()
            if not t:
                return default
            try:
                return float(t)
            except ValueError:
                return default

        pre_def, post_def = self._phase_default_window(self.phase_combo.currentText())
        return Namespace(
            observed_dir=obs,
            synthetic_dir=syn,
            align_phase=self.phase_combo.currentText(),
            taup_model="iasp91",
            align_source=self.align_src_combo.currentData(),
            time_min=fnum(self.tmin_edit, pre_def),
            time_max=fnum(self.tmax_edit, post_def),
            distance_min=fnum(self.dmin_edit),
            distance_max=fnum(self.dmax_edit),
            max_traces=int(fnum(self.maxtraces_edit)) if self.maxtraces_edit.text().strip() else None,
            amplitude_scale=amplitude_scale,
            observed_station_keys=observed_station_keys,
            normalize=self.norm_combo.currentData(),
            use_observed_manual_picks=self.manual_pick_chk.isChecked(),
            require_observed_manual_pick=self.manual_pick_chk.isChecked(),
            target_phase=self.target_phase_combo.currentData(),
            use_crosscorr_align=self.xcorr_align_chk.isChecked(),
            crosscorr_tau_max=fnum(self.xcorr_tau_edit, 10.0),
            misfit_mode=self.misfit_mode_combo.currentData(),
            sort_by=self.sort_combo.currentData(),
            reverse_order=self.reverse_chk.isChecked(),
            bandpass_freqmin=None,
            bandpass_freqmax=None,
        )


# ---------------------------------------------------------------------------
# 拟合窗 —— 拾取窗范式（5 台站/页，观测+理论叠绘）
# ---------------------------------------------------------------------------


class DSMFitCompareWindow(QMainWindow):
    """DSM 拟合窗：5 台站/页，每行观测(黑)与 DSM 理论(红)叠绘，n/b 翻页。"""

    MAXIDX = 5
    AMP_MIN = 0.0
    AMP_MAX = 3.0
    AMP_STEP = 0.1

    def __init__(self, context: "DSMFitContext", parent=None):
        super().__init__(parent)
        self.context = context
        self.event_dir = str(context.obs_dir or context.runtime_event_dir or context.wavepath)
        self.suffix = context.suffix or ".sac"
        self.pairs: list = []
        self.skipped: list[str] = []
        self.args: Namespace | None = None
        self.ipage = 0
        self.axpages = 0
        self._overview: DSMGroupOverviewWindow | None = None
        # 运行时振幅缩放（复刻预览窗范式：amp 只乘 y，ylim 固定）。
        self.amplitude_scale = 1.0
        # 场景A：obs 限定为可见集合；场景B/manual：None。
        self._auto_observed_station_keys: set[str] | None = None
        self._auto_plotted = False
        self.setWindowTitle("DSM 拟合窗 (观测 vs 正演)")
        self.resize(1400, 900)
        self._build_ui()
        self._define_shortcuts()
        self._auto_configure(context)

    # ---- UI ----

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # --- 参数 dock（左侧，可滚动）---
        self.param_panel = _ParamPanel(self.context, on_groups_changed=lambda: None)
        scroll = QScrollArea()
        scroll.setWidget(self.param_panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setMinimumWidth(330)
        scroll.setMaximumWidth(380)
        dock_container = QWidget()
        dv = QVBoxLayout(dock_container)
        dv.setContentsMargins(0, 0, 0, 0)
        gb = QGroupBox("参数")
        gb_layout = QVBoxLayout(gb)
        gb_layout.addWidget(scroll)
        dv.addWidget(gb)
        self.param_dock = dock_container

        # --- 顶部工具条 ---
        bar = QHBoxLayout()
        self.plot_btn = QPushButton("绘制")
        self.plot_btn.clicked.connect(self._on_plot)
        self.prev_btn = QPushButton("上一页 (b)")
        self.prev_btn.clicked.connect(self._prev)
        self.next_btn = QPushButton("下一页 (n)")
        self.next_btn.clicked.connect(self._next)
        self.overview_btn = QPushButton("组总览…")
        self.overview_btn.clicked.connect(self._open_overview)
        self.save_btn = QPushButton("存为 PNG…")
        self.save_btn.clicked.connect(self._on_save_png)
        self.csv_btn = QPushButton("存为 CSV…")
        self.csv_btn.clicked.connect(self._on_save_csv)
        self.page_lbl = QLabel("第 0/0 页")
        for w in (self.plot_btn, self.prev_btn, self.next_btn,
                  self.overview_btn, self.save_btn, self.csv_btn, self.page_lbl):
            bar.addWidget(w)

        # 振幅缩放控件（同预览窗：A- / A+ / A= + 输入框）
        bar.addWidget(QLabel("Amp"))
        self.amp_edit = QLineEdit(f"{self.amplitude_scale:g}")
        self.amp_edit.setFixedWidth(56)
        self.amp_edit.setValidator(QDoubleValidator(self.AMP_MIN, self.AMP_MAX, 3))
        self.amp_edit.returnPressed.connect(self._apply_amplitude_from_edit)
        bar.addWidget(self.amp_edit)
        self.amp_minus_btn = QPushButton("A-")
        self.amp_minus_btn.clicked.connect(lambda: self._adjust_amplitude(-self.AMP_STEP))
        self.amp_plus_btn = QPushButton("A+")
        self.amp_plus_btn.clicked.connect(lambda: self._adjust_amplitude(self.AMP_STEP))
        self.amp_equal_btn = QPushButton("A=")
        self.amp_equal_btn.clicked.connect(self._apply_amplitude_from_edit)
        for w in (self.amp_minus_btn, self.amp_plus_btn, self.amp_equal_btn):
            bar.addWidget(w)

        self.status_lbl = QLabel("尚未绘制")
        bar.addWidget(self.status_lbl, 1)
        root.addLayout(bar)

        # --- skip 明细 ---
        self.skip_view = QTextEdit()
        self.skip_view.setReadOnly(True)
        self.skip_view.setMaximumHeight(80)
        self.skip_view.setPlaceholderText("未进图的台站及原因将显示在这里")
        root.addWidget(self.skip_view)

        # --- 画布：5 行子图 ---
        self.fig = Figure(figsize=(21, 11), dpi=100, constrained_layout=True)
        self.axs = [self.fig.add_subplot(self.MAXIDX, 1, i + 1) for i in range(self.MAXIDX)]
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(0, 0)
        self.canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        root.addWidget(self.canvas, 1)

        # 把参数 dock 放到左侧
        outer = QHBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.param_dock)
        outer.addWidget(central, 1)
        wrapper = QWidget()
        wrapper.setLayout(outer)
        self.setCentralWidget(wrapper)
        self.statusBar().showMessage("就绪")

    def _define_shortcuts(self):
        sc_n = QShortcut(QKeySequence("n"), self)
        sc_n.activated.connect(self._next)
        sc_b = QShortcut(QKeySequence("b"), self)
        sc_b.activated.connect(self._prev)
        # 空格 / Esc 关闭窗口：用 QShortcut（窗口级，在子控件吃掉事件前拦截），
        # 不用 keyPressEvent——焦点在按钮/canvas 时后者不触发。
        for seq in (Qt.Key.Key_Space, Qt.Key.Key_Escape):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self.close)

    # ---- 自动配置（场景 A/B）----

    def _auto_configure(self, context: "DSMFitContext"):
        """根据上下文自动选 obs/synth 目录，并在 showEvent 时自动出图。"""
        scenario = context.scenario
        if scenario == "B":
            # synth = 当前 dsm group 目录（wavepath）
            if context.synth_dir:
                synth_path = Path(context.synth_dir)
                if not self.param_panel.select_synthetic_dir(synth_path):
                    self.param_panel.set_manual_synthetic_dir(synth_path)
            if context.obs_dir:
                self.param_panel.set_observed_dir(context.obs_dir)
            self._auto_observed_station_keys = None
            self.statusBar().showMessage(
                f"场景B: synth={context.synth_dir}  obs={context.obs_dir}", 5000)
        elif scenario == "A":
            self._auto_observed_station_keys = set(context.visible_station_keys)
            best = self._best_matching_group(context.visible_station_keys)
            if best is not None:
                self.param_panel.select_synthetic_dir(best)
                self.statusBar().showMessage(
                    f"场景A: 已按可见集合({len(context.visible_station_keys)}台)自动匹配 DSM group", 5000)
            else:
                self.statusBar().showMessage("场景A: 未找到匹配的 DSM group，请手动选", 5000)
        else:
            self._auto_observed_station_keys = None

    def _best_matching_group(self, visible_keys: set[str]) -> Path | None:
        """在所有已扫描 group 里选台站 key 集合与可见集合 Jaccard 最高的。"""
        if not visible_keys:
            return None
        best_path: Path | None = None
        best_jac = -1.0
        best_inter = -1
        for path in self.param_panel.group_paths():
            try:
                synth_keys = set(_build_station_map(path, SYNTHETIC_PATTERNS))
            except Exception:  # noqa: BLE001
                continue
            if not synth_keys:
                continue
            inter = visible_keys & synth_keys
            union = visible_keys | synth_keys
            jac = len(inter) / len(union) if union else 0.0
            if jac > best_jac or (abs(jac - best_jac) < 1e-9 and len(inter) > best_inter):
                best_path, best_jac, best_inter = path, jac, len(inter)
        return best_path

    # ---- 振幅缩放（预览窗范式）----

    def _set_amplitude(self, new_scale: float):
        try:
            scale = float(new_scale)
        except (TypeError, ValueError):
            return
        clamped = min(self.AMP_MAX, max(self.AMP_MIN, scale))
        if abs(clamped - self.amplitude_scale) < 1e-9:
            self._sync_amp_edit()
            return
        self.amplitude_scale = clamped
        if self.args is not None:
            self.args.amplitude_scale = clamped
        self._sync_amp_edit()
        if self.pairs:
            self._plot_page()
        if self._overview is not None:
            self._overview.set_amplitude_from_parent(clamped)

    def _adjust_amplitude(self, delta: float):
        self._set_amplitude(self.amplitude_scale + delta)

    def _apply_amplitude_from_edit(self):
        try:
            self._set_amplitude(float(self.amp_edit.text()))
        except ValueError:
            self._sync_amp_edit()

    def _sync_amp_edit(self):
        self.amp_edit.setText(f"{self.amplitude_scale:g}")
        self.statusBar().showMessage(f"Amp x{self.amplitude_scale:g}", 2000)

    # ---- 绘制 ----

    def _on_plot(self):
        args = self.param_panel.build_args(self.amplitude_scale, self._auto_observed_station_keys)
        if args is None:
            self.status_lbl.setText("未选择有效的观测 / DSM group 目录")
            return
        self.plot_btn.setEnabled(False)
        self.status_lbl.setText("绘制中…")
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            pairs, skipped = _build_pairs(args)
            self.pairs = pairs
            self.skipped = skipped
            self.args = args
            self.ipage = 0
            self.axpages, _ = _paginate(len(pairs), self.MAXIDX)
            if not pairs:
                self.status_lbl.setText("无配对台站，未出图")
                self._show_skipped(skipped)
                for ax in self.axs:
                    ax.cla(); ax.axis('off')
                self.canvas.draw_idle()
                self._update_page_label()
                return
            self._plot_page()
            target_residuals = [
                p.target_delta_residual_s
                for p in pairs
                if np.isfinite(getattr(p, "target_delta_residual_s", np.nan))
            ]
            target_summary = ""
            if target_residuals:
                target = getattr(args, "target_phase", "")
                target_summary = f"  meanΔ{target}-{args.align_phase}={float(np.mean(target_residuals)):+.2f}s"
            if getattr(args, "use_crosscorr_align", False):
                mean_cc = float(np.mean([p.cross_corr_max for p in pairs])) if pairs else 0.0
                mean_misfit = float(np.mean([p.misfit_value for p in pairs])) if pairs else 0.0
                mean_vr = float(np.mean([p.variance_reduction for p in pairs])) if pairs else 0.0
                misfit_label = _misfit_label(args)
                self.status_lbl.setText(
                    f"matched={len(pairs)}  skipped={len(skipped)}  "
                    f"meanCC={mean_cc:.2f}  mean{misfit_label}={mean_misfit:.2f}  meanVR={mean_vr:.2f}"
                    f"{target_summary}"
                )
            else:
                self.status_lbl.setText(f"matched={len(pairs)}  skipped={len(skipped)}{target_summary}")
            self._show_skipped(skipped)
            if self._overview is not None:
                self._overview.refresh(pairs, args)
        except SystemExit as exc:
            self.status_lbl.setText(f"未出图: {exc}")
        except Exception as exc:  # noqa: BLE001
            self.status_lbl.setText(f"出错: {exc}")
            import traceback
            self.skip_view.setPlainText(traceback.format_exc())
        finally:
            self.plot_btn.setEnabled(True)

    def _plot_page(self):
        """绘制当前页的 5 个台站，每行观测(黑)+理论(红)叠绘。"""
        args = self.args
        amp = self.amplitude_scale
        tmin = getattr(args, "time_min", -10.0) if args else -10.0
        tmax = getattr(args, "time_max", 70.0) if args else 70.0
        _, page_idx = _paginate(len(self.pairs), self.MAXIDX)
        idxs = page_idx[self.ipage] if page_idx else np.array([], dtype=int)

        for slot, ax in enumerate(self.axs):
            ax.cla()
            if slot < len(idxs):
                pair = self.pairs[int(idxs[slot])]
                ax.plot(pair.observed_t, amp * pair.observed_y,
                        color="black", linewidth=0.9, alpha=0.95)
                ax.plot(pair.synthetic_t, amp * pair.synthetic_y,
                        color="red", linewidth=0.9, alpha=0.85)
                ax.axvline(0.0, color="0.65", linewidth=0.9, linestyle="--")
                ax.set_xlim(tmin, tmax)
                # y 已归一到 [-1,1]，ylim 固定，让 amp 只控制波形胖瘦（修旧 bug）。
                ax.set_ylim(-1.3, 1.3)
                ax.grid(axis="x", color="0.9", linewidth=0.5)
                az_txt = f"  az={pair.azimuth_deg:.0f}" if pair.azimuth_deg is not None else ""
                ax.text(0.01, 0.95, f"{pair.station_key}  dis={pair.distance_deg:.2f}{az_txt}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=10, color="#F4606C")
                metric_parts = []
                if getattr(args, "use_crosscorr_align", False):
                    misfit_label = _misfit_label(args)
                    metric_parts.append(
                        f"CC={pair.cross_corr_max:.2f}  {misfit_label}={pair.misfit_value:.2f}  "
                        f"τ={pair.time_shift_s:+.2f}s  A={pair.amplitude_factor:.2f}  "
                        f"VR={pair.variance_reduction:.2f}"
                    )
                target_txt = _target_delta_label(pair, args)
                if target_txt:
                    metric_parts.append(target_txt)
                if metric_parts:
                    metrics_txt = "\n".join(metric_parts)
                    ax.text(0.99, 0.95, metrics_txt,
                            transform=ax.transAxes, ha="right", va="top",
                            fontsize=9, color="#1E88A8",
                            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.72", alpha=0.86))
                ax.set_ylabel("amp", fontsize=8)
                if slot < self.MAXIDX - 1:
                    ax.tick_params(labelbottom=False)
            else:
                ax.axis('off')
        self.canvas.draw_idle()
        self._update_page_label()

    def _update_page_label(self):
        total = max(self.axpages, 0 if not self.pairs else 1)
        cur = self.ipage + 1 if self.pairs else 0
        self.page_lbl.setText(f"第 {cur}/{total} 页")

    def _prev(self):
        if not self.pairs or self.ipage <= 0:
            return
        self.ipage -= 1
        self._plot_page()

    def _next(self):
        if not self.pairs or self.ipage >= self.axpages - 1:
            return
        self.ipage += 1
        self._plot_page()

    def _show_skipped(self, skipped: list[str]):
        if not skipped:
            self.skip_view.setPlainText("")
            return
        self.skip_view.setPlainText("\n".join(f"skip: {s}" for s in skipped))

    def _open_overview(self):
        # 若已有总览窗且仍可见 → 置顶刷新；已关闭（或 Qt 对象已销毁）则清理引用新建。
        ov = self._overview
        if ov is not None:
            try:
                still_alive = ov.isVisible()
            except RuntimeError:
                still_alive = False
            if still_alive:
                try:
                    ov.raise_()
                    ov.activateWindow()
                except Exception:
                    pass
                if self.pairs and self.args is not None:
                    ov.refresh(self.pairs, self.args)
                return
            self._overview = None
        win = DSMGroupOverviewWindow(self.pairs, self.args, parent=self)
        win.show()
        self._overview = win
        self.statusBar().showMessage("已打开组总览窗", 3000)

    def _on_save_png(self):
        if not self.pairs:
            self.status_lbl.setText("请先绘制对比图")
            return
        default = f"{Path(self.event_dir).name}_dsm_fit.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存拟合窗 PNG", default, "PNG (*.png)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=220)
            self.status_lbl.setText(f"已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            self.status_lbl.setText(f"保存失败: {exc}")

    def _on_save_csv(self):
        if not self.pairs:
            self.status_lbl.setText("请先绘制对比图")
            return
        default = f"{Path(self.event_dir).name}_dsm_fit_pairs.csv"
        path, _ = QFileDialog.getSaveFileName(self, "保存配对与指标 CSV", default, "CSV (*.csv)")
        if not path:
            return
        try:
            write_pair_csv(Path(path), self.pairs)
            self.status_lbl.setText(f"已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            self.status_lbl.setText(f"保存失败: {exc}")

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_geom_set", False):
            # 与 DePhaseKit 主拾取窗一致：最大化铺满工作区（WSLg 回退居中近全屏）。
            maximize_on_workarea(self, frac=0.98)
            self._geom_set = True
        if not self._auto_plotted:
            self._auto_plotted = True
            # 场景 A/B：打开即自动出图；manual 不自动（无可见集合且非 dsm 树）。
            if self.context.scenario in ("A", "B"):
                QTimer.singleShot(0, self._on_plot)


# ---------------------------------------------------------------------------
# 组总览窗 —— 预览窗范式（整组按震中距排列的 obs/synth 剖面）
# ---------------------------------------------------------------------------


class DSMGroupOverviewWindow(QMainWindow):
    """DSM 组总览窗：整组全部台站的观测(黑)+理论(红)叠绘，按震中距排列。"""

    AMP_MIN = 0.0
    AMP_MAX = 3.0
    AMP_STEP = 0.1

    def __init__(self, pairs, args, parent=None):
        super().__init__(parent)
        self.pairs = pairs or []
        self.args = args
        self.amplitude_scale = float(getattr(args, "amplitude_scale", 1.0) or 1.0)
        self.setWindowTitle("DSM 组总览窗 (观测 vs 正演)")
        self.resize(900, 1100)
        self._build_ui()
        self._draw()
        self._define_shortcuts()

    def _define_shortcuts(self):
        # 空格 / Esc 关闭窗口：用 QShortcut（窗口级），避免焦点在"存为PNG"按钮上时
        # 空格被当成点击按钮。
        for seq in (Qt.Key.Key_Space, Qt.Key.Key_Escape):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self.close)

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        bar = QHBoxLayout()
        save_btn = QPushButton("存为 PNG…")
        save_btn.clicked.connect(self._on_save_png)

        # 振幅缩放控件（同拟合窗/预览窗）
        self.amp_edit = QLineEdit(f"{self.amplitude_scale:g}")
        self.amp_edit.setFixedWidth(56)
        self.amp_edit.setValidator(QDoubleValidator(self.AMP_MIN, self.AMP_MAX, 3))
        self.amp_edit.returnPressed.connect(self._apply_amplitude_from_edit)
        self.amp_minus_btn = QPushButton("A-")
        self.amp_minus_btn.clicked.connect(lambda: self._adjust_amplitude(-self.AMP_STEP))
        self.amp_plus_btn = QPushButton("A+")
        self.amp_plus_btn.clicked.connect(lambda: self._adjust_amplitude(self.AMP_STEP))
        self.amp_equal_btn = QPushButton("A=")
        self.amp_equal_btn.clicked.connect(self._apply_amplitude_from_edit)

        bar.addWidget(save_btn)
        bar.addWidget(QLabel("Amp"))
        bar.addWidget(self.amp_edit)
        for w in (self.amp_minus_btn, self.amp_plus_btn, self.amp_equal_btn):
            bar.addWidget(w)
        self.status_lbl = QLabel("")
        bar.addWidget(self.status_lbl, 1)
        root.addLayout(bar)

        self.fig = Figure(figsize=(11, 14), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.canvas, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")

    def refresh(self, pairs, args):
        self.pairs = pairs or []
        self.args = args
        if args is not None and getattr(args, "amplitude_scale", None) is not None:
            self.amplitude_scale = float(args.amplitude_scale)
            self._sync_amp_edit()
        self._draw()

    def set_amplitude_from_parent(self, scale: float):
        """父窗改 amp 时单向同步到本窗。"""
        self.amplitude_scale = min(self.AMP_MAX, max(self.AMP_MIN, float(scale)))
        if self.args is not None:
            self.args.amplitude_scale = self.amplitude_scale
        self._sync_amp_edit()
        self._draw()

    def _set_amplitude(self, new_scale: float):
        try:
            scale = float(new_scale)
        except (TypeError, ValueError):
            return
        clamped = min(self.AMP_MAX, max(self.AMP_MIN, scale))
        if abs(clamped - self.amplitude_scale) < 1e-9:
            self._sync_amp_edit()
            return
        self.amplitude_scale = clamped
        if self.args is not None:
            self.args.amplitude_scale = clamped
        self._sync_amp_edit()
        self._draw()

    def _adjust_amplitude(self, delta: float):
        self._set_amplitude(self.amplitude_scale + delta)

    def _apply_amplitude_from_edit(self):
        try:
            self._set_amplitude(float(self.amp_edit.text()))
        except ValueError:
            self._sync_amp_edit()

    def _sync_amp_edit(self):
        self.amp_edit.setText(f"{self.amplitude_scale:g}")

    def _draw(self):
        self.fig.clf()
        if not self.pairs:
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, "无配对台站", ha="center", va="center", transform=ax.transAxes)
            self.canvas.draw_idle()
            self.status_lbl.setText("无配对台站")
            return
        args = self.args
        amp = self.amplitude_scale
        tmin = getattr(args, "time_min", -10.0) if args else -10.0
        tmax = getattr(args, "time_max", 70.0) if args else 70.0
        ax = self.fig.add_subplot(111)
        distances = []
        use_xcorr = bool(getattr(args, "use_crosscorr_align", False)) if args else False
        use_target = bool(getattr(args, "target_phase", "")) if args else False
        for pair in self.pairs:
            offset = pair.distance_deg
            distances.append(offset)
            ax.plot(pair.observed_t, offset + amp * pair.observed_y,
                    color="black", linewidth=0.9, alpha=0.95)
            ax.plot(pair.synthetic_t, offset + amp * pair.synthetic_y,
                    color="red", linewidth=0.9, alpha=0.85)
            if use_xcorr or use_target:
                label_parts = []
                if use_xcorr:
                    label_parts.append(f"CC={pair.cross_corr_max:.2f} {_misfit_label(args)}={pair.misfit_value:.2f}")
                target_txt = _target_delta_label(pair, args)
                if target_txt:
                    label_parts.append(target_txt)
                if label_parts:
                    ax.text(tmax, offset, " " + " ".join(label_parts),
                            fontsize=7, color="#1E88A8", va="center", ha="left")
        ax.axvline(0.0, color="0.65", linewidth=0.9, linestyle="--")
        ax.set_xlim(tmin, tmax)
        ax.set_ylim(min(distances) - 1.0, max(distances) + 1.0)
        ax.grid(axis="y", color="0.88", linewidth=0.6)
        phase = getattr(args, "align_phase", "?") if args else "?"
        ax.set_xlabel(f"Time (s) aligned on {phase}")
        ax.set_ylabel("Distance (deg)")
        obs_dir = getattr(args, "observed_dir", None) if args else None
        obs_name = obs_dir.name if hasattr(obs_dir, "name") else (str(obs_dir) if obs_dir else "")
        title = f"Observed (black) vs Synthetic (red)\n{obs_name}"
        if use_xcorr:
            mean_cc = float(np.mean([p.cross_corr_max for p in self.pairs]))
            mean_misfit = float(np.mean([p.misfit_value for p in self.pairs]))
            mean_vr = float(np.mean([p.variance_reduction for p in self.pairs]))
            title = (
                f"Observed (black) vs Synthetic (red)  |  mean CC={mean_cc:.2f} "
                f"mean {_misfit_label(args)}={mean_misfit:.2f} mean VR={mean_vr:.2f}\n{obs_name}"
            )
        elif use_target:
            residuals = [
                p.target_delta_residual_s
                for p in self.pairs
                if np.isfinite(getattr(p, "target_delta_residual_s", np.nan))
            ]
            if residuals:
                title = (
                    f"Observed (black) vs Synthetic (red)  |  "
                    f"mean Δ{args.target_phase}-{args.align_phase}={float(np.mean(residuals)):+.2f}s\n{obs_name}"
                )
        ax.set_title(title, pad=12)
        from matplotlib.lines import Line2D
        proxies = [Line2D([0], [0], color="black", lw=1.2),
                   Line2D([0], [0], color="red", lw=1.2)]
        # 加大 handle 与文字、列与列之间的间距，避免线段和文字重叠。
        self.fig.legend(proxies, ["Observed", "Synthetic"],
                        loc="upper center", bbox_to_anchor=(0.5, 0.965),
                        ncol=2, frameon=False,
                        handlelength=2.2, handletextpad=1.2, columnspacing=3.0,
                        borderaxespad=0.5)
        self.fig.subplots_adjust(top=0.88, left=0.09, right=0.975, bottom=0.07)
        self.canvas.draw_idle()
        self.status_lbl.setText(f"已绘制 {len(self.pairs)} 个台站")

    def _on_save_png(self):
        if not self.pairs:
            self.status_lbl.setText("无图可存")
            return
        default = "dsm_group_overview.png"
        path, _ = QFileDialog.getSaveFileName(self, "保存组总览 PNG", default, "PNG (*.png)")
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=220)
            self.status_lbl.setText(f"已保存: {path}")
        except Exception as exc:  # noqa: BLE001
            self.status_lbl.setText(f"保存失败: {exc}")

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_geom_set", False):
            # 与 DePhaseKit 主拾取窗一致：最大化铺满工作区（WSLg 回退居中近全屏）。
            maximize_on_workarea(self, frac=0.98)
            self._geom_set = True
