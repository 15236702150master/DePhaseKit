#!/usr/bin/env python3
"""Stack 厚度审阅交互窗。

三视图联动：
  - 左：选中 group 的 stack 预览（TopDist / Overlay），Step 5 接入
  - 右上：穿透点底图（GMT 地形底图 + 厚度色表散点 + 事件红星 + colorbar）
  - 右下：event/group 表格（可排序、可标记状态）

选中状态三视图双向联动（Step 4）。右键直跳主 ppk / stack 子系统（Step 7）。

范式照搬 ``dsm_fit_compare_dialog.DSMFitCompareWindow``：QMainWindow + 工具条 +
FigureCanvas + QShortcut。数据层见 ``stack_thickness_review_core``，索引层见
``stack_group_thickness_index``。穿透点统一 24.4 km / prem。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QItemSelectionModel, QEvent, QTimer
from PySide6.QtGui import QAction, QCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableView,
    QTreeWidget,
    QTreeWidgetItem,
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

from stack_group_thickness_index import (
    DEFAULT_PIERCE_DEPTH_KM,
    DEFAULT_PIERCE_MODEL,
    REVIEW_STATUS_VALUES,
    STAR_COLORS,
    ThicknessIndex,
    ThicknessPoint,
    build_thickness_index,
    invalidate_cache,
    load_review_marks,
    set_review_mark,
)
from stack_thickness_review_core import compute_outlier_score, load_member_pierce_points
from window_geometry import maximize_on_workarea

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DPK_ROOT = Path(__file__).resolve().parent
BASEMAP_HELPER = DPK_ROOT / "plot_standard_pierce_map.sh"
DEFAULT_SCAN_ROOT = PROJECT_ROOT / "data" / "output" / "stack" / "analysis"
DEFAULT_REGION = "-33.5/-23/-61/-55"  # 南桑威奇，与 overview 脚本一致
THICKNESS_MIN = 6.0
THICKNESS_MAX = 20.0
OUTLIER_THRESHOLD = 3.0  # |z| 超此值视为异常描边

# 每个事件分配不同形状（同事件同形状），形状区分事件、颜色表示厚度，两维正交。
# 顺序与事件出现顺序对应；事件中心用灰色 + 大尺寸同形状。
# 全 filled marker，避免 unfilled marker(数字/+/x)忽略 edgecolor 的警告。
# 含字母 filled + 自定义 Path 形状（正多边形 / 同心圆 / 新月 / 太阳 / 梅花 / 空心星）。
def _regular_polygon_marker(n: int, rotation: float = 0.0):
    import numpy as np
    from matplotlib.path import Path
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    verts = np.c_[np.cos(angles), np.sin(angles)]
    verts = np.vstack([verts, verts[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _ring_marker(r_outer: float, r_inner: float, n: int = 64):
    """同心圆：外圆 + 内孔（反向绕向，filled 时中间留孔）。"""
    import numpy as np
    from matplotlib.path import Path
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    outer = np.c_[r_outer * np.cos(t), r_outer * np.sin(t)]
    inner = np.c_[r_inner * np.cos(t[::-1]), r_inner * np.sin(t[::-1])]
    verts = np.concatenate([outer, [outer[0]], inner, [inner[0]]])
    codes = ([Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _crescent_marker(r_outer: float = 1.0, r_inner: float = 0.85, dx: float = 0.45, n: int = 64):
    """新月形：大圆减去偏移的小圆。"""
    import numpy as np
    from matplotlib.path import Path
    t = np.linspace(0, 2 * np.pi, n, endpoint=True)
    outer = np.c_[r_outer * np.cos(t), r_outer * np.sin(t)]
    inner = np.c_[r_inner * np.cos(t[::-1]) + dx, r_inner * np.sin(t[::-1])]
    verts = np.concatenate([outer, [outer[0]], inner, [inner[0]]])
    codes = ([Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _star_marker(r_outer: float, r_inner: float, n_spikes: int, rotation: float = np.pi / 2):
    """星形/太阳形/梅花：交替外/内半径的多边形。n_spikes=尖角数。"""
    import numpy as np
    from matplotlib.path import Path
    k = 2 * n_spikes
    ang = np.linspace(rotation, rotation + 2 * np.pi, k, endpoint=False)
    r = np.empty(k)
    r[0::2] = r_outer
    r[1::2] = r_inner
    verts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    verts = np.vstack([verts, verts[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (k - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _hollow_star_marker(n_spikes: int = 5, r_outer: float = 1.0, r_inner: float = 0.4, r_hole: float = 0.5):
    """空心星：5 角星外轮廓 + 中心反向小五边形孔。"""
    import numpy as np
    from matplotlib.path import Path
    k = 2 * n_spikes
    ang = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, k, endpoint=False)
    r = np.empty(k)
    r[0::2] = r_outer
    r[1::2] = r_inner
    outer = np.c_[r * np.cos(ang), r * np.sin(ang)]
    outer = np.vstack([outer, outer[0]])
    ang2 = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, n_spikes, endpoint=False)
    inner = np.c_[r_hole * np.cos(ang2[::-1]), r_hole * np.sin(ang2[::-1])]
    inner = np.vstack([inner, inner[0]])
    verts = np.concatenate([outer, inner])
    codes = ([Path.MOVETO] + [Path.LINETO] * (len(outer) - 2) + [Path.CLOSEPOLY]
             + [Path.MOVETO] + [Path.LINETO] * (len(inner) - 2) + [Path.CLOSEPOLY])
    return Path(verts, codes)


def _rounded_rect_marker(w: float = 1.6, h: float = 1.0, r: float = 0.25, n: int = 16):
    """圆角矩形：4 段圆弧角 + 直边。"""
    import numpy as np
    from matplotlib.path import Path
    hw, hh = w / 2, h / 2
    corners = [
        (hw - r, hh - r, 0, np.pi / 2),            # 右上
        (-hw + r, hh - r, np.pi / 2, np.pi),       # 左上
        (-hw + r, -hh + r, np.pi, 3 * np.pi / 2),  # 左下
        (hw - r, -hh + r, 3 * np.pi / 2, 2 * np.pi),  # 右下
    ]
    arcs = []
    codes = []
    first = True
    for cx, cy, a0, a1 in corners:
        t = np.linspace(a0, a1, n)
        arc = np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
        arcs.append(arc)
        codes.append([Path.MOVETO if first else Path.LINETO] + [Path.LINETO] * (n - 1))
        first = False
    verts = np.vstack([np.concatenate(arcs), np.concatenate(arcs)[0]])
    all_codes = np.concatenate(codes).tolist() + [Path.CLOSEPOLY]
    return Path(verts, all_codes)


def _cloud_marker(n: int = 24):
    """云彩形：底部平直，顶部由若干半圆凸起构成。"""
    import numpy as np
    from matplotlib.path import Path
    # 底部从左到右，再沿顶部多个半圆凸起回到左
    bumps = [(-0.9, 0.0, 0.45), (-0.35, 0.25, 0.45), (0.25, 0.3, 0.5), (0.8, 0.05, 0.4)]
    verts = [(-0.9, -0.35), (0.9, -0.35)]  # 底边
    codes = [Path.MOVETO, Path.LINETO]
    # 右侧上行 + 顶部凸起 + 左侧下行
    for cx, cy, r in bumps:
        # 半圆从右到左（顶部）
        t = np.linspace(0, np.pi, n)
        arc = np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
        for v in arc:
            verts.append((float(v[0]), float(v[1])))
            codes.append(Path.LINETO)
    verts.append((-0.9, -0.35))
    codes.append(Path.CLOSEPOLY)
    return Path(np.array(verts, dtype=float), codes)


def _heart_marker(n: int = 48):
    """爱心形：心形参数曲线。"""
    import numpy as np
    from matplotlib.path import Path
    t = np.linspace(0, 2 * np.pi, n)
    x = 0.55 * 16 * np.sin(t) ** 3 / 16.0
    y = 0.55 * (13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)) / 16.0
    verts = np.vstack([np.c_[x, y], np.c_[x[0], y[0]]])
    codes = [Path.MOVETO] + [Path.LINETO] * (n - 1) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _droplet_marker():
    """水滴形：上尖下圆的单闭合路径。"""
    import numpy as np
    from matplotlib.path import Path
    left = np.c_[np.linspace(0, -0.5, 15), np.linspace(1, 0, 15)]
    tc = np.linspace(np.pi, 2 * np.pi, 25)
    bot = np.c_[0.5 * np.cos(tc), 0.5 * np.sin(tc) - 0.5]
    right = np.c_[np.linspace(0.5, 0, 15), np.linspace(0, 1, 15)]
    verts = np.vstack([left, bot, right, [left[0, 0], left[0, 1]]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _lightning_marker():
    """闪电形：锯齿状闭合多边形。"""
    import numpy as np
    from matplotlib.path import Path
    verts = np.array([
        (0.1, 1.0), (-0.5, 0.1), (0.0, 0.1), (-0.1, -1.0), (0.5, 0.0), (0.0, 0.0), (0.3, 1.0),
    ], dtype=float)
    codes = [Path.MOVETO] + [Path.LINETO] * 5 + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _parallelogram_marker(shear: float = 0.45):
    """平行四边形：左右两条斜边。shear=水平剪切量。"""
    import numpy as np
    from matplotlib.path import Path
    verts = np.array([
        (-0.6 + shear, 1.0), (0.6 + shear, 1.0),
        (0.6 - shear, -1.0), (-0.6 - shear, -1.0),
        (-0.6 + shear, 1.0),
    ], dtype=float)
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _trapezoid_marker(top_width: float = 0.8, bottom_width: float = 1.4, height: float = 1.8):
    """等腰梯形：上底短、下底长，方向稳定，便于和 1997 现有符号区分。"""
    import numpy as np
    from matplotlib.path import Path
    ht = height / 2.0
    tw = top_width / 2.0
    bw = bottom_width / 2.0
    verts = np.array([
        (-tw, ht),
        (tw, ht),
        (bw, -ht),
        (-bw, -ht),
        (-tw, ht),
    ], dtype=float)
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _sector_marker(r: float = 1.0, start_deg: float = 60.0, end_deg: float = 300.0, n: int = 40):
    """扇形：圆心 + 两直边 + 圆弧（饼图切片状）。"""
    import numpy as np
    from matplotlib.path import Path
    a0 = np.deg2rad(start_deg)
    a1 = np.deg2rad(end_deg)
    t = np.linspace(a0, a1, n)
    arc = np.c_[r * np.cos(t), r * np.sin(t)]
    verts = np.vstack([[(0.0, 0.0)], arc, [(0.0, 0.0)]])
    codes = [Path.MOVETO] + [Path.LINETO] * n + [Path.CLOSEPOLY]
    return Path(verts, codes)


def _mirror_marker_h(marker):
    """左右镜像一个 Path marker，用来生成“同符号反方向”版本。"""
    import numpy as np
    from matplotlib.path import Path
    if not hasattr(marker, "vertices"):
        if marker == "<":
            return ">"
        if marker == ">":
            return "<"
        return marker
    verts = np.asarray(marker.vertices, dtype=float).copy()
    verts[:, 0] *= -1.0
    codes = None if marker.codes is None else np.asarray(marker.codes).copy()
    return Path(verts, codes)


def _diamond_marker(r: float = 1.0):
    """正菱形：上下左右四点，对角线等长（区别于 matplotlib 内置 D 的宽菱形）。"""
    import numpy as np
    from matplotlib.path import Path
    verts = np.array([(0, r), (r, 0), (0, -r), (-r, 0), (0, r)], dtype=float)
    codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    return Path(verts, codes)


EVENT_MARKERS = [
    "o", "s", "^", _diamond_marker(), "v",
    _ring_marker(1.0, 0.5),                  # 同心圆
    _droplet_marker(),                        # 水滴
    "<", ">",
    _sector_marker(),                         # 扇形
    _lightning_marker(),                      # 闪电
    _star_marker(1.0, 0.45, n_spikes=5),     # 梅花(5 瓣)
    "H",
    _parallelogram_marker(),                  # 平行四边形
    _star_marker(1.0, 0.55, n_spikes=12),    # 填充太阳(12 尖)
    _rounded_rect_marker(),                   # 圆角矩形
    _regular_polygon_marker(9),
    _regular_polygon_marker(10),
    _regular_polygon_marker(12),
]

_YEAR_2010_MARKER = _sector_marker()
YEAR_MARKER_OVERRIDE = {
    "2010": _YEAR_2010_MARKER,
    "2013": _trapezoid_marker(),
}
PIERCE_POINT_SIZE = 55       # 穿透点圆圈尺寸（原 120，缩小）
EVENT_CENTER_SIZE = 200      # 事件中心符号尺寸（大）


@dataclass
class ThicknessReviewContext:
    scan_root: str
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM
    model: str = DEFAULT_PIERCE_MODEL
    # 右键直跳回调（由 ppk 主窗注入）：
    #   open_event_callback(event_dir, group_name) — 打开源事件拾取窗，仅显该 group 成员
    #   open_stack_callback(event_dir, stack_wave_name) — 打开该 group 的 stack 工作区拾取窗
    open_event_callback: Callable[[str, str], None] | None = None
    open_stack_callback: Callable[[str, str], None] | None = None


# ---------------------------------------------------------------------------
# 表格模型
# ---------------------------------------------------------------------------

COLUMNS = [
    ("event", "Event"),
    ("group_name", "Group"),
    ("map_color", "Map"),       # 该 group 在穿透点图上的符号+厚度色，便于对照
    ("pair_kind", "Pair"),
    ("phase_kind", "Phase"),
    ("align_marker", "Align"),
    ("member_count_used", "N"),
    ("gcarc", "gcarc"),
    ("thickness_km", "Thick(km)"),
    ("outlier_z", "|z|"),
    ("status", "Status"),
]

# Status 列用彩色圆圈替代文字，扫一眼即知审阅状态。
STATUS_COLORS = {
    "pending": "#9aa0a6",   # 灰 — 待查
    "suspect": "#e8443a",   # 红 — 可疑
    "fixed":   "#2ea043",   # 绿 — 已修
    "ignore":  "#f0c020",   # 黄 — 忽略
}
STATUS_LABELS = {
    "pending": "待查",
    "suspect": "可疑",
    "fixed":   "已修",
    "ignore":  "忽略",
}
STATUS_PIXMAP_CACHE: dict[str, object] = {}


def _status_pixmap(status: str):
    """返回状态对应的彩色圆圈 QPixmap（缓存）。"""
    from PySide6.QtGui import QColor, QPainter, QPixmap, QPen
    pix = STATUS_PIXMAP_CACHE.get(status)
    if pix is not None:
        return pix
    color = QColor(STATUS_COLORS.get(status, STATUS_COLORS["pending"]))
    pix = QPixmap(18, 18)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(color)
    p.setPen(QPen(QColor(255, 255, 255, 200), 1))
    p.drawEllipse(2, 2, 14, 14)
    p.end()
    STATUS_PIXMAP_CACHE[status] = pix
    return pix


def event_marker_map(event_keys) -> dict:
    """事件 key → marker（按出现顺序分配，与地图一致）。"""
    mapping = {}
    for i, ek in enumerate(event_keys):
        marker = EVENT_MARKERS[i % len(EVENT_MARKERS)]
        event_text = str(ek).split("/", 1)[-1]
        year = event_text[:4] if len(event_text) >= 4 and event_text[:4].isdigit() else ""
        mapping[ek] = YEAR_MARKER_OVERRIDE.get(year, marker)
    return mapping


_MAP_PIXMAP_CACHE: dict[tuple, object] = {}


def _map_color_pixmap(marker, thickness_km: float, *, edge_color: str = "black", line_width: float = 0.8):
    """渲染该 group 在地图上的符号+厚度色为 QPixmap（缓存），用于表格 Map 列对照。

    注意：用显式 ``FigureCanvasAgg`` 离屏渲染，**不能**调 ``matplotlib.use("Agg")``。
    ``matplotlib.use`` 会全局切走 pyplot 后端（QtAgg→Agg），之后 WaveFigure.plot_preview
    里 ``plt.figure(...)`` 拿到的是 ``FigureCanvasAgg``、无 Qt window → 预览窗打不开
    （FigureCanvasAgg is non-interactive 警告 + 快捷键重试死循环）。本模块主图/地图
    预览用的是显式 ``Figure``+``FigureCanvas(QTAgg)`` 模式，此处照此模式用 Agg canvas
    离屏出图即可，不污染全局 backend。
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.cm import get_cmap
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    import io as _io
    from PySide6.QtGui import QPixmap
    cache_key = (id(marker), round(thickness_km, 2), edge_color, round(float(line_width), 2))
    pix = _MAP_PIXMAP_CACHE.get(cache_key)
    if pix is not None:
        return pix
    fig = Figure(figsize=(0.22, 0.22), dpi=100)
    FigureCanvasAgg(fig)  # 离屏 Agg canvas，不切全局 backend
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    cmap = get_cmap("jet")
    norm = Normalize(vmin=THICKNESS_MIN, vmax=THICKNESS_MAX)
    ax.scatter([0], [0], s=260, c=[cmap(norm(thickness_km))], marker=marker,
               edgecolors=edge_color, linewidths=line_width)
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", transparent=True)
    fig.clear()
    buf.seek(0)
    pix = QPixmap()
    pix.loadFromData(buf.read(), "PNG")
    _MAP_PIXMAP_CACHE[cache_key] = pix
    return pix


class ThicknessTableModel(QAbstractTableModel):
    def __init__(self, points: list[ThicknessPoint] | None = None):
        super().__init__()
        self._points: list[ThicknessPoint] = []
        self._outlier: dict[str, float] = {}
        self._status: dict[str, str] = {}  # group_key -> status
        self._z: dict[str, float] = {}     # group_key -> |z|
        if points:
            self.set_points(points, {})

    def set_points(self, points: list[ThicknessPoint], outlier: dict[str, float],
                   event_keys_order: list[str] | None = None) -> None:
        self.beginResetModel()
        self._points = list(points)
        self._outlier = dict(outlier)
        self._z = {p.group_key: float(abs(self._outlier.get(p.group_key, 0.0))) for p in self._points}
        self._status = {p.group_key: "pending" for p in self._points}
        # marker 映射基于全量事件顺序（与地图一致），子集筛选后形状不变
        self._event_marker = event_marker_map(event_keys_order or [])
        self.endResetModel()

    def set_status(self, group_key: str, status: str) -> None:
        self._status[group_key] = status
        row = self.row_for_group_key(group_key)
        if row is not None:
            idx = self.index(row, len(COLUMNS) - 1)
            self.dataChanged.emit(idx, idx)

    def points(self) -> list[ThicknessPoint]:
        return list(self._points)

    def point(self, row: int) -> ThicknessPoint | None:
        if 0 <= row < len(self._points):
            return self._points[row]
        return None

    def group_key_for_row(self, row: int) -> str | None:
        p = self.point(row)
        return p.group_key if p else None

    def row_for_group_key(self, group_key: str) -> int | None:
        for i, p in enumerate(self._points):
            if p.group_key == group_key:
                return i
        return None

    def status_for(self, group_key: str) -> str:
        return self._status.get(group_key, "pending")

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        if column < 0 or column >= len(COLUMNS):
            return
        key, _ = COLUMNS[column]
        self.layoutAboutToBeChanged.emit()
        def sort_value(p: ThicknessPoint):
            if key == "outlier_z":
                return self._z.get(p.group_key, 0.0)
            if key == "status":
                return self._status.get(p.group_key, "pending")
            val = getattr(p, key, None)
            return ("" if val is None else val)
        try:
            self._points.sort(
                key=sort_value,
                reverse=(order == Qt.SortOrder.DescendingOrder),
            )
        except TypeError:
            self._points.sort(
                key=lambda p: str(sort_value(p)),
                reverse=(order == Qt.SortOrder.DescendingOrder),
            )
        self.layoutChanged.emit()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._points)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(COLUMNS):
            return COLUMNS[section][1]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        p = self._points[index.row()]
        key, _ = COLUMNS[index.column()]
        status = self._status.get(p.group_key, "pending")
        if key == "map_color":
            # Map 列：该 group 的符号+厚度色图标，与穿透点图一致便于对照
            if role == Qt.ItemDataRole.DisplayRole:
                return ""
            if role == Qt.ItemDataRole.DecorationRole:
                marker = self._event_marker.get(p.event_key, "o")
                try:
                    return _map_color_pixmap(marker, p.thickness_km)
                except Exception:
                    return None
            if role == Qt.ItemDataRole.ToolTipRole:
                z = self._z.get(p.group_key, 0.0)
                suffix = "  [outlier]" if z >= OUTLIER_THRESHOLD else ""
                return f"{p.event_key} / {p.group_name}  thick={p.thickness_km:.2f} km{suffix}"
            return None
        if key == "status":
            # Status 列：彩色圆圈（DecorationRole），文字留空，悬停显示文字。
            if role == Qt.ItemDataRole.DisplayRole:
                return ""
            if role == Qt.ItemDataRole.DecorationRole:
                return _status_pixmap(status)
            if role == Qt.ItemDataRole.ToolTipRole:
                return f"{status} — {STATUS_LABELS.get(status, status)}"
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignCenter)
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            if key == "outlier_z":
                return f"{self._z.get(p.group_key, 0.0):.2f}"
            val = getattr(p, key, None)
            if val is None:
                return ""
            if isinstance(val, float):
                return f"{val:.2f}" if key in ("gcarc", "thickness_km") else f"{val:.1f}"
            return str(val)
        if role == Qt.ItemDataRole.UserRole:
            return p.group_key
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignCenter)
        return None


# ---------------------------------------------------------------------------
# 底图渲染
# ---------------------------------------------------------------------------

def _compute_region(points: list[ThicknessPoint]) -> str:
    if not points:
        return DEFAULT_REGION
    lons = [p.longitude for p in points] + [p.event_lon for p in points]
    lats = [p.latitude for p in points] + [p.event_lat for p in points]
    lon0, lon1 = min(lons), max(lons)
    lat0, lat1 = min(lats), max(lats)
    pad_lon = max(0.5, (lon1 - lon0) * 0.08)
    pad_lat = max(0.5, (lat1 - lat0) * 0.08)
    # 扩大到固定下界：经度至少到 -32°，纬度至少到 -62°，覆盖南桑威奇全貌。
    lon0 = min(lon0 - pad_lon, -32.0)
    lat0 = min(lat0 - pad_lat, -62.0)
    lon1 = lon1 + pad_lon
    lat1 = lat1 + pad_lat
    return f"{lon0:.2f}/{lon1:.2f}/{lat0:.2f}/{lat1:.2f}"


def render_basemap(region: str, stamp: Path) -> Path | None:
    """用 plot_standard_pierce_map.sh 渲染空白底图 PNG（无事件红星，多事件）。"""
    if not BASEMAP_HELPER.exists():
        return None
    stamp.parent.mkdir(parents=True, exist_ok=True)
    empty = stamp.parent / f"{stamp.name}.empty.txt"
    empty.write_text("", encoding="utf-8")
    env = dict(**os.environ)
    env["INPUT_FILE"] = str(empty)
    env["OUTPUT_PREFIX"] = str(stamp)
    env["REGION"] = region
    env.pop("EVENT_LON", None)
    env.pop("EVENT_LAT", None)
    try:
        subprocess.run(["bash", str(BASEMAP_HELPER)], check=True, cwd=str(DPK_ROOT), env=env)
    except Exception:
        return None
    png = stamp.with_suffix(".png")
    return png if png.exists() else None


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class ThicknessReviewWindow(QMainWindow):
    def __init__(self, context: ThicknessReviewContext, parent=None):
        super().__init__(parent)
        self.context = context
        self.scan_root = Path(context.scan_root).expanduser().resolve()
        self.index: ThicknessIndex = ThicknessIndex()
        self.region: str = DEFAULT_REGION
        self._basemap_png: Path | None = None
        self._outlier: dict[str, float] = {}
        self._selection: list[str] = []  # 有序 group_keys，首个为主选
        # matplotlib 艺术家
        self._map_scatter = None
        self._map_highlight = None
        self._map_member_scatter = None
        self._map_stars = None
        self._cb = None  # colorbar，重绘前 remove 避免累积
        # 预览 LRU 缓存：(group_key, mode) -> PreviewBundle
        self._preview_cache: dict[tuple, object] = {}
        # group 级筛选：可见 group_key 集合（None=全部可见）
        self._visible_groups: set[str] | None = None
        self._filter_syncing = False
        self._map_view_bounds: tuple[float, float, float, float] | None = None
        self._refresh_pending = False
        self._rebuild_running = False
        self._live_refresh_timer = QTimer(self)
        self._live_refresh_timer.setInterval(1500)
        self._live_refresh_timer.timeout.connect(self._poll_refresh_timer)
        self.setWindowTitle("Stack 厚度审阅")
        self.resize(1500, 950)
        self._geom_set = False
        self._build_ui()
        self._define_shortcuts()
        self._rebuild_index()
        self._live_refresh_timer.start()

    # ---- UI ----

    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        # 工具条
        bar = QHBoxLayout()
        self.dir_btn = QPushButton("选目录…")
        self.dir_btn.clicked.connect(self._choose_dir)
        bar.addWidget(self.dir_btn)
        self.dir_lbl = QLabel(str(self.scan_root))
        self.dir_lbl.setStyleSheet("color:#444;")
        bar.addWidget(self.dir_lbl, 1)

        bar.addWidget(QLabel("预览模式"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["TopDist", "Overlay"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        bar.addWidget(self.mode_combo)

        self.outlier_chk = QCheckBox("异常描边")
        self.outlier_chk.setChecked(True)
        self.outlier_chk.toggled.connect(self._redraw_map)
        bar.addWidget(self.outlier_chk)

        self.member_chk = QCheckBox("成员下垫")
        self.member_chk.setChecked(False)
        self.member_chk.toggled.connect(self._redraw_map)
        bar.addWidget(self.member_chk)

        self.rebuild_btn = QPushButton("重建索引")
        self.rebuild_btn.clicked.connect(self._rebuild_index)
        bar.addWidget(self.rebuild_btn)

        self.status_lbl = QLabel("就绪")
        bar.addWidget(self.status_lbl)
        root.addLayout(bar)

        # 三视图
        h = QHBoxLayout()
        # 左：预览
        self.preview_fig = Figure(figsize=(7, 9), dpi=100)
        self.preview_ax = self.preview_fig.add_subplot(111)
        self._draw_preview_placeholder()
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.preview_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        h.addWidget(self.preview_canvas, 1)

        # 右：上地图 + 下表格
        right = QVBoxLayout()

        # 事件/group 筛选面板：两级勾选树（事件→group），勾选即时刷新地图+表格
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("筛选:"))
        self.event_filter = QTreeWidget()
        self.event_filter.setHeaderHidden(True)
        self.event_filter.setMaximumHeight(110)
        self.event_filter.itemChanged.connect(self._on_filter_item_changed)
        filter_row.addWidget(self.event_filter, 1)
        btn_col = QVBoxLayout()
        self.btn_all = QPushButton("全选")
        self.btn_all.clicked.connect(lambda: self._set_all_groups(True))
        self.btn_none = QPushButton("全不选")
        self.btn_none.clicked.connect(lambda: self._set_all_groups(False))
        self.btn_expand = QPushButton("展开")
        self.btn_expand.setCheckable(True)
        self.btn_expand.clicked.connect(self._toggle_expand)
        btn_col.addWidget(self.btn_all)
        btn_col.addWidget(self.btn_none)
        btn_col.addWidget(self.btn_expand)
        filter_row.addLayout(btn_col)
        self.hide_event_center_chk = QCheckBox("隐藏事件中心")
        self.hide_event_center_chk.setChecked(True)
        self.hide_event_center_chk.toggled.connect(self._redraw_map)
        filter_row.addWidget(self.hide_event_center_chk)
        filter_row.addWidget(QLabel("地图"))
        self.zoom_in_btn = QPushButton("放大")
        self.zoom_in_btn.clicked.connect(lambda: self._apply_map_zoom(0.6))
        filter_row.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QPushButton("缩小")
        self.zoom_out_btn.clicked.connect(lambda: self._apply_map_zoom(1.6))
        filter_row.addWidget(self.zoom_out_btn)
        self.zoom_reset_btn = QPushButton("重置")
        self.zoom_reset_btn.clicked.connect(self._reset_map_zoom)
        filter_row.addWidget(self.zoom_reset_btn)
        right.addLayout(filter_row)

        self.map_fig = Figure(figsize=(8, 7), dpi=100)
        self.map_ax = self.map_fig.add_axes([0.02, 0.10, 0.96, 0.86])
        self.map_canvas = FigureCanvas(self.map_fig)
        self.map_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.map_canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        right.addWidget(self.map_canvas, 3)

        self.table_model = ThicknessTableModel()
        self.table = QTableView()
        self.table.setModel(self.table_model)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableView.SelectionMode.ExtendedSelection)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(0, 200)   # Event
        self.table.setColumnWidth(1, 70)    # Group
        self.table.setColumnWidth(2, 38)    # Map (符号+厚度色图标)
        # Status 列（最后一列）缩窄：圆圈不需要宽列
        self.table.horizontalHeader().setStretchLastSection(False)
        status_col = len(COLUMNS) - 1
        self.table.setColumnWidth(status_col, 44)
        self.table.setMinimumHeight(140)
        self.table.selectionModel().selectionChanged.connect(self._on_table_selection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_table_context)
        right.addWidget(self.table, 1)
        h.addLayout(right, 2)
        root.addLayout(h, 1)

        self.setCentralWidget(central)
        self.statusBar().showMessage("就绪")
        # 地图点击事件：左键选最近点，右键选最近点+菜单（不用 pick_event，
        # pickradius 会把附近同符号的点一起选中）。
        self.map_canvas.mpl_connect("button_press_event", self._on_map_click)
        self._syncing = False  # 防止表格/地图选中递归

    def _define_shortcuts(self):
        # 与主系统 / DSM 对话框一致：空格 / Esc 关闭窗口。
        for seq in (Qt.Key.Key_Space, Qt.Key.Key_Escape):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(self.close)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not getattr(self, "_geom_set", False):
            # 与 DePhaseKit 主拾取窗一致：最大化铺满工作区（WSLg 回退居中近全屏）。
            maximize_on_workarea(self, frac=0.98)
            self._geom_set = True

    def changeEvent(self, event):  # noqa: N802
        # 窗口重新激活（从主 ppk / 叠加子系统切回）时，检查 sidecar/members
        # 是否变化，变了才刷新表格+地图。覆盖主 ppk 改拾取、叠加子系统重跑 stack。
        if event.type() == QEvent.Type.WindowActivate:
            self._maybe_refresh_on_focus()
        super().changeEvent(event)

    def _maybe_refresh_on_focus(self):
        """获得焦点时若有变化则增量刷新；无变化则跳过（不耗时）。"""
        import time
        now = time.monotonic()
        # 去抖：1 秒内不重复检查
        if getattr(self, "_last_focus_refresh", 0) and now - self._last_focus_refresh < 1.0:
            return
        self._last_focus_refresh = now
        try:
            from stack_group_thickness_index import _cache_fingerprint
            new_fp = _cache_fingerprint(self.scan_root)
        except Exception:
            return
        old_fp = getattr(self, "_last_fingerprint", None)
        if old_fp is not None and old_fp == new_fp:
            return  # 无变化
        self._last_fingerprint = new_fp
        self._schedule_rebuild()

    def _poll_refresh_timer(self):
        if not self.isVisible():
            return
        self._maybe_refresh_on_focus()

    def request_external_refresh(self):
        """供主 ppk / stack 子系统在写入结果后主动触发刷新。"""
        self._last_focus_refresh = 0
        self._schedule_rebuild(delay_ms=80)

    def _schedule_rebuild(self, delay_ms: int = 0):
        if self._rebuild_running:
            self._refresh_pending = True
            return
        if self._refresh_pending:
            return
        self._refresh_pending = True
        QTimer.singleShot(max(0, int(delay_ms)), self._run_scheduled_rebuild)

    def _run_scheduled_rebuild(self):
        if self._rebuild_running or not self._refresh_pending:
            return
        self._refresh_pending = False
        self._rebuild_index()

    # ---- 索引 ----

    def _rebuild_index(self):
        if self._rebuild_running:
            self._refresh_pending = True
            return
        self._rebuild_running = True
        previous_selection = list(getattr(self, "_selection", []))
        self.statusBar().showMessage("构建索引中…")
        QApplication.processEvents()
        try:
            self.index = build_thickness_index(
                self.scan_root,
                pierce_depth_km=self.context.pierce_depth_km,
                model=self.context.model,
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"索引构建失败: {exc}", 8000)
            return
        self._outlier = compute_outlier_score(self.index.points)
        # 载入审阅标记
        marks_by_event: dict[str, dict] = {}
        for p in self.index.points:
            if p.source_event_dir and p.source_event_dir not in marks_by_event:
                marks_by_event[p.source_event_dir] = load_review_marks(p.source_event_dir)
        self.table_model.set_points(
            self.index.points, self._outlier,
            event_keys_order=list(self.index.event_colors.keys()))
        for p in self.index.points:
            entry = marks_by_event.get(p.source_event_dir, {}).get(p.group_name)
            if entry:
                self.table_model.set_status(p.group_key, entry.get("status", "pending"))
        self.region = _compute_region(self.index.points)
        self._map_view_bounds = None
        self._basemap_png = None
        self._populate_event_filter()
        self._redraw_map()
        # 保存当前指纹，供 focusInEvent 增量判断
        try:
            from stack_group_thickness_index import _cache_fingerprint
            self._last_fingerprint = _cache_fingerprint(self.scan_root)
        except Exception:
            self._last_fingerprint = None
        n = len(self.index.points)
        ne = len(self.index.event_colors)
        self.statusBar().showMessage(f"{n} 个 group / {ne} 个事件  (region {self.region})", 6000)
        self.status_lbl.setText(f"{n} groups / {ne} events")
        if previous_selection:
            remaining = [gk for gk in previous_selection if gk in self.index.group_key_to_point()]
            if remaining:
                self._set_selection(remaining)
        self._rebuild_running = False
        if self._refresh_pending:
            self._schedule_rebuild(delay_ms=120)

    def _choose_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择 stack analysis 目录", str(self.scan_root))
        if d:
            self.scan_root = Path(d).resolve()
            self.dir_lbl.setText(str(self.scan_root))
            self._rebuild_index()

    # ---- 事件筛选 ----

    def _populate_event_filter(self):
        """重建事件→group 两级勾选树，默认全选。"""
        self._filter_syncing = True
        try:
            self.event_filter.clear()
            # 按 event_key 聚合 group
            by_event: dict[str, list[ThicknessPoint]] = {}
            for p in self.index.points:
                by_event.setdefault(p.event_key, []).append(p)
            for i, ek in enumerate(self.index.event_colors.keys(), start=1):
                ev_item = QTreeWidgetItem(self.event_filter, [f"#{i}  {ek}"])
                ev_item.setData(0, Qt.ItemDataRole.UserRole, ("event", ek))
                ev_item.setCheckState(0, Qt.CheckState.Checked)
                for p in by_event.get(ek, []):
                    g_item = QTreeWidgetItem(ev_item, [f"  {p.group_name}  ({p.pair_kind})"])
                    g_item.setData(0, Qt.ItemDataRole.UserRole, ("group", p.group_key))
                    g_item.setCheckState(0, Qt.CheckState.Checked)
            self._visible_groups = None  # None = 全部可见
            # 默认全部折叠；按"展开"按钮才展开
            self.event_filter.collapseAll()
            if getattr(self, "btn_expand", None) is not None:
                self.btn_expand.setChecked(False)
                self.btn_expand.setText("展开")
        finally:
            self._filter_syncing = False

    def _collect_visible_group_keys(self) -> set[str] | None:
        """从树勾选状态收集可见 group_key 集合；全选返回 None。"""
        checked: set[str] = set()
        total = 0
        for i in range(self.event_filter.topLevelItemCount()):
            ev_item = self.event_filter.topLevelItem(i)
            for j in range(ev_item.childCount()):
                g_item = ev_item.child(j)
                total += 1
                _, gk = g_item.data(0, Qt.ItemDataRole.UserRole)
                if g_item.checkState(0) == Qt.CheckState.Checked:
                    checked.add(gk)
        if total and len(checked) == total:
            return None
        return checked

    def _selected_visible_points(self) -> list[ThicknessPoint]:
        """按 group 勾选筛选后的穿透点子集。"""
        vis = self._visible_groups
        if vis is None:
            return list(self.index.points)
        return [p for p in self.index.points if p.group_key in vis]

    def _on_filter_item_changed(self, item):
        if self._filter_syncing:
            return
        kind, key = item.data(0, Qt.ItemDataRole.UserRole)
        if kind == "event":
            # 事件勾选联动其下所有 group
            state = item.checkState(0)
            self._filter_syncing = True
            try:
                for j in range(item.childCount()):
                    item.child(j).setCheckState(0, state)
            finally:
                self._filter_syncing = False
        self._visible_groups = self._collect_visible_group_keys()
        self._apply_filter()

    def _apply_filter(self):
        """筛选变化后刷新表格（只显示可见 group）+ 地图。"""
        vis_points = self._selected_visible_points()
        # 保留审阅标记
        old_status = dict(self.table_model._status) if hasattr(self.table_model, "_status") else {}
        self.table_model.set_points(
            vis_points, self._outlier,
            event_keys_order=list(self.index.event_colors.keys()))
        # 恢复状态标记
        for p in vis_points:
            if p.group_key in old_status:
                self.table_model.set_status(p.group_key, old_status[p.group_key])
        self._redraw_map()

    def _set_all_groups(self, checked: bool):
        self._filter_syncing = True
        try:
            state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
            it = QTreeWidgetItemIterator(self.event_filter)
            while it.value() is not None:
                it.value().setCheckState(0, state)
                it += 1
        finally:
            self._filter_syncing = False
        self._visible_groups = None if checked else set()
        self._apply_filter()

    def _toggle_expand(self):
        """切换展开/折叠所有事件；按钮文字随之变化。"""
        if self.btn_expand.isChecked():
            self.event_filter.expandAll()
            self.btn_expand.setText("折叠")
        else:
            self.event_filter.collapseAll()
            self.btn_expand.setText("展开")

    def _map_default_bounds(self) -> tuple[float, float, float, float]:
        lon0, lon1, lat0, lat1 = (float(v) for v in self.region.split("/"))
        return lon0, lon1, lat0, lat1

    def _map_bounds(self) -> tuple[float, float, float, float]:
        return self._map_view_bounds or self._map_default_bounds()

    def _primary_selected_point(self) -> ThicknessPoint | None:
        if not self._selection:
            return None
        primary = self._selection[0]
        return next((p for p in self.index.points if p.group_key == primary), None)

    def _apply_map_zoom(self, factor: float):
        if factor <= 0:
            return
        lon0, lon1, lat0, lat1 = self._map_default_bounds()
        default_lon_span = lon1 - lon0
        default_lat_span = lat1 - lat0
        cur_lon0, cur_lon1, cur_lat0, cur_lat1 = self._map_bounds()
        cur_lon_span = cur_lon1 - cur_lon0
        cur_lat_span = cur_lat1 - cur_lat0
        new_lon_span = min(default_lon_span, max(default_lon_span * 0.08, cur_lon_span * factor))
        new_lat_span = min(default_lat_span, max(default_lat_span * 0.08, cur_lat_span * factor))
        focus = self._primary_selected_point()
        center_lon = float(focus.longitude) if focus is not None else (cur_lon0 + cur_lon1) / 2.0
        center_lat = float(focus.latitude) if focus is not None else (cur_lat0 + cur_lat1) / 2.0
        new_lon0 = center_lon - new_lon_span / 2.0
        new_lon1 = center_lon + new_lon_span / 2.0
        new_lat0 = center_lat - new_lat_span / 2.0
        new_lat1 = center_lat + new_lat_span / 2.0
        if new_lon0 < lon0:
            shift = lon0 - new_lon0
            new_lon0 += shift
            new_lon1 += shift
        if new_lon1 > lon1:
            shift = new_lon1 - lon1
            new_lon0 -= shift
            new_lon1 -= shift
        if new_lat0 < lat0:
            shift = lat0 - new_lat0
            new_lat0 += shift
            new_lat1 += shift
        if new_lat1 > lat1:
            shift = new_lat1 - lat1
            new_lat0 -= shift
            new_lat1 -= shift
        self._map_view_bounds = (new_lon0, new_lon1, new_lat0, new_lat1)
        self._redraw_map()

    def _reset_map_zoom(self):
        self._map_view_bounds = None
        self._redraw_map()

    # ---- 地图 ----

    def _ensure_basemap(self) -> Path | None:
        if self._basemap_png is not None and self._basemap_png.exists():
            return self._basemap_png
        # 用 region 的稳定哈希（sha1）做 stamp 文件名——Python 内置 hash() 每进程
        # 随机化（PYTHONHASHSEED），会导致每次启动生成新 PNG、旧的不被复用，累积冗余。
        import hashlib
        digest = hashlib.sha1(self.region.encode("utf-8")).hexdigest()[:16]
        stamp = Path(tempfile.gettempdir()) / f"thickness_review_basemap_{digest}"
        png = stamp.with_suffix(".png")
        # 已存在且非空则直接复用，不重跑 GMT
        if png.exists() and png.stat().st_size > 0:
            self._basemap_png = png
            return png
        png = render_basemap(self.region, stamp)
        self._basemap_png = png
        return png

    def _redraw_map(self):
        ax = self.map_ax
        ax.clear()
        points = self._selected_visible_points()
        lon0, lon1, lat0, lat1 = self._map_default_bounds()
        view_lon0, view_lon1, view_lat0, view_lat1 = self._map_bounds()
        basemap = self._ensure_basemap()
        if basemap is not None:
            import matplotlib.image as mpimg
            try:
                img = mpimg.imread(basemap)
                ax.imshow(img, extent=[lon0, lon1, lat0, lat1], origin="upper", aspect="auto", zorder=0)
            except Exception:
                pass
        ax.set_xlim(view_lon0, view_lon1)
        ax.set_ylim(view_lat0, view_lat1)
        # 让 axes 匹配 region 的经纬度比，底图不被 canvas 纵横比拉变形；
        # adjustable="datalim" 让 matplotlib 调整数据范围以容纳固定比例，
        # 散点与底图都用经纬度坐标，仍精确对齐。
        if (lat1 - lat0) > 0:
            ax.set_aspect((lon1 - lon0) / (lat1 - lat0), adjustable="datalim")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if not points:
            self.map_canvas.draw_idle()
            return

        norm = Normalize(vmin=THICKNESS_MIN, vmax=THICKNESS_MAX)
        cmap = get_cmap("jet")

        # 事件 → 形状 映射：统一走 event_marker_map()，这样默认绘制 / 点击高亮 /
        # 表格 Map 列都使用同一套年份覆盖逻辑，不会出现“初始是闪电，点一下变梯形”。
        event_keys_order = list(self.index.event_colors.keys())
        event_marker = event_marker_map(event_keys_order)

        # 按形状分组绘制穿透点（scatter 一次一个 marker）。
        # 每个点：厚度色填充 + 同事件形状。异常提示只留在列表，不混进地图颜色语义。
        self._map_scatch_group_keys = []  # 与各 scatter 的点顺序对齐，供 pick
        by_marker: dict[str, list[tuple[float, float, tuple, str, float, str]]] = {}
        for p in points:
            mk = event_marker.get(p.event_key, "o")
            color = cmap(norm(p.thickness_km))
            edge = "black"
            lw = 0.4
            by_marker.setdefault(mk, []).append(
                (p.longitude, p.latitude, color, edge, lw, p.group_key)
            )

        self._map_scatter = None
        self._map_scatter_artists: list = []
        for mk, items in by_marker.items():
            xs = [it[0] for it in items]
            ys = [it[1] for it in items]
            cs = [it[2] for it in items]
            edges = [it[3] for it in items]
            lws = [it[4] for it in items]
            keys = [it[5] for it in items]
            sc = ax.scatter(
                xs, ys, s=PIERCE_POINT_SIZE, c=cs, marker=mk,
                edgecolors=edges, linewidths=lws,
                zorder=5,
            )
            sc._dpk_group_keys = keys  # pick 时按局部 ind 取
            self._map_scatter_artists.append(sc)
            if self._map_scatter is None:
                self._map_scatter = sc

        # 事件中心：灰色 + 大尺寸同形状，白描边，标注事件序号（可隐藏）
        seen: set[str] = set()
        self._event_index: dict[str, int] = {}  # event_key -> 序号(1-based)
        hide_center = self.hide_event_center_chk.isChecked()
        for p in points:
            if p.event_key in seen:
                continue
            seen.add(p.event_key)
            self._event_index[p.event_key] = len(self._event_index) + 1
            if hide_center:
                continue
            mk = event_marker.get(p.event_key, "o")
            ax.scatter(
                [p.event_lon], [p.event_lat],
                marker=mk, s=EVENT_CENTER_SIZE,
                facecolors="#555555", edgecolors="white", linewidths=1.2,
                alpha=0.85, zorder=6,
            )
            # 序号标注
            ax.text(
                p.event_lon, p.event_lat, str(self._event_index[p.event_key]),
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white", zorder=7,
            )

        # colorbar：用独立 cax 固定位置，不绑主 ax（避免 colorbar 侵占主 axes 空间
        # 把地图挤小）；重绘前 remove 旧的避免累积。
        if getattr(self, "_cb", None) is not None:
            try:
                self._cb.remove()
            except Exception:
                pass
            self._cb = None
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cax = self.map_fig.add_axes([0.25, 0.035, 0.50, 0.022])
        self._cb = self.map_fig.colorbar(sm, cax=cax, orientation="horizontal")
        self._cb.set_label("Crustal Thickness (km)")

        self._draw_highlight()
        self.map_canvas.draw_idle()

    def _draw_highlight(self):
        """在地图上高亮当前选中：符号自身发光（path_effects 光晕描边）。"""
        for attr in ("_map_highlight", "_map_member_scatter"):
            art = getattr(self, attr, None)
            if art is not None:
                for a in (art if isinstance(art, list) else [art]):
                    try:
                        a.remove()
                    except Exception:
                        pass
                setattr(self, attr, None)
        if not self._selection:
            self.map_canvas.draw_idle()
            return
        sel = self._selection
        pts = [p for p in self.index.points if p.group_key in sel]
        if not pts:
            self.map_canvas.draw_idle()
            return
        import matplotlib.patheffects as pe
        from matplotlib.cm import get_cmap as _get_cmap
        from matplotlib.colors import Normalize as _Norm
        norm = _Norm(vmin=THICKNESS_MIN, vmax=THICKNESS_MAX)
        cmap = _get_cmap("jet")
        event_keys_order = list(self.index.event_colors.keys())
        emap = event_marker_map(event_keys_order)
        primary = pts[0].group_key
        # 选中点：同 marker + 厚度色 + 发光描边（符号自身发光，不画外光圈）
        glow_artists = []
        for p in pts:
            mk = emap.get(p.event_key, "o")
            color = cmap(norm(p.thickness_km))
            is_primary = (p.group_key == primary)
            sc = self.map_ax.scatter(
                [p.longitude], [p.latitude],
                s=PIERCE_POINT_SIZE * (1.25 if is_primary else 1.0),
                c=[color], marker=mk,
                edgecolors="white", linewidths=1.4,
                zorder=8,
            )
            # 发光：亮黄半透明粗描边 + 白色内描边
            sc.set_path_effects([
                pe.withStroke(linewidth=5.0, foreground="#fff200", alpha=0.55),
                pe.withStroke(linewidth=2.0, foreground="white", alpha=0.9),
            ])
            glow_artists.append(sc)
        # 存为列表，便于 remove
        self._map_highlight = glow_artists
        # 成员穿透点下垫（仅主选，避免过密）
        if self.member_chk.isChecked():
            primary_pt = next((p for p in pts if p.group_key == primary), None)
            if primary_pt is not None:
                member_pts = load_member_pierce_points(primary_pt)
                if member_pts:
                    self._map_member_scatter = self.map_ax.scatter(
                        [lon for lon, _ in member_pts],
                        [lat for _, lat in member_pts],
                        s=18, c="#ff7f0e", alpha=0.55, edgecolors="none", zorder=4,
                    )
        self.map_canvas.draw_idle()

    # ---- 选中联动 ----

    def _set_selection(self, group_keys, *, extend: bool = False, toggle: bool = False):
        """设置选中集合并联动三视图。group_keys 可为单值或可迭代。"""
        if isinstance(group_keys, str):
            group_keys = [group_keys]
        group_keys = [g for g in group_keys if g]
        if not group_keys:
            self._selection = []
            self._sync_table_selection([])
            self._draw_highlight()
            self._refresh_preview()
            return
        if extend:
            # 追加到已有选择
            existing = list(self._selection)
            for g in group_keys:
                if g not in existing:
                    existing.append(g)
            self._selection = existing
        elif toggle:
            existing = list(self._selection)
            for g in group_keys:
                if g in existing:
                    existing.remove(g)
                else:
                    existing.append(g)
            self._selection = existing
        else:
            self._selection = list(group_keys)
        self._sync_table_selection(self._selection)
        self._draw_highlight()
        self._refresh_preview()

    def _sync_table_selection(self, group_keys):
        """程序化选中表格行（不触发递归）。

        选中后滚动使主选行可见——点地图穿透点直接跳到对应行，不用上下翻。
        表格开了 ``setSortingEnabled``，视图行号与模型行号不一致，故用
        ``model.index(row, 0)`` 让 QTableView 自己映射到当前排序后的视图行。
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            sm = self.table.selectionModel()
            sm.clear()
            flags = QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows
            for gk in group_keys:
                row = self.table_model.row_for_group_key(gk)
                if row is not None:
                    sm.select(self.table_model.index(row, 0), flags)
            # 滚动到首个选中行（鼠标在地图上点的那个），使其在视口内可见。
            if group_keys:
                first_row = self.table_model.row_for_group_key(group_keys[0])
                if first_row is not None:
                    idx = self.table_model.index(first_row, 0)
                    self.table.scrollTo(idx, QAbstractItemView.ScrollHint.PositionAtCenter)
                    self.table.setCurrentIndex(idx)
        finally:
            self._syncing = False

    def _on_table_selection(self):
        if self._syncing:
            return
        rows = {idx.row() for idx in self.table.selectionModel().selectedRows()}
        group_keys = [self.table_model.group_key_for_row(r) for r in sorted(rows)]
        group_keys = [g for g in group_keys if g]
        self._selection = list(group_keys)
        self._draw_highlight()
        self._refresh_preview()

    def _on_map_click(self, event):
        """左键选最近穿透点；右键选最近点并弹上下文菜单。"""
        if event.inaxes is not self.map_ax or event.xdata is None:
            return
        # 在当前可见点里找最近点（按数据坐标欧氏距离），只取一个，
        # 避免 pickradius 把附近同符号的点一起选中。
        points = self._selected_visible_points()
        if not points:
            return
        best = None
        best_d = 1e9
        for p in points:
            d = (p.longitude - event.xdata) ** 2 + (p.latitude - event.ydata) ** 2
            if d < best_d:
                best_d = d
                best = p
        if best is None:
            return
        # 距离阈值（经纬度平方）：太远视为没点中，避免误选。
        if best_d > 0.15:
            return
        key = event.key if event.key else None
        extend = bool(key and "shift" in key)
        toggle = bool(key and ("control" in key or "ctrl" in key))
        self._set_selection([best.group_key], extend=extend, toggle=toggle)
        if event.button == 3:
            self._popup_context_for(best.group_key, QCursor.pos())

    # ---- 上下文菜单 + 审阅标记 ----

    def _on_table_context(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return
        group_key = self.table_model.group_key_for_row(index.row())
        if not group_key:
            return
        global_pos = self.table.viewport().mapToGlobal(pos)
        self._popup_context_for(group_key, global_pos)

    def _popup_context_for(self, group_key: str, global_pos):
        point = next((p for p in self.index.points if p.group_key == group_key), None)
        if point is None:
            return
        menu = QMenu(self)
        title = QAction(f"{point.event_key} / {point.group_name}", menu)
        title.setEnabled(False)
        menu.addAction(title)
        menu.addSeparator()
        status_menu = menu.addMenu("标记状态")
        cur = self.table_model.status_for(group_key)
        for status, label in (
            ("pending", "待查"),
            ("suspect", "可疑"),
            ("fixed", "已修"),
            ("ignore", "忽略"),
        ):
            act = QAction(("✓ " if status == cur else "   ") + label, status_menu)
            act.triggered.connect(lambda _=False, s=status, gk=group_key, p=point: self._set_status(gk, p, s))
            status_menu.addAction(act)
        menu.addSeparator()
        open_ppk = QAction("在主 ppk 打开此事件…", menu)
        open_ppk.triggered.connect(lambda _=False, p=point: self._open_in_ppk(p))
        menu.addAction(open_ppk)
        open_stack = QAction("在叠加子系统打开此 group…", menu)
        open_stack.triggered.connect(lambda _=False, p=point: self._open_in_stack(p))
        menu.addAction(open_stack)
        menu.exec(global_pos)

    def _set_status(self, group_key: str, point: ThicknessPoint, status: str):
        self.table_model.set_status(group_key, status)
        try:
            set_review_mark(point.source_event_dir, point.group_name, status=status)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"标记写入失败: {exc}", 6000)
            return
        self.statusBar().showMessage(
            f"已标记 {point.group_name} = {status}（写入 thickness_review.json）", 4000)

    def _open_in_ppk(self, point: ThicknessPoint):
        cb = self.context.open_event_callback
        if cb is None:
            self.statusBar().showMessage(
                f"未注入 ppk 回调；事件目录: {point.source_event_dir}", 8000)
            return
        try:
            cb(point.source_event_dir, point.group_name)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"打开 ppk 失败: {exc}", 6000)

    def _open_in_stack(self, point: ThicknessPoint):
        """打开该 group 的 stack 工作区拾取窗（主图只显示该 group 的 stack.sac）。"""
        cb = self.context.open_stack_callback
        if cb is None:
            self.statusBar().showMessage(
                f"未注入 stack 回调；事件目录: {point.source_event_dir}", 8000)
            return
        # 从 sidecar 读 stack_wave_name
        import json
        try:
            payload = json.loads(Path(point.sidecar_path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.statusBar().showMessage(f"读取 sidecar 失败: {point.sidecar_path}", 6000)
            return
        stack_wave_name = str(payload.get("stack_wave_name") or "").strip()
        if not stack_wave_name:
            self.statusBar().showMessage("sidecar 缺 stack_wave_name", 6000)
            return
        try:
            cb(point.source_event_dir, stack_wave_name)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"打开 stack 工作区失败: {exc}", 6000)

    # ---- 预览 ----

    def _draw_preview_placeholder(self):
        ax = self.preview_ax
        ax.clear()
        ax.text(0.5, 0.5, "Select a group\nto show its stack preview",
                ha="center", va="center", transform=ax.transAxes, fontsize=13, color="#666")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _refresh_preview(self):
        """按当前选中刷新左侧预览。"""
        if not self._selection:
            self._draw_preview_placeholder()
            self.preview_canvas.draw_idle()
            return
        primary = self._selection[0]
        point = next((p for p in self.index.points if p.group_key == primary), None)
        if point is None:
            self._draw_preview_placeholder()
            self.preview_canvas.draw_idle()
            return
        mode = "top" if self.mode_combo.currentText() == "TopDist" else "overlay"
        cache_key = (primary, mode)
        bundle = self._preview_cache.get(cache_key)
        if bundle is None:
            from stack_thickness_review_core import build_preview_traces
            bundle = build_preview_traces(point, display_mode=mode)
            self._preview_cache[cache_key] = bundle
        self._draw_preview(bundle, point)
        self.preview_canvas.draw_idle()

    def _draw_preview(self, bundle, point: ThicknessPoint):
        ax = self.preview_ax
        ax.clear()
        traces = bundle.traces
        if not traces:
            ax.text(0.5, 0.5, f"No drawable traces\n{bundle.note}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=12, color="#900")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return
        align = bundle.align_marker
        if bundle.display_mode == "overlay":
            # 成员灰，stack 红
            for tr in traces:
                if tr.is_stack:
                    continue
                ax.plot(tr.t_array, tr.y_array, color="#8888cc", linewidth=0.6, alpha=0.7, zorder=2)
            for tr in traces:
                if tr.is_stack:
                    ax.plot(tr.t_array, tr.y_array, color="#d62728", linewidth=1.8, zorder=4)
            ax.axhline(0.0, color="0.7", linewidth=0.5, zorder=1)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel("Normalized amplitude")
        else:  # top
            # 与 WaveFigure.plot_waves 一致：每条道已峰值归一化到 [-1,1]，
            # 乘固定缩放因子 enf（单位=gcarc 度），再加 gcarc 偏移。这样无论
            # 某条原始振幅多大，都只占 enf 高度，不会撑爆 y 轴。
            gcarcs = [tr.gcarc for tr in traces if tr.gcarc == tr.gcarc]
            if gcarcs:
                span = max(gcarcs) - min(gcarcs)
            else:
                span = 1.0
            enf = max(0.4, span * 0.06) if span > 0 else 0.5
            for tr in traces:
                if tr.is_stack:
                    continue
                y = tr.gcarc + tr.y_array * enf
                ax.plot(tr.t_array, y, color="#8888cc", linewidth=0.6, alpha=0.7, zorder=2)
            for tr in traces:
                if tr.is_stack:
                    y = tr.gcarc + tr.y_array * enf
                    ax.plot(tr.t_array, y, color="#d62728", linewidth=1.8, zorder=4)
            ax.set_ylabel("gcarc (deg)")
        ax.set_xlabel(f"Time after {align} (s)")
        title = f"{point.event_key}  {point.group_name}  [{point.pair_kind}]\n"
        title += f"thick={point.thickness_km:.2f} km  N={point.member_count_used}  |z|={abs(self._outlier.get(point.group_key,0.0)):.2f}"
        if not bundle.stack_available:
            title += "  (no stack)"
        ax.set_title(title, fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)

    def _on_mode_changed(self, _text: str):
        self._preview_cache.clear()
        self._refresh_preview()


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Stack 厚度审阅交互窗")
    parser.add_argument("--root", default=str(DEFAULT_SCAN_ROOT), help="stack analysis 目录")
    args = parser.parse_args(argv)
    app = QApplication.instance() or QApplication(sys.argv)
    ctx = ThicknessReviewContext(scan_root=args.root)
    win = ThicknessReviewWindow(ctx)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
