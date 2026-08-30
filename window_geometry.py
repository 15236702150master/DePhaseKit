import os
import re
import subprocess
import tempfile

from PySide6.QtCore import QRect
from PySide6.QtGui import QCursor, QGuiApplication


_WORKAREA_PATTERN = re.compile(r'WA L=(-?\d+) T=(-?\d+) R=(-?\d+) B=(-?\d+)')


def _parse_rect(pattern, output):
    if not output:
        return None
    match = pattern.search(output)
    if match is None:
        return None
    left, top, right, bottom = map(int, match.groups())
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def parse_windows_workarea_output(output):
    return _parse_rect(_WORKAREA_PATTERN, output)


def _screen_geometry(screen):
    if screen is None:
        return None
    try:
        geometry = screen.geometry()
    except Exception:
        return None
    if geometry is None or not geometry.isValid():
        return None
    return geometry


def _screen_at_cursor():
    try:
        return QGuiApplication.screenAt(QCursor.pos())
    except Exception:
        return None


def resolve_qt_screen(widget=None, screen=None, prefer_cursor=False):
    if screen is not None:
        return screen

    candidates = []
    if prefer_cursor:
        candidates.append(_screen_at_cursor)
    if widget is not None:
        candidates.extend([
            lambda: widget.screen(),
            lambda: widget.windowHandle().screen() if widget.windowHandle() is not None else None,
        ])
    if not prefer_cursor:
        candidates.append(_screen_at_cursor)
    candidates.append(QGuiApplication.primaryScreen)

    for getter in candidates:
        try:
            candidate = getter()
        except Exception:
            candidate = None
        if candidate is not None:
            return candidate
    return None


def _windows_monitor_workarea_script(point=None):
    monitor_block = ""
    point_init = ""
    if point is not None:
        point_x, point_y = point
        point_init = (
            "$pt = New-Object WA+POINT\n"
            f"$pt.X = {int(point_x)}\n"
            f"$pt.Y = {int(point_y)}\n"
            "$monitor = [WA]::MonitorFromPoint($pt, [WA]::MONITOR_DEFAULTTONEAREST)\n"
            "if ($monitor -ne [IntPtr]::Zero) {\n"
            "  $mi = New-Object WA+MONITORINFO\n"
            "  $mi.cbSize = [System.Runtime.InteropServices.Marshal]::SizeOf([type] [WA+MONITORINFO])\n"
            "  if ([WA]::GetMonitorInfo($monitor, [ref]$mi)) {\n"
            '    Write-Output ("WA L={0} T={1} R={2} B={3}" -f $mi.rcWork.L, $mi.rcWork.T, $mi.rcWork.R, $mi.rcWork.B)\n'
            "    return\n"
            "  }\n"
            "}\n"
        )
        monitor_block = (
            "[StructLayout(LayoutKind.Sequential)]public struct POINT{public int X,Y;}\n"
            "[StructLayout(LayoutKind.Sequential, CharSet=CharSet.Auto)]public struct MONITORINFO{"
            "public int cbSize;public RECT rcMonitor;public RECT rcWork;public int dwFlags;}\n"
            "[DllImport(\"user32.dll\")]public static extern IntPtr MonitorFromPoint(POINT pt,uint flags);\n"
            "[DllImport(\"user32.dll\",CharSet=CharSet.Auto)]public static extern bool GetMonitorInfo(IntPtr h,ref MONITORINFO i);\n"
            "public const uint MONITOR_DEFAULTTONEAREST=2;\n"
        )

    return (
        'Add-Type @"\n'
        "using System;using System.Runtime.InteropServices;\n"
        "public class WA{[StructLayout(LayoutKind.Sequential)]public struct RECT{public int L,T,R,B;}\n"
        f"{monitor_block}"
        '[DllImport("user32.dll",CharSet=CharSet.Auto)]public static extern bool SystemParametersInfo(int u,int p,ref RECT r,int f);}\n'
        '"@\n'
        f"{point_init}"
        "$r=New-Object WA+RECT;[WA]::SystemParametersInfo(0x0030,0,[ref]$r,0)|Out-Null\n"
        'Write-Output ("WA L={0} T={1} R={2} B={3}" -f $r.L,$r.T,$r.R,$r.B)\n'
    )


def _run_powershell_script(ps_script):
    try:
        with tempfile.NamedTemporaryFile('w', suffix='.ps1', delete=False, encoding='utf-8') as handle:
            handle.write(ps_script)
            script_path = handle.name
        try:
            try:
                windows_path = subprocess.run(
                    ['wslpath', '-w', script_path],
                    capture_output=True,
                    text=True,
                    timeout=2,
                ).stdout.strip()
            except Exception:
                windows_path = script_path
            completed = subprocess.run(
                ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', windows_path],
                capture_output=True,
                text=True,
                timeout=5,
                errors='ignore',
            )
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass
    except Exception:
        return None
    return completed.stdout


def detect_windows_workarea(screen=None):
    geometry = _screen_geometry(screen)
    forced_margin = os.environ.get('DPK_TASKBAR_MARGIN')
    if forced_margin is not None and geometry is not None:
        try:
            margin = int(forced_margin)
        except ValueError:
            margin = None
        if margin is not None:
            return (
                geometry.x(),
                geometry.y(),
                geometry.width(),
                max(200, geometry.height() - margin),
            )

    point = None
    if geometry is not None:
        center = geometry.center()
        point = (center.x(), center.y())

    output = _run_powershell_script(_windows_monitor_workarea_script(point=point))
    return parse_windows_workarea_output(output)


def screen_workarea_rect(widget=None, screen=None, prefer_cursor=False):
    target_screen = resolve_qt_screen(widget=widget, screen=screen, prefer_cursor=prefer_cursor)
    if target_screen is None:
        return None

    windows_workarea = detect_windows_workarea(screen=target_screen)
    if windows_workarea is not None:
        return QRect(*windows_workarea)

    try:
        available = target_screen.availableGeometry()
    except Exception:
        available = None
    if available is not None and available.isValid():
        return available

    return _screen_geometry(target_screen)


def center_widget_on_workarea(widget, frac=0.96):
    """Resize widget to `frac` of the workarea and center it.

    Robust fallback chain (Windows workarea -> screen availableGeometry ->
    screen geometry -> primary screen) so the window still centers when the
    powershell workarea probe fails (common on WSLg). Returns True on success.
    """
    if widget is None:
        return False
    workarea = screen_workarea_rect(widget=widget)
    if workarea is None or not workarea.isValid():
        # last-resort: primary screen geometry
        screen = QGuiApplication.primaryScreen()
        geo = _screen_geometry(screen)
        if geo is None:
            return False
        workarea = QRect(*geo)
    try:
        target_w = max(400, int(workarea.width() * float(frac)))
        target_h = max(300, int(workarea.height() * float(frac)))
        target_x = workarea.x() + max(0, (workarea.width() - target_w) // 2)
        target_y = workarea.y() + max(0, (workarea.height() - target_h) // 2)
        widget.resize(target_w, target_h)
        widget.move(target_x, target_y)
    except Exception:
        return False
    return True


def center_widget_keep_size(widget):
    """Move widget to the center of the workarea without changing its size.

    Used for matplotlib figure windows whose size is driven by figsize; only the
    position needs centering. Returns the target (x, y) or None."""
    if widget is None:
        return None
    workarea = screen_workarea_rect(widget=widget)
    if workarea is None or not workarea.isValid():
        screen = QGuiApplication.primaryScreen()
        geo = _screen_geometry(screen)
        if geo is None:
            return None
        workarea = QRect(*geo)
    try:
        w = max(1, int(widget.width()))
        h = max(1, int(widget.height()))
        target_x = workarea.x() + max(0, (workarea.width() - w) // 2)
        target_y = workarea.y() + max(0, (workarea.height() - h) // 2)
        widget.move(target_x, target_y)
        return (target_x, target_y)
    except Exception:
        return None


def maximize_on_workarea(widget, frac=0.98):
    """与 ppk 主拾取窗 _set_geom_center 一致的打开方式：showMaximized() 最大化铺满
    工作区；若 WM 忽略最大化（WSLg/XWayland 常见），回退到 frac 比例的居中近全屏几何。

    供 dsm 拟合窗/组总览窗复用，使其窗口大小位置与 dephasekit 主拾取窗一致。返回 True 表示
    成功设置了几何。"""
    if widget is None:
        return False
    try:
        widget.showMaximized()
        geo = widget.geometry()
        wa = screen_workarea_rect(widget=widget)
        if wa is not None and wa.isValid():
            # 窗口没真正展开到接近工作区大小 → WM 忽略了 showMaximized，强制居中近全屏。
            if geo.width() < int(wa.width() * 0.9) or geo.height() < int(wa.height() * 0.9):
                center_widget_on_workarea(widget, frac=frac)
                widget.show()
    except Exception:
        return False
    return True


