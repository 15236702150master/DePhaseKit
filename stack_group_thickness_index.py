#!/usr/bin/env python3
"""共享的 stack-group 地壳厚度索引层。

把 ``codes/plot/pierces_with_thickness/plot_stack_group_thickness_overview.py`` 里
``collect_points()`` 的逻辑抽成可复用模块，供静态总览绘图脚本与交互式审阅模块共用，
避免两套口径漂移。

穿透点统一 **24.4 km / prem**（与 ``pierce_point_cache.DEFAULT_PIERCE_DEPTH_KM`` 一致）。

每个 ``ThicknessPoint`` 对应一个合格的 stack group（具备完整 t5+t9 或 t6+t8 配对）：
  - 经纬度 = 该 group 所有 used 成员穿透点的均值
  - 颜色编码 = 由配对时差计算的地壳厚度

审阅标记（``pending``/``suspect``/``fixed``/``ignore``）按事件持久化到 sidecar 同目录的
``thickness_review.json``，键为 ``group_name``。
"""

from __future__ import annotations

import csv
import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from pierce_point_cache import (
    DEFAULT_OUTPUT_ROOT as PIERCE_CACHE_ROOT,
    DEFAULT_PIERCE_DEPTH_KM,
    ensure_pierce_file,
    load_pierce_points,
    pierce_file_path,
)
from forward.constants import DEFAULT_CRUST_VP, DEFAULT_CRUST_VS
from stack_crustal_thickness import (
    DEFAULT_TAUP_BIN,
    calculate_pp_pmp_thickness,
    calculate_sp_smp_thickness,
    fetch_taup_ray_parameter,
)
from stack_system import (
    PROJECT_ROOT,
    STACK_OUTPUT_ROOT,
    STACK_ROOT,
    ensure_stack_storage_ready,
    stack_metadata_dir_for_event,
)

DEFAULT_PIERCE_MODEL = "prem"
REVIEW_STATUS_VALUES = ("pending", "suspect", "fixed", "ignore")
REVIEW_FILE_NAME = "thickness_review.json"
# 缓存放在 scan_root（通常=analysis）之外，避免 glob 误入。
INDEX_CACHE_DIR = STACK_ROOT / "thickness_index"

# 复用 overview 脚本的事件星色板（事件图例用）。
STAR_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]


@dataclass
class ThicknessPoint:
    """一个合格 stack group 的平均穿透点 + 地壳厚度。"""

    dataset: str
    event: str
    event_label: str
    group_name: str
    pair_kind: str          # "t5+t9" 或 "t6+t8"
    phase_kind: str         # "sP" 或 "pP"
    longitude: float
    latitude: float
    thickness_km: float
    event_lon: float
    event_lat: float
    align_marker: str       # 如 "t6"
    member_count_used: int
    gcarc: float
    evdp: float
    result_package_dir: str
    sidecar_path: str
    source_event_dir: str

    @property
    def event_key(self) -> str:
        """与 overview 脚本一致的 `dataset/event` 键。"""
        return f"{self.dataset}/{self.event}" if self.dataset else self.event

    @property
    def group_key(self) -> str:
        """全局唯一键：event_key + group + pair。"""
        return f"{self.event_key}|{self.group_name}|{self.pair_kind}"


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number in (-12345.0, 12345.0):
        return None
    return number


def _pair_rule(markers: dict) -> tuple[str, str] | None:
    """t5+t9 → sP；t6+t8 → pP。与 overview 脚本一致。"""
    t5 = _finite(markers.get("t5"))
    t6 = _finite(markers.get("t6"))
    t8 = _finite(markers.get("t8"))
    t9 = _finite(markers.get("t9"))
    if t5 is not None and t9 is not None:
        return ("t5+t9", "sP")
    if t6 is not None and t8 is not None:
        return ("t6+t8", "pP")
    return None


def _event_label(event: str) -> str:
    if len(event) >= 10:
        return event[:10].replace("_", "-")
    return event


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _result_members_file(payload: dict) -> Path | None:
    result_package_dir = str(payload.get("result_package_dir") or "").strip()
    if not result_package_dir:
        return None
    path = Path(result_package_dir).expanduser().resolve() / "members.txt"
    return path if path.exists() else None


def _used_member_wave_names(members_file: Path) -> list[str]:
    wave_names: list[str] = []
    try:
        with members_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if str(row.get("status") or "").strip().lower() != "used":
                    continue
                wave_name = str(row.get("wave_name") or "").strip()
                if wave_name:
                    wave_names.append(wave_name)
    except OSError:
        return []
    return wave_names


def _dataset_event_from_sidecar(sidecar_path: Path, scan_root: Path) -> tuple[str, str]:
    """从 sidecar 在 scan_root 下的相对位置推 dataset/event 标签。

    scan_root/<dataset>/<event>/stack_*.json  → (dataset, event)
    scan_root/<event>/stack_*.json            → ("", event)
    无法判定时回退用父目录名。
    """
    try:
        rel = sidecar_path.resolve().relative_to(scan_root.resolve())
        parts = rel.parts
    except ValueError:
        parts = sidecar_path.parts
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return "", parts[0]
    return "", sidecar_path.parent.name


# ---------------------------------------------------------------------------
# 核心收集
# ---------------------------------------------------------------------------

def _thickness_for_payload(
    payload: dict,
    pair_kind: str,
    phase_kind: str,
    evdp: float,
    gcarc: float,
    taup_bin: Path,
) -> float | None:
    markers = payload.get("markers", {}) or {}
    t5 = _finite(markers.get("t5"))
    t6 = _finite(markers.get("t6"))
    t8 = _finite(markers.get("t8"))
    t9 = _finite(markers.get("t9"))
    try:
        if phase_kind == "pP" and t6 is not None and t8 is not None:
            ray_param = fetch_taup_ray_parameter(
                taup_bin, evdp_km=evdp, gcarc_deg=gcarc, phase="pP", model="prem"
            )
            return calculate_pp_pmp_thickness(t6 - t8, DEFAULT_CRUST_VP, ray_param)
        if phase_kind == "sP" and t5 is not None and t9 is not None:
            ray_param = fetch_taup_ray_parameter(
                taup_bin, evdp_km=evdp, gcarc_deg=gcarc, phase="sP", model="prem"
            )
            return calculate_sp_smp_thickness(
                t5 - t9, DEFAULT_CRUST_VP, DEFAULT_CRUST_VS, ray_param
            )
    except Exception:
        return None
    return None


def _iter_sidecar_paths(scan_root: Path) -> list[Path]:
    """递归扫 scan_root 下所有 *.stack.json，跳过隐藏/_trash 目录。"""
    if not scan_root.exists():
        return []
    paths: list[Path] = []
    for json_path in sorted(scan_root.rglob("*.stack.json")):
        if not json_path.is_file():
            continue
        if any(part.startswith(".") or part == "_trash_invalid" for part in json_path.parts):
            continue
        paths.append(json_path)
    return paths


def collect_points(
    scan_root: Path,
    *,
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM,
    model: str = DEFAULT_PIERCE_MODEL,
    ensure_pierce: bool = True,
    taup_bin: str | Path = DEFAULT_TAUP_BIN,
) -> tuple[list[ThicknessPoint], "OrderedDict[str, str]"]:
    """扫描 scan_root，返回 (厚度点列表, event_key→星色)。

    每个 ``*.stack.json`` 视为一个潜在 group。配对不完整、缺成员、缺穿透点或厚度
    非法的会被跳过。穿透点缓存缺失时按需用 ``ensure_pierce_file`` 生成（可关）。
    """
    scan_root = Path(scan_root).expanduser().resolve()
    taup_bin = Path(taup_bin).expanduser().resolve()
    points: list[ThicknessPoint] = []
    event_colors: "OrderedDict[str, str]" = OrderedDict()
    color_index = 0
    # (dataset, event, phase) → 成员穿透点缓存，避免同事件多组重复加载。
    phase_cache: dict[tuple[str, str, str], dict] = {}

    for sidecar_path in _iter_sidecar_paths(scan_root):
        payload = _load_json(sidecar_path)
        if not payload:
            continue
        markers = payload.get("markers", {}) or {}
        rule = _pair_rule(markers)
        if rule is None:
            continue
        pair_kind, phase_kind = rule

        geometry = payload.get("geometry", {}) or {}
        event_info = payload.get("event", {}) or {}
        gcarc = _finite(geometry.get("gcarc_mean"))
        event_lon = _finite(event_info.get("evlo"))
        event_lat = _finite(event_info.get("evla"))
        evdp = _finite(event_info.get("evdp"))
        if None in (gcarc, event_lon, event_lat, evdp):
            continue

        members_file = _result_members_file(payload)
        if members_file is None:
            continue
        member_wave_names = _used_member_wave_names(members_file)
        if not member_wave_names:
            continue

        source_event_dir = str(payload.get("source_event_dir") or "").strip()
        if not source_event_dir:
            continue
        dataset, event = _dataset_event_from_sidecar(sidecar_path, scan_root)

        cache_key = (dataset, event, phase_kind)
        if cache_key not in phase_cache:
            try:
                if ensure_pierce:
                    ensure_pierce_file(
                        source_event_dir, phase_kind, model, pierce_depth_km=pierce_depth_km
                    )
                pierce_path = pierce_file_path(
                    source_event_dir, phase_kind, model, pierce_depth_km=pierce_depth_km
                )
                phase_cache[cache_key] = load_pierce_points(pierce_path)
            except Exception:
                phase_cache[cache_key] = {}
        records = phase_cache[cache_key]

        longitudes: list[float] = []
        latitudes: list[float] = []
        for wave_name in member_wave_names:
            record = records.get(wave_name)
            if record is None:
                continue
            longitudes.append(float(record.longitude))
            latitudes.append(float(record.latitude))
        if not longitudes or not latitudes:
            continue
        longitude = float(sum(longitudes) / len(longitudes))
        latitude = float(sum(latitudes) / len(latitudes))

        thickness = _thickness_for_payload(
            payload, pair_kind, phase_kind, evdp, gcarc,
            taup_bin,
        )
        if thickness is None or not math.isfinite(thickness) or thickness <= 0.0:
            continue

        event_key = f"{dataset}/{event}" if dataset else event
        if event_key not in event_colors:
            event_colors[event_key] = STAR_COLORS[color_index % len(STAR_COLORS)]
            color_index += 1

        align_marker = str(payload.get("align_marker") or "").strip()
        group_name = str(payload.get("group_name") or "").strip()
        result_package_dir = str(payload.get("result_package_dir") or "").strip()

        points.append(
            ThicknessPoint(
                dataset=dataset,
                event=event,
                event_label=_event_label(event),
                group_name=group_name,
                pair_kind=pair_kind,
                phase_kind=phase_kind,
                longitude=longitude,
                latitude=latitude,
                thickness_km=float(thickness),
                event_lon=event_lon,
                event_lat=event_lat,
                align_marker=align_marker,
                member_count_used=len(member_wave_names),
                gcarc=gcarc,
                evdp=evdp,
                result_package_dir=result_package_dir,
                sidecar_path=str(sidecar_path),
                source_event_dir=source_event_dir,
            )
        )

    return points, event_colors


# ---------------------------------------------------------------------------
# 带缓存的索引构建
# ---------------------------------------------------------------------------

@dataclass
class ThicknessIndex:
    points: list[ThicknessPoint] = field(default_factory=list)
    event_colors: "OrderedDict[str, str]" = field(default_factory=OrderedDict)
    scan_root: str = ""
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM
    model: str = DEFAULT_PIERCE_MODEL

    def group_key_to_point(self) -> dict[str, ThicknessPoint]:
        return {p.group_key: p for p in self.points}


def _cache_path(scan_root: Path, pierce_depth_km: float, model: str) -> Path:
    key = f"{scan_root}|{pierce_depth_km:.2f}|{model}"
    digest = __import__("hashlib").sha1(key.encode("utf-8")).hexdigest()[:16]
    return INDEX_CACHE_DIR / f"{digest}.json"


def _cache_fingerprint(scan_root: Path) -> dict[str, str]:
    # 用内容哈希而非 mtime：WSL2 等环境 mtime 分辨率粗，重写后 mtime 可能不变。
    # 纳入 sidecar（含 markers/align_marker/gcarc/pair）+ members.txt（成员数），
    # 这样主 ppk 改拾取/重跑 stack 后，获得焦点刷新能检测到变化。
    import hashlib
    fp: dict[str, str] = {}
    for p in _iter_sidecar_paths(scan_root):
        try:
            digest = hashlib.sha1(p.read_bytes()).hexdigest()[:16]
        except OSError:
            digest = "missing"
        fp[str(p)] = digest
        # 同目录结果包的 members.txt 也纳入（成员数变化）
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        rpd = str(payload.get("result_package_dir") or "").strip()
        if rpd:
            mfile = Path(rpd).expanduser().resolve() / "members.txt"
            try:
                fp[f"members:{mfile}"] = hashlib.sha1(mfile.read_bytes()).hexdigest()[:16]
            except OSError:
                fp[f"members:{mfile}"] = "missing"
    return fp


def build_thickness_index(
    scan_root: Path,
    *,
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM,
    model: str = DEFAULT_PIERCE_MODEL,
    ensure_pierce: bool = True,
    use_cache: bool = True,
    taup_bin: str | Path = DEFAULT_TAUP_BIN,
) -> ThicknessIndex:
    """带磁盘缓存的索引构建。缓存按 sidecar mtime 失效。"""
    ensure_stack_storage_ready()
    scan_root = Path(scan_root).expanduser().resolve()
    cache_file = _cache_path(scan_root, pierce_depth_km, model)

    if use_cache and cache_file.exists():
        try:
            blob = json.loads(cache_file.read_text(encoding="utf-8"))
            if (
                blob.get("scan_root") == str(scan_root)
                and float(blob.get("pierce_depth_km")) == float(pierce_depth_km)
                and blob.get("model") == model
                and blob.get("fingerprint") == _cache_fingerprint(scan_root)
            ):
                points = [ThicknessPoint(**p) for p in blob.get("points", [])]
                colors = OrderedDict(blob.get("event_colors", []))
                return ThicknessIndex(
                    points=points, event_colors=colors,
                    scan_root=str(scan_root),
                    pierce_depth_km=pierce_depth_km, model=model,
                )
        except Exception:
            pass  # 缓存损坏则重建

    points, event_colors = collect_points(
        scan_root,
        pierce_depth_km=pierce_depth_km,
        model=model,
        ensure_pierce=ensure_pierce,
        taup_bin=taup_bin,
    )
    index = ThicknessIndex(
        points=points, event_colors=event_colors,
        scan_root=str(scan_root),
        pierce_depth_km=pierce_depth_km, model=model,
    )

    try:
        INDEX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        blob = {
            "scan_root": str(scan_root),
            "pierce_depth_km": float(pierce_depth_km),
            "model": model,
            "fingerprint": _cache_fingerprint(scan_root),
            "points": [asdict(p) for p in points],
            "event_colors": list(event_colors.items()),
        }
        tmp = cache_file.with_name(f"{cache_file.name}.tmp")
        tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache_file)
    except OSError:
        pass
    return index


def invalidate_cache(scan_root: Path, *, pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM, model: str = DEFAULT_PIERCE_MODEL) -> None:
    cache_file = _cache_path(Path(scan_root).expanduser().resolve(), pierce_depth_km, model)
    try:
        cache_file.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 审阅标记持久化
# ---------------------------------------------------------------------------

def _review_file_for_event(event_dir: str | Path) -> Path:
    """审阅标记文件路径 = 事件 stack metadata 目录下 thickness_review.json。"""
    return stack_metadata_dir_for_event(event_dir) / REVIEW_FILE_NAME


def load_review_marks(event_dir: str | Path) -> dict[str, dict]:
    """返回 {group_name: {status, note}}。"""
    path = _review_file_for_event(event_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def save_review_marks(event_dir: str | Path, marks: dict[str, dict]) -> None:
    """原子写审阅标记。status 归一到合法值。"""
    path = _review_file_for_event(event_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned: dict[str, dict] = {}
    for group_name, entry in marks.items():
        entry = entry if isinstance(entry, dict) else {}
        status = str(entry.get("status") or "pending").strip().lower()
        if status not in REVIEW_STATUS_VALUES:
            status = "pending"
        cleaned[str(group_name)] = {"status": status, "note": str(entry.get("note") or "")}
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def set_review_mark(event_dir: str | Path, group_name: str, *, status: str, note: str = "") -> None:
    marks = load_review_marks(event_dir)
    marks[str(group_name)] = {"status": status, "note": note}
    save_review_marks(event_dir, marks)


__all__ = [
    "ThicknessPoint",
    "ThicknessIndex",
    "collect_points",
    "build_thickness_index",
    "invalidate_cache",
    "load_review_marks",
    "save_review_marks",
    "set_review_mark",
    "STAR_COLORS",
    "REVIEW_STATUS_VALUES",
    "DEFAULT_PIERCE_MODEL",
    "DEFAULT_PIERCE_DEPTH_KM",
]
