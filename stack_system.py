from __future__ import annotations

import datetime as _dt
import json
import math
import numbers
import re
import shutil
from pathlib import Path

import obspy

from pierce_point_cache import PROJECT_ROOT, relative_event_path


STACK_ROOT = PROJECT_ROOT / "data" / "output" / "stack"
STACK_DATA_ROOT = STACK_ROOT / "stack_files"
STACK_OUTPUT_ROOT = STACK_ROOT / "analysis"
LEGACY_STACK_DATA_ROOT = PROJECT_ROOT / "data" / "stack"
LEGACY_STACK_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "process" / "stack"
LEGACY_STACK_PHASE_ROOT = PROJECT_ROOT / "data" / "output" / "phases" / "stack"
LEGACY_STACK_PIERCE_ROOT = PROJECT_ROOT / "data" / "output" / "pierce_points" / "output" / "process" / "stack"
STACK_EVENT_MARKER = ".stack_event.json"
STACK_INDEX_FILE = ".stack_index.json"
_STACK_STORAGE_READY = False

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def _natural_sort_key(value: str) -> tuple:
    """Sort strings so embedded numbers compare by magnitude (group2 < group10).

    Each segment is wrapped as (type_tag, value) so digit and non-digit segments
    never mix types in the same tuple (Python 3 forbids str < int comparison).
    Digit segments tag 0 (numeric compare), text segments tag 1.
    """
    key = []
    for part in _NATURAL_SORT_RE.split(value or ""):
        if part == "" or part is None:
            continue
        if part.isdigit():
            key.append((0, int(part), ""))
        else:
            key.append((1, 0, part.lower()))
    return tuple(key)


def ensure_stack_storage_ready() -> dict:
    global _STACK_STORAGE_READY
    if _STACK_STORAGE_READY:
        return {"moved_count": 0, "moved": [], "stack_root": str(STACK_ROOT)}
    report = migrate_legacy_stack_storage()
    _STACK_STORAGE_READY = True
    return report


def stack_event_dir_for_source(event_dir: str | Path) -> Path:
    ensure_stack_storage_ready()
    return STACK_DATA_ROOT / relative_event_path(event_dir)


def stack_output_dir_for_source(event_dir: str | Path) -> Path:
    ensure_stack_storage_ready()
    return STACK_OUTPUT_ROOT / relative_event_path(event_dir)


def _stack_relative_event_path(path: str | Path) -> Path | None:
    resolved_path = Path(path).expanduser().resolve()
    for root in (
        STACK_DATA_ROOT,
        STACK_OUTPUT_ROOT,
        LEGACY_STACK_DATA_ROOT,
        LEGACY_STACK_OUTPUT_ROOT,
        LEGACY_STACK_PHASE_ROOT,
        LEGACY_STACK_PIERCE_ROOT,
    ):
        try:
            relative_path = resolved_path.relative_to(root)
        except ValueError:
            continue
        if len(relative_path.parts) >= 2:
            return Path(*relative_path.parts[:2])
    return None


def _legacy_stack_data_relative_path(path: str | Path) -> Path | None:
    resolved_path = Path(path).expanduser().resolve()
    try:
        relative_path = resolved_path.relative_to(LEGACY_STACK_DATA_ROOT)
    except ValueError:
        return None
    if len(relative_path.parts) >= 2:
        return Path(*relative_path.parts[:2])
    return None


def stack_metadata_dir_for_event(event_dir: str | Path) -> Path:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    if _legacy_stack_event_marker_path(event_path).exists():
        return event_path
    relative_path = _stack_relative_event_path(event_path)
    if relative_path is not None:
        return STACK_OUTPUT_ROOT / relative_path
    return stack_output_dir_for_source(event_path)


def stack_event_marker_path(event_dir: str | Path) -> Path:
    return stack_metadata_dir_for_event(event_dir) / STACK_EVENT_MARKER


def stack_index_path(event_dir: str | Path) -> Path:
    return stack_metadata_dir_for_event(event_dir) / STACK_INDEX_FILE


def _legacy_stack_event_marker_path(event_dir: str | Path) -> Path:
    return Path(event_dir).expanduser().resolve() / STACK_EVENT_MARKER


def _legacy_stack_index_path(event_dir: str | Path) -> Path:
    return Path(event_dir).expanduser().resolve() / STACK_INDEX_FILE


def stack_sidecar_path(stack_sac_path: str | Path, *, event_dir: str | Path | None = None) -> Path:
    sac_path = Path(stack_sac_path).expanduser().resolve()
    if event_dir is None:
        event_path = _stack_event_root_for_path(sac_path)
    else:
        event_path = Path(event_dir).expanduser().resolve()
    wave_name = stack_wave_name_from_path(event_path, sac_path)
    sidecar_name = f"{Path(wave_name).stem}.stack.json"
    return stack_metadata_dir_for_event(event_path) / sidecar_name


def stack_wave_name_from_path(event_dir: str | Path, stack_sac_path: str | Path) -> str:
    event_path = Path(event_dir).expanduser().resolve()
    sac_path = Path(stack_sac_path).expanduser().resolve()
    try:
        return sac_path.relative_to(event_path).as_posix()
    except ValueError:
        return sac_path.name


def iter_stack_sac_paths(event_dir: str | Path, *, suffix: str = ".sac") -> list[Path]:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    if not event_path.exists():
        return []
    suffix_lower = str(suffix or ".sac").lower()
    paths = []
    for sac_path in event_path.rglob("*"):
        if not sac_path.is_file():
            continue
        relative_parts = sac_path.relative_to(event_path).parts
        if any(part.startswith(".") or part == "_trash_invalid" for part in relative_parts):
            continue
        if sac_path.name.lower().endswith(suffix_lower):
            paths.append(sac_path)
    return sorted(paths, key=lambda path: _natural_sort_key(path.relative_to(event_path).as_posix()))


def iter_stack_sidecar_paths(event_dir: str | Path) -> list[Path]:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    search_roots: list[Path] = []
    metadata_dir = stack_metadata_dir_for_event(event_path)
    if metadata_dir.exists():
        search_roots.append(metadata_dir)
    if event_path.exists():
        search_roots.append(event_path)
    paths_by_resolved: dict[Path, Path] = {}
    for root in search_roots:
        for json_path in root.rglob("*.stack.json"):
            if not json_path.is_file():
                continue
            try:
                relative_parts = json_path.relative_to(root).parts
            except ValueError:
                relative_parts = json_path.parts
            if any(part.startswith(".") or part == "_trash_invalid" for part in relative_parts):
                continue
            paths_by_resolved.setdefault(json_path.resolve(), json_path)
    return sorted(
        paths_by_resolved.values(),
        key=lambda path: (0 if path.is_relative_to(metadata_dir) else 1, _natural_sort_key(str(path))),
    )


def stack_sac_time_window(event_dir: str | Path, *, suffix: str = ".sac") -> list[float] | None:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    start_values = []
    end_values = []
    for sac_path in iter_stack_sac_paths(event_path, suffix=suffix):
        try:
            trace = obspy.read(str(sac_path), headonly=True)[0]
        except Exception:
            continue
        sac = getattr(trace.stats, "sac", None)
        if sac is None:
            continue
        start_value = _json_finite(getattr(sac, "b", None))
        end_value = _json_finite(getattr(sac, "e", None))
        if start_value is None or end_value is None:
            continue
        start_values.append(float(start_value))
        end_values.append(float(end_value))
    if not start_values or not end_values:
        return None
    return [math.floor(min(start_values)), math.ceil(max(end_values))]


def is_stack_event_dir(event_dir: str | Path) -> bool:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    try:
        event_path.relative_to(STACK_DATA_ROOT)
        return True
    except ValueError:
        # Output-side stack metadata can exist for an ordinary source event after
        # a stack workspace was created.  That should not make `dpk <source>`
        # open the stack subsystem; only an actual stack workspace path, or the
        # older in-directory marker layout, should select stack mode.
        return _legacy_stack_event_marker_path(event_path).exists()


def load_json_file(path: str | Path, default=None):
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _load_json_file_with_error(path: str | Path, default=None):
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        return default, ""
    try:
        return json.loads(file_path.read_text(encoding="utf-8")), ""
    except Exception as exc:
        return default, f"{type(exc).__name__}: {exc}"


def _dict_or_empty(value) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _list_or_empty(value) -> list:
    return list(value) if isinstance(value, list) else []


def _count_or_default(value, default: int = 0):
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _changed_top_level_fields(original: dict, normalized: dict) -> list[str]:
    fields = []
    for key in sorted(set(original.keys()) | set(normalized.keys())):
        if original.get(key) != normalized.get(key):
            fields.append(str(key))
    return fields


def _invalid_json_payload_reason(payload) -> str:
    if isinstance(payload, dict):
        return ""
    return f"expected JSON object, got {type(payload).__name__}"


def _stack_wave_name_from_sidecar_path(sidecar_path: str | Path, event_dir: str | Path | None = None) -> str:
    json_path = Path(sidecar_path).expanduser().resolve()
    json_name = json_path.name
    suffix = ".stack.json"
    if json_name.endswith(suffix):
        base_name = json_name[:-len(suffix)]
        if base_name.lower().endswith(".sac"):
            sac_name = base_name
        else:
            sac_name = f"{base_name}.sac"
    else:
        stem = Path(json_name).stem.replace(".stack", "")
        if stem.lower().endswith(".sac"):
            sac_name = stem
        else:
            sac_name = f"{stem}.sac" if stem else ""
    if event_dir is None or not sac_name:
        return sac_name
    return stack_wave_name_from_path(event_dir, json_path.with_name(sac_name))


def _stack_package_name_from_wave_name(stack_wave_name: str) -> str:
    wave_path = Path(str(stack_wave_name or ""))
    if len(wave_path.parts) > 1:
        return wave_path.parts[0]
    name = wave_path.name
    if name.lower().endswith(".sac"):
        return name[:-4]
    return name


def _source_event_dir_from_stack_path(event_dir: str | Path) -> Path | None:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    for root in (STACK_DATA_ROOT, LEGACY_STACK_DATA_ROOT):
        try:
            relative_path = event_path.relative_to(root)
        except ValueError:
            continue
        return (PROJECT_ROOT / "data" / relative_path).resolve()
    return None


def _migrate_tree_contents(source_root: Path, target_root: Path) -> list[tuple[Path, Path]]:
    moved: list[tuple[Path, Path]] = []
    if not source_root.exists():
        return moved
    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        shutil.move(str(path), str(destination))
        moved.append((path, destination))
    for directory in sorted(source_root.rglob("*"), reverse=True):
        if directory.is_dir():
            try:
                directory.rmdir()
            except OSError:
                pass
    try:
        source_root.rmdir()
    except OSError:
        pass
    return moved


def migrate_legacy_stack_storage() -> dict:
    STACK_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    STACK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    moved = []
    moved.extend(_migrate_tree_contents(LEGACY_STACK_DATA_ROOT, STACK_DATA_ROOT))
    moved.extend(_migrate_tree_contents(LEGACY_STACK_OUTPUT_ROOT, STACK_OUTPUT_ROOT))
    moved.extend(_migrate_tree_contents(LEGACY_STACK_PHASE_ROOT, STACK_OUTPUT_ROOT))
    moved.extend(_migrate_tree_contents(LEGACY_STACK_PIERCE_ROOT, STACK_OUTPUT_ROOT))
    return {
        "moved_count": len(moved),
        "moved": [(str(src), str(dst)) for src, dst in moved],
        "stack_root": str(STACK_ROOT),
    }


def _stack_event_root_for_path(path: str | Path) -> Path:
    current = Path(path).expanduser().resolve()
    relative_path = _stack_relative_event_path(current)
    if relative_path is not None:
        return (STACK_DATA_ROOT / relative_path).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if stack_event_marker_path(candidate).exists():
            return candidate
    return current


def _resolved_stack_marker_path(event_dir: str | Path) -> Path:
    event_path = Path(event_dir).expanduser().resolve()
    marker_path = stack_event_marker_path(event_path)
    if marker_path.exists():
        return marker_path
    legacy_path = _legacy_stack_event_marker_path(event_path)
    if legacy_path.exists():
        return legacy_path
    return marker_path


def _resolved_stack_index_path(event_dir: str | Path) -> Path:
    event_path = Path(event_dir).expanduser().resolve()
    index_path = stack_index_path(event_path)
    if index_path.exists():
        return index_path
    legacy_path = _legacy_stack_index_path(event_path)
    if legacy_path.exists():
        return legacy_path
    return index_path


def _is_under_stack_output_root(path: str | Path) -> bool:
    try:
        Path(path).expanduser().resolve().relative_to(STACK_OUTPUT_ROOT)
        return True
    except Exception:
        return False


def _remap_legacy_stack_output_path(path: str | Path) -> Path | None:
    resolved_path = Path(path).expanduser().resolve()
    for legacy_root in (
        LEGACY_STACK_OUTPUT_ROOT,
        LEGACY_STACK_PHASE_ROOT,
        LEGACY_STACK_PIERCE_ROOT,
    ):
        try:
            relative_path = resolved_path.relative_to(legacy_root)
        except ValueError:
            continue
        return STACK_OUTPUT_ROOT / relative_path
    return None


def _stack_package_metadata_payload(result_package_path: Path, outputs: dict | None = None) -> dict:
    candidates = []
    if outputs:
        json_output = outputs.get("json")
        if isinstance(json_output, str):
            candidates.append(Path(json_output).expanduser().resolve())
    candidates.append(result_package_path / "meta.json")
    for candidate in candidates:
        payload = load_json_file(candidate, default={}) or {}
        if isinstance(payload, dict) and payload:
            return payload
    return {}


_STACK_PACKAGE_DETAIL_KEYS = (
    "wave_names_requested",
    "wave_names_aligned",
    "wave_names_used",
    "skipped_missing_reference",
    "skipped_normalization",
    "moveout_applied",
    "moveout_skipped",
)


_STACK_SIDECAR_PACKAGE_ONLY_KEYS = set(_STACK_PACKAGE_DETAIL_KEYS)


def _stack_sidecar_storage_payload(payload: dict) -> dict:
    return {
        key: value
        for key, value in dict(payload or {}).items()
        if key not in _STACK_SIDECAR_PACKAGE_ONLY_KEYS
    }


def _stack_sidecar_group_name(payload: dict | None) -> str:
    sidecar = payload or {}
    group_name = str(sidecar.get("group_name") or "").strip()
    if group_name:
        return group_name
    scope_text = str(sidecar.get("scope") or "").strip()
    if scope_text.lower().startswith("group:"):
        return scope_text.split(":", 1)[1].strip()
    return ""


def _stack_sidecar_member_set(payload: dict | None) -> set[str]:
    sidecar = payload or {}
    for key in ("wave_names_used", "wave_names_aligned", "wave_names_requested"):
        values = sidecar.get(key)
        if isinstance(values, list) and values:
            return {str(item) for item in values if str(item)}
    return set()


def _safe_remove_stack_package(path_value) -> str:
    if not isinstance(path_value, str) or not path_value.strip():
        return ""
    package_path = Path(path_value).expanduser().resolve()
    if not package_path.exists() or not package_path.is_dir():
        return ""
    if not package_path.name.startswith("stack"):
        return ""
    if not (package_path / "meta.json").exists():
        return ""
    shutil.rmtree(package_path)
    return str(package_path)


def _delete_stack_artifacts(event_dir: str | Path, wave_name: str, sidecar: dict | None = None) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    stack_wave_name = str(wave_name or "").strip()
    if not stack_wave_name:
        return {"wave_name": "", "removed": []}
    sac_path = (event_path / stack_wave_name).resolve()
    sidecar_path = stack_sidecar_path(sac_path, event_dir=event_path)
    payload = sidecar or load_stack_sidecar_map(event_path).get(stack_wave_name, {})
    removed = []
    if sac_path.exists() and sac_path.is_file():
        sac_path.unlink()
        removed.append(str(sac_path))
    if sidecar_path.exists() and sidecar_path.is_file():
        sidecar_path.unlink()
        removed.append(str(sidecar_path))
    package_removed = _safe_remove_stack_package(payload.get("result_package_dir"))
    if package_removed:
        removed.append(package_removed)
    return {"wave_name": stack_wave_name, "removed": removed}


def delete_stack_config(event_dir: str | Path, stack_wave_name: str, *, refresh_index: bool = True) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    sidecar = load_stack_sidecar_map(event_path).get(str(stack_wave_name or "").strip(), {})
    result = _delete_stack_artifacts(event_path, stack_wave_name, sidecar)
    if refresh_index and result.get("removed"):
        write_stack_workspace_index(event_path)
    return result


def delete_stack_group_configs(
    event_dir: str | Path,
    group_name: str,
    *,
    refresh_index: bool = True,
) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    target_group = str(group_name or "").strip().lower()
    deleted = []
    if not target_group:
        return {"group_name": "", "deleted": deleted}
    for wave_name, sidecar in list(load_stack_sidecar_map(event_path).items()):
        if _stack_sidecar_group_name(sidecar).lower() != target_group:
            continue
        result = _delete_stack_artifacts(event_path, wave_name, sidecar)
        if result.get("removed"):
            deleted.append(result)
    if refresh_index and deleted:
        write_stack_workspace_index(event_path)
    return {"group_name": group_name, "deleted": deleted}


def _normalize_stack_event_marker_payload(event_dir: str | Path, payload: dict) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    normalized = dict(payload or {})
    normalized["mode"] = "stack"
    source_event_dir = normalized.get("source_event_dir")
    if not source_event_dir:
        inferred_source_event_dir = _source_event_dir_from_stack_path(event_path)
        if inferred_source_event_dir is not None:
            source_event_dir = str(inferred_source_event_dir)
            normalized["source_event_dir"] = source_event_dir
    if source_event_dir:
        source_path = Path(source_event_dir).expanduser().resolve()
        normalized["source_event_dir"] = str(source_path)
        normalized.setdefault("source_event_name", source_path.name)
    output_dir = normalized.get("output_dir")
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        if not _is_under_stack_output_root(output_path) and source_event_dir:
            output_path = stack_output_dir_for_source(source_event_dir)
        normalized["output_dir"] = str(output_path)
    elif source_event_dir:
        normalized["output_dir"] = str(stack_output_dir_for_source(source_event_dir))
    if "source_event_name" not in normalized:
        source_name = ""
        if source_event_dir:
            source_name = Path(source_event_dir).expanduser().resolve().name
        if not source_name:
            source_name = event_path.name
        normalized["source_event_name"] = str(source_name)
    return normalized


def _normalize_stack_sidecar_payload(
    event_dir: str | Path,
    sidecar_path: str | Path,
    payload: dict,
    marker: dict | None = None,
) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    json_path = Path(sidecar_path).expanduser().resolve()
    normalized = dict(payload or {})
    normalized["mode"] = "stack"
    marker = marker or load_stack_event_marker(event_path)
    canonical_source_event_dir = str(marker.get("source_event_dir") or source_event_dir_for_runtime(event_path))
    canonical_source_event_name = str(
        marker.get("source_event_name") or Path(canonical_source_event_dir).name
    )
    canonical_output_dir = Path(
        marker.get("output_dir") or stack_output_dir_for_source(canonical_source_event_dir)
    ).expanduser().resolve()
    expected_stack_wave_name = _stack_wave_name_from_sidecar_path(json_path, event_path)
    stack_wave_name = str(normalized.get("stack_wave_name") or expected_stack_wave_name).strip()
    if expected_stack_wave_name:
        stack_wave_name = expected_stack_wave_name
    normalized["stack_wave_name"] = stack_wave_name
    normalized["source_event_dir"] = canonical_source_event_dir
    normalized["source_event_name"] = canonical_source_event_name

    package_name = _stack_package_name_from_wave_name(stack_wave_name)
    canonical_package_dir = canonical_output_dir / package_name

    result_package_dir = normalized.get("result_package_dir")
    if result_package_dir:
        result_package_path = Path(result_package_dir).expanduser().resolve()
        remapped_result_package_path = _remap_legacy_stack_output_path(result_package_path)
        if remapped_result_package_path is not None:
            result_package_path = remapped_result_package_path
        elif not _is_under_stack_output_root(result_package_path) and not result_package_path.exists():
            result_package_path = canonical_package_dir
    else:
        result_package_path = canonical_package_dir
    normalized["result_package_dir"] = str(result_package_path)

    outputs = _dict_or_empty(normalized.get("outputs"))
    for key, value in list(outputs.items()):
        if not isinstance(value, str):
            continue
        try:
            candidate = Path(value).expanduser().resolve()
        except Exception:
            continue
        remapped_candidate = _remap_legacy_stack_output_path(candidate)
        if remapped_candidate is not None:
            outputs[key] = str(remapped_candidate)
        elif not _is_under_stack_output_root(candidate) and not candidate.exists():
            outputs[key] = str(canonical_package_dir / Path(value).name)

    package_metadata = _stack_package_metadata_payload(result_package_path, outputs)
    if package_metadata:
        for key in _STACK_PACKAGE_DETAIL_KEYS:
            if key not in normalized:
                normalized[key] = package_metadata.get(key, [] if "wave_names" in key or "skipped" in key or "moveout" in key else None)
        for key in ("wave_count_requested", "wave_count_input", "wave_count_used"):
            if key not in normalized and key in package_metadata:
                normalized[key] = package_metadata.get(key)
        if not outputs:
            outputs = _dict_or_empty(package_metadata.get("outputs"))
            for key, value in list(outputs.items()):
                if not isinstance(value, str):
                    continue
                try:
                    candidate = Path(value).expanduser().resolve()
                except Exception:
                    continue
                remapped_candidate = _remap_legacy_stack_output_path(candidate)
                if remapped_candidate is not None:
                    outputs[key] = str(remapped_candidate)
                elif not _is_under_stack_output_root(candidate) and not candidate.exists():
                    outputs[key] = str(result_package_path / Path(value).name)
    normalized["outputs"] = outputs
    normalized["geometry"] = _dict_or_empty(normalized.get("geometry"))
    normalized["event"] = _dict_or_empty(normalized.get("event"))
    if "group_name" in normalized:
        normalized["group_name"] = str(normalized.get("group_name") or "").strip()

    markers = _dict_or_empty(normalized.get("markers"))
    normalized["markers"] = {
        f"t{idx}": _json_finite(markers.get(f"t{idx}", markers.get(str(idx))))
        for idx in range(10)
    }
    user_markers = _dict_or_empty(normalized.get("user_markers"))
    normalized["user_markers"] = {
        key: _json_finite(user_markers.get(key))
        for key in ("user1", "user2", "user3", "user4", "user5")
    }

    for key in (
        "wave_names_requested",
        "wave_names_aligned",
        "wave_names_used",
        "skipped_missing_reference",
        "skipped_normalization",
        "moveout_applied",
        "moveout_skipped",
    ):
        normalized[key] = _list_or_empty(normalized.get(key))

    normalized["wave_count_requested"] = _count_or_default(
        normalized.get("wave_count_requested"),
        len(normalized["wave_names_requested"]),
    )
    normalized["wave_count_input"] = _count_or_default(
        normalized.get("wave_count_input"),
        len(normalized["wave_names_aligned"]),
    )
    normalized["wave_count_used"] = _count_or_default(
        normalized.get("wave_count_used"),
        len(normalized["wave_names_used"]),
    )
    return normalized


def write_stack_event_marker(
    event_dir: str | Path,
    *,
    source_event_dir: str | Path,
    output_dir: str | Path,
    source_event_name: str | None = None,
) -> Path:
    event_path = Path(event_dir).expanduser().resolve()
    event_path.mkdir(parents=True, exist_ok=True)
    marker_path = stack_event_marker_path(event_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "stack",
        "source_event_dir": str(Path(source_event_dir).expanduser().resolve()),
        "output_dir": str(Path(output_dir).expanduser().resolve()),
        "source_event_name": str(source_event_name or Path(source_event_dir).expanduser().resolve().name),
    }
    marker_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return marker_path


def load_stack_event_marker(event_dir: str | Path) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    payload = load_json_file(_resolved_stack_marker_path(event_path), default={}) or {}
    if not isinstance(payload, dict):
        return {}
    if not payload:
        return {}
    return _normalize_stack_event_marker_payload(event_path, payload)


def stack_output_dir_for_runtime(event_dir: str | Path) -> Path:
    marker = load_stack_event_marker(event_dir)
    output_dir = marker.get("output_dir")
    if output_dir:
        output_path = Path(output_dir).expanduser().resolve()
        try:
            output_path.relative_to(STACK_OUTPUT_ROOT)
            return output_path
        except ValueError:
            pass
    return stack_output_dir_for_source(source_event_dir_for_runtime(event_dir))


def load_stack_sidecar_map(event_dir: str | Path) -> dict[str, dict]:
    event_path = Path(event_dir).expanduser().resolve()
    marker = load_stack_event_marker(event_path)
    sidecars: dict[str, dict] = {}
    if not event_path.exists():
        return sidecars
    for json_path in iter_stack_sidecar_paths(event_path):
        payload, error = _load_json_file_with_error(json_path, default=None)
        if error or not isinstance(payload, dict):
            continue
        payload = _normalize_stack_sidecar_payload(event_path, json_path, payload, marker=marker)
        wave_name = str(payload.get("stack_wave_name") or _stack_wave_name_from_sidecar_path(json_path, event_path)).strip()
        if wave_name:
            sidecars[wave_name] = payload
    return sidecars


def write_stack_sidecar_payload(
    stack_sac_path: str | Path,
    payload: dict,
    *,
    event_dir: str | Path | None = None,
) -> dict:
    sac_path = Path(stack_sac_path).expanduser().resolve()
    event_path = Path(event_dir).expanduser().resolve() if event_dir is not None else sac_path.parent
    json_path = stack_sidecar_path(sac_path, event_dir=event_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_stack_sidecar_payload(event_path, json_path, payload)
    normalized = _json_safe_payload(_stack_sidecar_storage_payload(normalized))
    temp_path = json_path.with_name(f"{json_path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(json_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return normalized


def update_stack_sidecar_markers(
    stack_sac_path: str | Path,
    *,
    markers: dict | None = None,
    user_markers: dict | None = None,
) -> dict:
    sac_path = Path(stack_sac_path).expanduser().resolve()
    event_path = _stack_event_root_for_path(sac_path)
    json_path = stack_sidecar_path(sac_path, event_dir=event_path)
    payload = load_json_file(json_path, default={}) or {}
    payload.setdefault("mode", "stack")
    payload["stack_wave_name"] = stack_wave_name_from_path(event_path, sac_path)
    if markers is not None:
        payload["markers"] = {
            f"t{idx}": _json_finite(markers.get(f"t{idx}", markers.get(str(idx))))
            for idx in range(10)
        }
    if user_markers is not None:
        payload["user_markers"] = {
            key: _json_finite(user_markers.get(key))
            for key in ("user1", "user2", "user3", "user4", "user5")
        }
    payload["updated_at"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return write_stack_sidecar_payload(sac_path, payload, event_dir=_stack_event_root_for_path(sac_path))


def stack_wave_summary_from_sidecar(sidecar: dict | None) -> str:
    payload = sidecar or {}
    parts = []
    for key in ("scope", "stack_type", "normalize"):
        value = str(payload.get(key, "") or "").strip()
        if value:
            parts.append(value)
    try:
        wave_count_used = int(payload.get("wave_count_used", 0) or 0)
    except (TypeError, ValueError):
        wave_count_used = 0
    if wave_count_used > 0:
        parts.append(f"N={wave_count_used}")
    return " | ".join(parts)


def _json_finite(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return value
    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _json_safe_payload(value):
    if isinstance(value, dict):
        return {str(key): _json_safe_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_payload(item) for item in value]
    return _json_finite(value)


def source_event_dir_for_runtime(event_dir: str | Path) -> Path:
    event_path = Path(event_dir).expanduser().resolve()
    if not is_stack_event_dir(event_path):
        return event_path
    marker = load_stack_event_marker(event_path)
    source_event_dir = marker.get("source_event_dir")
    if source_event_dir:
        return Path(source_event_dir).expanduser().resolve()
    inferred_source_event_dir = _source_event_dir_from_stack_path(event_path)
    if inferred_source_event_dir is not None:
        return inferred_source_event_dir
    return event_path


def resolve_stack_workspace_dir(event_dir: str | Path) -> Path:
    ensure_stack_storage_ready()
    event_path = Path(event_dir).expanduser().resolve()
    legacy_relative_path = _legacy_stack_data_relative_path(event_path)
    if legacy_relative_path is not None:
        migrated_path = STACK_DATA_ROOT / legacy_relative_path
        if event_path.exists():
            _migrate_tree_contents(event_path, migrated_path)
            migrated_path.mkdir(parents=True, exist_ok=True)
        return migrated_path
    if is_stack_event_dir(event_path):
        return event_path
    return stack_event_dir_for_source(source_event_dir_for_runtime(event_path))


def ensure_stack_workspace_dir(event_dir: str | Path) -> Path:
    ensure_stack_storage_ready()
    input_path = Path(event_dir).expanduser().resolve()
    if is_stack_event_dir(input_path):
        stack_event_dir = input_path
        source_event_dir = source_event_dir_for_runtime(input_path)
    else:
        source_event_dir = input_path
        stack_event_dir = stack_event_dir_for_source(source_event_dir)
    write_stack_event_marker(
        stack_event_dir,
        source_event_dir=source_event_dir,
        output_dir=stack_output_dir_for_source(source_event_dir),
        source_event_name=source_event_dir.name,
    )
    write_stack_workspace_index(stack_event_dir)
    return stack_event_dir


def build_stack_workspace_manifest(event_dir: str | Path) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    marker = load_stack_event_marker(event_path)
    sidecars = load_stack_sidecar_map(event_path)
    health = inspect_stack_event_health(event_path)
    invalid_by_path = {
        str(Path(item.get("path", "")).expanduser().resolve()): item.get("reason", "")
        for item in health.get("invalid_sac_files", [])
        if item.get("path")
    }
    invalid_sidecars_by_wave = {
        item.get("wave_name", ""): item
        for item in health.get("invalid_sidecars", [])
        if item.get("wave_name")
    }
    repair_sidecars_by_wave = {
        item.get("wave_name", ""): item
        for item in health.get("sidecars_needing_repair", [])
        if item.get("wave_name")
    }
    stack_items = []
    if event_path.exists():
        for sac_path in iter_stack_sac_paths(event_path):
            wave_name = stack_wave_name_from_path(event_path, sac_path)
            sidecar = sidecars.get(wave_name, {})
            geometry = sidecar.get("geometry", {}) or {}
            resolved_sac_path = str(sac_path.resolve())
            read_error = invalid_by_path.get(resolved_sac_path, "")
            sidecar_path = stack_sidecar_path(sac_path, event_dir=event_path)
            invalid_sidecar = invalid_sidecars_by_wave.get(wave_name, {})
            repair_sidecar = repair_sidecars_by_wave.get(wave_name, {})
            wave_names_requested = list(sidecar.get("wave_names_requested", []) or [])
            wave_names_aligned = list(sidecar.get("wave_names_aligned", []) or [])
            wave_names_used = list(sidecar.get("wave_names_used", []) or [])
            skipped_missing = list(sidecar.get("skipped_missing_reference", []) or [])
            skipped_normalization = list(sidecar.get("skipped_normalization", []) or [])
            moveout_skipped = list(sidecar.get("moveout_skipped", []) or [])
            stack_items.append({
                "wave_name": wave_name,
                "path": resolved_sac_path,
                "valid_sac": not read_error,
                "read_error": read_error,
                "sidecar_path": str(sidecar_path),
                "has_sidecar": sidecar_path.exists(),
                "sidecar_valid": sidecar_path.exists() and not invalid_sidecar,
                "sidecar_error": invalid_sidecar.get("reason", ""),
                "sidecar_needs_repair": bool(repair_sidecar),
                "sidecar_repair_fields": list(repair_sidecar.get("fields", []) or []),
                "summary": stack_wave_summary_from_sidecar(sidecar),
                "result_package_dir": sidecar.get("result_package_dir", ""),
                "wave_count_used": sidecar.get("wave_count_used", 0),
                "wave_count_requested": sidecar.get("wave_count_requested", 0),
                "member_counts": {
                    "requested": len(wave_names_requested),
                    "aligned": len(wave_names_aligned),
                    "used": len(wave_names_used),
                    "skipped_missing_reference": len(skipped_missing),
                    "skipped_normalization": len(skipped_normalization),
                    "skipped_moveout": len(moveout_skipped),
                },
                "members": {
                    "requested": wave_names_requested,
                    "aligned": wave_names_aligned,
                    "used": wave_names_used,
                    "skipped_missing_reference": skipped_missing,
                    "skipped_normalization": skipped_normalization,
                    "skipped_moveout": moveout_skipped,
                },
                "gcarc_mean": _json_finite(geometry.get("gcarc_mean", geometry.get("gcarc", None))),
                "az_mean": _json_finite(geometry.get("az_mean", geometry.get("az", None))),
                "baz_mean": _json_finite(geometry.get("baz_mean", geometry.get("baz", None))),
                "pierce_lon_mean": _json_finite(geometry.get("pierce_lon_mean", None)),
                "pierce_lat_mean": _json_finite(geometry.get("pierce_lat_mean", None)),
            })
    return {
        "event_dir": str(event_path),
        "source_event_dir": marker.get("source_event_dir", ""),
        "source_event_name": marker.get("source_event_name", ""),
        "output_dir": marker.get("output_dir", ""),
        "stack_count": len(stack_items),
        "valid_stack_count": sum(1 for item in stack_items if item["valid_sac"]),
        "invalid_stack_count": sum(1 for item in stack_items if not item["valid_sac"]),
        "stacks": stack_items,
    }


def build_stack_workspace_index(event_dir: str | Path) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    manifest = build_stack_workspace_manifest(event_path)
    health = inspect_stack_event_health(event_path)
    index_manifest = dict(manifest)
    index_manifest["stacks"] = []
    for item in manifest.get("stacks", []):
        index_item = dict(item)
        index_item.pop("members", None)
        index_manifest["stacks"].append(index_item)
    return {
        "mode": "stack_index",
        "generated_at": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **index_manifest,
        "health": health,
    }


def write_stack_workspace_index(event_dir: str | Path) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    event_path.mkdir(parents=True, exist_ok=True)
    payload = build_stack_workspace_index(event_path)
    index_path = stack_index_path(event_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = index_path.with_name(f"{index_path.name}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(index_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return payload


def repair_stack_event_metadata(event_dir: str | Path, persist: bool = True) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    report = {
        "event_dir": str(event_path),
        "marker_updated": False,
        "marker_created": False,
        "invalid_marker": "",
        "sidecars_updated": [],
        "invalid_sidecars": [],
    }
    if not event_path.exists():
        return report

    marker_path = stack_event_marker_path(event_path)
    raw_marker, marker_error = _load_json_file_with_error(_resolved_stack_marker_path(event_path), default=None)
    if marker_error:
        report["invalid_marker"] = marker_error
    elif raw_marker is not None:
        marker_payload_error = _invalid_json_payload_reason(raw_marker)
        if marker_payload_error:
            report["invalid_marker"] = marker_payload_error
        elif raw_marker:
            normalized_marker = _normalize_stack_event_marker_payload(event_path, raw_marker)
            if normalized_marker != raw_marker:
                report["marker_updated"] = True
                if persist:
                    marker_path.write_text(
                        json.dumps(normalized_marker, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    elif raw_marker is None and _source_event_dir_from_stack_path(event_path) is not None:
        normalized_marker = _normalize_stack_event_marker_payload(event_path, {})
        report["marker_created"] = True
        report["marker_updated"] = True
        if persist:
            marker_path.write_text(
                json.dumps(normalized_marker, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    marker = load_stack_event_marker(event_path)
    for json_path in iter_stack_sidecar_paths(event_path):
        raw_payload, sidecar_error = _load_json_file_with_error(json_path, default=None)
        wave_name = _stack_wave_name_from_sidecar_path(json_path, event_path)
        if sidecar_error:
            report["invalid_sidecars"].append({
                "path": str(json_path),
                "wave_name": wave_name,
                "reason": sidecar_error,
            })
            continue
        payload_error = _invalid_json_payload_reason(raw_payload)
        if payload_error:
            report["invalid_sidecars"].append({
                "path": str(json_path),
                "wave_name": wave_name,
                "reason": payload_error,
            })
            continue
        normalized_payload = _json_safe_payload(
            _stack_sidecar_storage_payload(
                _normalize_stack_sidecar_payload(event_path, json_path, raw_payload, marker=marker)
            )
        )
        if normalized_payload != raw_payload:
            report["sidecars_updated"].append(str(json_path))
            if persist:
                json_path.write_text(
                    json.dumps(normalized_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    if persist:
        write_stack_workspace_index(event_path)
    return report


def inspect_stack_event_health(event_dir: str | Path) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    report = {
        "event_dir": str(event_path),
        "missing_marker": False,
        "invalid_marker": "",
        "marker_needs_repair": False,
        "marker_repair_fields": [],
        "invalid_sac_files": [],
        "missing_sidecars": [],
        "orphan_sidecars": [],
        "invalid_sidecars": [],
        "sidecars_needing_repair": [],
        "valid_sac_count": 0,
    }
    if not event_path.exists():
        report["missing_marker"] = True
        return report

    marker_path = _resolved_stack_marker_path(event_path)
    report["missing_marker"] = not marker_path.exists()
    marker = {}
    raw_marker, marker_error = _load_json_file_with_error(_resolved_stack_marker_path(event_path), default=None)
    if marker_error:
        report["invalid_marker"] = marker_error
    elif raw_marker is not None:
        marker_payload_error = _invalid_json_payload_reason(raw_marker)
        if marker_payload_error:
            report["invalid_marker"] = marker_payload_error
        elif raw_marker:
            normalized_marker = _normalize_stack_event_marker_payload(event_path, raw_marker)
            marker_fields = _changed_top_level_fields(raw_marker, normalized_marker)
            if marker_fields:
                report["marker_needs_repair"] = True
                report["marker_repair_fields"] = marker_fields
                marker = normalized_marker
            else:
                marker = raw_marker
    elif report["missing_marker"] and _source_event_dir_from_stack_path(event_path) is not None:
        normalized_marker = _normalize_stack_event_marker_payload(event_path, {})
        report["marker_needs_repair"] = True
        report["marker_repair_fields"] = sorted(normalized_marker.keys())
        marker = normalized_marker
    if not marker:
        marker = load_stack_event_marker(event_path)

    sac_names = set()
    for sac_path in iter_stack_sac_paths(event_path):
        sac_names.add(stack_wave_name_from_path(event_path, sac_path))
        sidecar = stack_sidecar_path(sac_path, event_dir=event_path)
        if not sidecar.exists():
            report["missing_sidecars"].append(str(sac_path))
        try:
            obspy.read(str(sac_path))
            report["valid_sac_count"] += 1
        except Exception as exc:
            report["invalid_sac_files"].append({
                "path": str(sac_path),
                "reason": str(exc),
            })

    for sidecar_path in iter_stack_sidecar_paths(event_path):
        wave_name = _stack_wave_name_from_sidecar_path(sidecar_path, event_path)
        if wave_name not in sac_names:
            report["orphan_sidecars"].append(str(sidecar_path))
        raw_payload, sidecar_error = _load_json_file_with_error(sidecar_path, default=None)
        if sidecar_error:
            report["invalid_sidecars"].append({
                "path": str(sidecar_path),
                "wave_name": wave_name,
                "reason": sidecar_error,
            })
            continue
        payload_error = _invalid_json_payload_reason(raw_payload)
        if payload_error:
            report["invalid_sidecars"].append({
                "path": str(sidecar_path),
                "wave_name": wave_name,
                "reason": payload_error,
            })
            continue
        normalized_payload = _json_safe_payload(
            _stack_sidecar_storage_payload(
                _normalize_stack_sidecar_payload(event_path, sidecar_path, raw_payload, marker=marker)
            )
        )
        changed_fields = _changed_top_level_fields(raw_payload, normalized_payload)
        if changed_fields:
            report["sidecars_needing_repair"].append({
                "path": str(sidecar_path),
                "wave_name": wave_name,
                "fields": changed_fields,
            })

    return report


def quarantine_invalid_stack_files(
    event_dir: str | Path,
    *,
    persist: bool = False,
) -> dict:
    event_path = Path(event_dir).expanduser().resolve()
    health = inspect_stack_event_health(event_path)
    invalid_items = health.get("invalid_sac_files", [])
    timestamp = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    quarantine_dir = event_path / "_trash_invalid" / timestamp
    report = {
        "event_dir": str(event_path),
        "quarantine_dir": str(quarantine_dir),
        "moved": [],
        "sidecars_moved": [],
        "invalid_count": len(invalid_items),
    }
    if not invalid_items:
        return report

    for item in invalid_items:
        sac_path = Path(item["path"]).expanduser().resolve()
        sidecar_path = stack_sidecar_path(sac_path, event_dir=event_path)
        entry = {
            "path": str(sac_path),
            "reason": item.get("reason", ""),
        }
        report["moved"].append(entry)
        if sidecar_path.exists():
            report["sidecars_moved"].append(str(sidecar_path))
        if not persist:
            continue
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target_sac = quarantine_dir / sac_path.name
        shutil.move(str(sac_path), str(target_sac))
        if sidecar_path.exists():
            target_sidecar = quarantine_dir / sidecar_path.name
            shutil.move(str(sidecar_path), str(target_sidecar))
    if persist:
        write_stack_workspace_index(event_path)
    return report
