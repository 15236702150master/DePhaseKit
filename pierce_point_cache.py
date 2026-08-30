from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "output" / "pierce_points"
DEFAULT_INTERNAL_EXTRACT_SCRIPT = Path(__file__).resolve().parent / "pierce_point_extract.py"
DEFAULT_TAUP_BIN = PROJECT_ROOT / "opt" / "TauP-3.1.0" / "bin" / "taup"
SUPPORTED_PHASES = ("pP", "sP")
SUPPORTED_MODELS = ("prem", "iasp91")
DEFAULT_PIERCE_DEPTH_KM = 24.4
MIN_PIERCE_DECIMAL_PLACES = 4


@dataclass
class PiercePointRecord:
    wave_name: str
    longitude: float
    latitude: float


def _event_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def relative_event_path(event_dir: str | Path) -> Path:
    event_path = _event_path(event_dir)
    data_root = PROJECT_ROOT / "data"
    try:
        return event_path.relative_to(data_root)
    except ValueError:
        return Path(event_path.parent.name) / event_path.name


def output_event_dir(event_dir: str | Path, output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> Path:
    return Path(output_root).expanduser().resolve() / relative_event_path(event_dir)


def pierce_file_path(
    event_dir: str | Path,
    phase: str,
    model: str,
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    safe_model = str(model).strip().lower()
    return output_event_dir(event_dir, output_root) / (
        f"pierce_points_{phase}_{pierce_depth_km:.1f}km_{safe_model}.txt"
    )


def _decimal_places(token: str) -> int:
    text = str(token).strip()
    if "." not in text:
        return 0
    return len(text.rsplit(".", 1)[1].rstrip())


def _pierce_file_needs_refresh(file_path: Path) -> bool:
    if not file_path.exists():
        return True
    try:
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            lon_token = parts[1]
            lat_token = parts[2]
            return (
                _decimal_places(lon_token) < MIN_PIERCE_DECIMAL_PLACES
                or _decimal_places(lat_token) < MIN_PIERCE_DECIMAL_PLACES
            )
    except OSError:
        return True
    return False


def ensure_pierce_file(
    event_dir: str | Path,
    phase: str,
    model: str,
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    taup_bin: str | Path = DEFAULT_TAUP_BIN,
) -> Path:
    phase_key = str(phase).strip()
    model_key = str(model).strip().lower()
    output_path = pierce_file_path(
        event_dir=event_dir,
        phase=phase_key,
        model=model_key,
        pierce_depth_km=pierce_depth_km,
        output_root=output_root,
    )
    if output_path.exists() and not _pierce_file_needs_refresh(output_path):
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    script_path = DEFAULT_INTERNAL_EXTRACT_SCRIPT
    command = [
        sys.executable,
        str(script_path),
        "--event-dir",
        str(_event_path(event_dir)),
        "--output-file",
        str(output_path),
        "--taup-bin",
        str(Path(taup_bin).expanduser().resolve()),
        "--model",
        model_key,
        "--phase",
        phase_key,
        "--pierce-depth-km",
        f"{float(pierce_depth_km):.1f}",
    ]
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))

    legacy_path = output_event_dir(event_dir, output_root) / f"pierce_points_{phase_key}_{float(pierce_depth_km):.1f}km.txt"
    if output_path.exists():
        return output_path
    if legacy_path.exists():
        try:
            legacy_path.replace(output_path)
        except Exception:
            return legacy_path
        return output_path
    raise FileNotFoundError(f"Pierce-point file was not created: {output_path}")


def ensure_event_pierce_files(
    event_dir: str | Path,
    phases: tuple[str, ...] = SUPPORTED_PHASES,
    models: tuple[str, ...] = SUPPORTED_MODELS,
    pierce_depth_km: float = DEFAULT_PIERCE_DEPTH_KM,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    taup_bin: str | Path = DEFAULT_TAUP_BIN,
) -> dict[tuple[str, str], Path]:
    ensured: dict[tuple[str, str], Path] = {}
    for model in models:
        for phase in phases:
            ensured[(phase, model)] = ensure_pierce_file(
                event_dir=event_dir,
                phase=phase,
                model=model,
                pierce_depth_km=pierce_depth_km,
                output_root=output_root,
                taup_bin=taup_bin,
            )
    return ensured


def load_pierce_points(file_path: str | Path) -> dict[str, PiercePointRecord]:
    records: dict[str, PiercePointRecord] = {}
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            wave_name = parts[0]
            try:
                longitude = float(parts[1])
                latitude = float(parts[2])
            except ValueError:
                continue
            if not (math.isfinite(longitude) and math.isfinite(latitude)):
                continue
            records[wave_name] = PiercePointRecord(
                wave_name=wave_name,
                longitude=longitude,
                latitude=latitude,
            )
    return records
