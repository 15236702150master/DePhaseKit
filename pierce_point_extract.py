from __future__ import annotations

import argparse
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


SAC_FLOAT_INDICES = {
    "stla": 31,
    "stlo": 32,
    "evla": 35,
    "evlo": 36,
    "evdp": 38,
    "az": 51,
    "gcarc": 53,
}


@dataclass
class SacHeader:
    stla: float
    stlo: float
    evla: float
    evlo: float
    evdp: float
    az: float
    gcarc: float


@dataclass
class PiercePoint:
    filename: str
    longitude: float
    latitude: float
    distance_deg: float
    depth_km: float
    travel_time_s: float


# SAC binary waveforms often use channel-style extensions (.bhz/.shz/.euz/...)
# instead of .sac. Match all of these so DSM theoretical .bhz events get pierce
# points generated; the file is still SAC-format and read by struct.unpack.
SAC_WAVEFORM_EXTENSIONS = (
    ".sac", ".SAC",
    ".bhz", ".BHZ", ".bhe", ".BHE", ".bhn", ".BHN",
    ".shz", ".SHZ", ".she", ".SHE", ".shn", ".SHN",
    ".euz", ".EUZ", ".eue", ".EUE", ".eun", ".EUN",
    ".lhz", ".LHZ", ".hhz", ".HHZ",
)


def list_sac_files(event_dir: str | Path, suffix=None) -> list[Path]:
    event_path = Path(event_dir)
    if suffix:
        suffixes = (suffix,) if isinstance(suffix, str) else tuple(suffix)
        sac_files = []
        for ext in suffixes:
            if not ext.startswith("."):
                ext = "." + ext
            sac_files.extend(event_path.glob(f"*{ext}"))
    else:
        sac_files = []
        for ext in SAC_WAVEFORM_EXTENSIONS:
            sac_files.extend(event_path.glob(f"*{ext}"))
    return sorted(set(sac_files))


def _unpack_sac_floats(sac_file: Path, endian: str) -> tuple[float, ...]:
    with sac_file.open("rb") as handle:
        header_bytes = handle.read(70 * 4)
    return struct.unpack(f"{endian}70f", header_bytes)


def read_sac_header(sac_file: str | Path) -> SacHeader:
    sac_path = Path(sac_file)
    candidates = []

    for endian in ("<", ">"):
        values = _unpack_sac_floats(sac_path, endian)
        header = {key: values[index] for key, index in SAC_FLOAT_INDICES.items()}
        score = 0
        if -90 <= header["stla"] <= 90:
            score += 1
        if -180 <= header["stlo"] <= 180:
            score += 1
        if -90 <= header["evla"] <= 90:
            score += 1
        if -180 <= header["evlo"] <= 180:
            score += 1
        if 0 <= header["evdp"] <= 700:
            score += 1
        candidates.append((score, header))

    best_score, best_header = max(candidates, key=lambda item: item[0])
    if best_score < 4:
        raise ValueError(f"无法可靠解析 SAC 头段: {sac_path}")
    return SacHeader(**best_header)


def run_taup_pierce(
    taup_bin: str | Path,
    model: str,
    phase: str,
    source_depth_km: float,
    event_lat: float,
    event_lon: float,
    station_lat: float,
    station_lon: float,
    pierce_depth_km: float,
    geodetic: bool = True,
) -> str:
    command = [
        str(taup_bin),
        "pierce",
        "--mod",
        model,
        "-h",
        f"{source_depth_km:.3f}",
        "-p",
        phase,
        "--sta",
        f"{station_lat:.6f}",
        f"{station_lon:.6f}",
        "--evt",
        f"{event_lat:.6f}",
        f"{event_lon:.6f}",
        "--pierce",
        f"{pierce_depth_km:.3f}",
        "--nodiscon",
        "--text",
    ]
    if geodetic:
        command.insert(2, "--geodetic")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "TauP 命令执行失败")
    return result.stdout


def parse_source_side_pierce_point(
    taup_output: str,
    target_depth_km: float,
    filename: str,
    depth_tolerance_km: float = 0.2,
) -> PiercePoint:
    for raw_line in taup_output.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(">") or line.startswith("["):
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            distance_deg = float(parts[0])
            depth_km = float(parts[1])
            travel_time_s = float(parts[2])
            latitude = float(parts[3])
            longitude = float(parts[4])
        except ValueError:
            continue

        if abs(depth_km - target_depth_km) <= depth_tolerance_km:
            return PiercePoint(
                filename=filename,
                longitude=longitude,
                latitude=latitude,
                distance_deg=distance_deg,
                depth_km=depth_km,
                travel_time_s=travel_time_s,
            )
    raise ValueError(f"未在 TauP 输出中找到 {target_depth_km} km 的震源侧穿透点")


def extract_event_pierce_points(
    event_dir: str | Path,
    taup_bin: str | Path,
    model: str = "prem",
    phase: str = "pP",
    pierce_depth_km: float = 24.4,
    geodetic: bool = True,
    min_gcarc_deg: float | None = None,
    max_gcarc_deg: float | None = None,
    min_az_deg: float | None = None,
    max_az_deg: float | None = None,
    suffix=None,
) -> tuple[str, list[PiercePoint]]:
    event_path = Path(event_dir)
    event_name = event_path.name
    results: list[PiercePoint] = []

    for sac_file in list_sac_files(event_path, suffix=suffix):
        header = read_sac_header(sac_file)

        if min_gcarc_deg is not None and header.gcarc < min_gcarc_deg:
            continue
        if max_gcarc_deg is not None and header.gcarc > max_gcarc_deg:
            continue
        if min_az_deg is not None and header.az < min_az_deg:
            continue
        if max_az_deg is not None and header.az > max_az_deg:
            continue
        if header.evdp <= pierce_depth_km:
            continue

        try:
            taup_output = run_taup_pierce(
                taup_bin=taup_bin,
                model=model,
                phase=phase,
                source_depth_km=header.evdp,
                event_lat=header.evla,
                event_lon=header.evlo,
                station_lat=header.stla,
                station_lon=header.stlo,
                pierce_depth_km=pierce_depth_km,
                geodetic=geodetic,
            )
            point = parse_source_side_pierce_point(
                taup_output=taup_output,
                target_depth_km=pierce_depth_km,
                filename=sac_file.name,
            )
            results.append(point)
        except Exception:
            continue
    return event_name, results


def calculate_average_coordinates(points: list[PiercePoint]) -> tuple[float | None, float | None]:
    if not points:
        return None, None
    avg_lon = sum(point.longitude for point in points) / len(points)
    avg_lat = sum(point.latitude for point in points) / len(points)
    return avg_lon, avg_lat


def write_event_output(
    output_file: str | Path,
    event_name: str,
    points: list[PiercePoint],
    model: str,
    phase: str,
    pierce_depth_km: float,
    min_gcarc_deg: float | None = None,
    max_gcarc_deg: float | None = None,
    min_az_deg: float | None = None,
    max_az_deg: float | None = None,
) -> tuple[float | None, float | None]:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    avg_lon, avg_lat = calculate_average_coordinates(points)

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# 事件: {event_name}\n")
        handle.write(f"# 模型: {model}\n")
        handle.write(f"# 震相: {phase}\n")
        handle.write(f"# 目标深度: {pierce_depth_km:.1f} km\n")
        handle.write(f"# 震中距筛选范围: {min_gcarc_deg} ~ {max_gcarc_deg} deg\n")
        handle.write(f"# 方位角筛选范围: {min_az_deg} ~ {max_az_deg} deg\n")
        handle.write(f"# 规则: 取 {phase} 在震源侧首次到达目标深度的穿透点\n")
        handle.write(f"# 有效台站数: {len(points)}\n")
        if avg_lon is not None and avg_lat is not None:
            handle.write(f"# 平均经度: {avg_lon:.6f}\n")
            handle.write(f"# 平均纬度: {avg_lat:.6f}\n")
        handle.write("# 格式: 文件名 经度 纬度\n")
        for point in points:
            handle.write(f"{point.filename} {point.longitude:.6f} {point.latitude:.6f}\n")
    return avg_lon, avg_lat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="提取单个事件的穿透点")
    parser.add_argument("--event-dir", required=True, help="事件目录")
    parser.add_argument("--output-file", required=True, help="输出文件")
    parser.add_argument("--taup-bin", required=True, help="TauP 可执行程序路径")
    parser.add_argument("--model", default="prem", help="TauP 模型")
    parser.add_argument("--phase", default="pP", help="震相")
    parser.add_argument("--pierce-depth-km", type=float, default=24.4, help="穿透点目标深度")
    parser.add_argument("--planar", action="store_true", help="关闭 geodetic 模式")
    parser.add_argument("--min-gcarc-deg", type=float, default=None, help="最小震中距")
    parser.add_argument("--max-gcarc-deg", type=float, default=None, help="最大震中距")
    parser.add_argument("--min-az-deg", type=float, default=None, help="最小方位角")
    parser.add_argument("--max-az-deg", type=float, default=None, help="最大方位角")
    parser.add_argument("--suffix", default=None, help="波形文件后缀(如 .bhz)；省略则匹配所有 SAC 波形后缀")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    event_name, points = extract_event_pierce_points(
        event_dir=args.event_dir,
        taup_bin=args.taup_bin,
        model=args.model,
        phase=args.phase,
        pierce_depth_km=args.pierce_depth_km,
        geodetic=not args.planar,
        min_gcarc_deg=args.min_gcarc_deg,
        max_gcarc_deg=args.max_gcarc_deg,
        min_az_deg=args.min_az_deg,
        max_az_deg=args.max_az_deg,
        suffix=args.suffix,
    )
    write_event_output(
        output_file=args.output_file,
        event_name=event_name,
        points=points,
        model=args.model,
        phase=args.phase,
        pierce_depth_km=args.pierce_depth_km,
        min_gcarc_deg=args.min_gcarc_deg,
        max_gcarc_deg=args.max_gcarc_deg,
        min_az_deg=args.min_az_deg,
        max_az_deg=args.max_az_deg,
    )


if __name__ == "__main__":
    main()
