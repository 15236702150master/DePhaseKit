#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone linear-stack + RMS-normalization script.

Extracted from the DePhaseKit project (WaveFigure.py). Given a directory of SAC
files, it aligns every trace on a SAC t-marker (default t0), extracts a common
time window, normalizes each trace by its RMS amplitude, and stacks them with a
plain linear (mean) stack. Results are written as a SAC trace, a two-column
text file, a preview PNG, a member-status list, and a JSON metadata file.

Only the linear-stack and RMS logic was carried over; bandpass, moveout,
phase-weighted stacking, pierce-point and crustal-thickness machinery from the
original project are intentionally omitted.

Usage:
    python stack_linear_rms.py <sac_dir> [-t t0] [-x xmin xmax] [-s .sac] [-o output_dir]
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import obspy


# --------------------------------------------------------------------------- #
# SAC header helpers (mirror WaveFigure._sac_attr / _sac_float / _safe_float) #
# --------------------------------------------------------------------------- #
def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


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


def _sac_float(trace, attr, default=math.nan):
    return _safe_float(_sac_attr(trace, attr, default))


def _normalize_marker_key(marker_key):
    marker_key = str(marker_key or '')
    if marker_key.startswith('t'):
        marker_key = marker_key[1:]
    return marker_key


# --------------------------------------------------------------------------- #
# Alignment + windowing (mirror EvtData)                                      #
# --------------------------------------------------------------------------- #
def extract_window(trace_data, start_index, end_index, wavelength):
    """Zero-padded slice of trace_data[start_index:end_index]."""
    window = np.zeros(wavelength, dtype=float)
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


def build_data_matrix(traces, reference_times, x1, x2, dt):
    """Build the [n_station, n_sample] windowed data matrix aligned on markers."""
    wavelength = max(1, int(round((x2 - x1) / dt)))
    n = len(traces)
    data = np.empty([n, wavelength], dtype=float)
    for i in range(n):
        b = _sac_float(traces[i], 'b', 0.0)
        t1_index = int(round((reference_times[i] - b) / dt))
        start_index = t1_index + int(round(x1 / dt))
        end_index = start_index + wavelength
        data[i] = extract_window(traces[i].data, start_index, end_index, wavelength)
    time_axis = x1 + np.arange(wavelength, dtype=float) * dt
    return data, time_axis, wavelength


# --------------------------------------------------------------------------- #
# Normalization + linear stack (mirror _preview_stack_normalize_rows /        #
# _compute_preview_linear_stack)                                              #
# --------------------------------------------------------------------------- #
def normalize_rows(data_rows):
    """RMS normalization. Returns (valid_rows, valid_mask, reasons)."""
    rows = np.asarray(data_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] == 0:
        return rows, np.asarray([], dtype=bool), []
    normalized_rows = rows.copy()
    valid_mask = np.all(np.isfinite(normalized_rows), axis=1)
    skipped_reasons = []
    for row_index, row in enumerate(normalized_rows):
        if not valid_mask[row_index]:
            skipped_reasons.append((row_index, 'non-finite data'))
            continue
        scale = float(np.sqrt(np.mean(np.square(row))))  # RMS
        if not np.isfinite(scale) or scale <= 0.0:
            valid_mask[row_index] = False
            skipped_reasons.append((row_index, 'zero rms scale'))
            continue
        normalized_rows[row_index] = row / scale
    return normalized_rows[valid_mask], valid_mask, skipped_reasons


def linear_stack(normalized_rows):
    """Plain linear (mean) stack of the normalized rows."""
    if normalized_rows.size == 0:
        return None
    return np.mean(normalized_rows, axis=0)


# --------------------------------------------------------------------------- #
# SAC output (mirror _write_preview_stack_sac)                                 #
# --------------------------------------------------------------------------- #
def write_stack_sac(output_path, template_trace, stack_data, dt, x1, x2,
                    align_marker, align_time, sta_num):
    stack_trace = template_trace.copy()
    stack_trace.data = np.asarray(stack_data, dtype=np.float32)
    stack_trace.stats.npts = len(stack_trace.data)
    stack_trace.stats.delta = float(dt)
    stack_trace.stats.sampling_rate = 1.0 / float(dt)
    try:
        stack_trace.stats.network = 'DPK'
        stack_trace.stats.station = 'STACK'
    except Exception:
        pass

    if hasattr(stack_trace.stats, 'sac'):
        window_length = float(x2 - x1)
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
        sac.user0 = float(sta_num)
        sac.kstnm = 'STACK'
        sac.knetwk = 'DPK'
        for marker_attr in ('t0', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9'):
            setattr(sac, marker_attr, math.nan)
        normalized_align = _normalize_marker_key(align_marker)
        if normalized_align is not None and normalized_align.isdigit():
            setattr(sac, f't{normalized_align}', float(-x1))

    output_path = Path(output_path)
    temp_path = output_path.with_name(f'.{output_path.name}.tmp')
    stack_trace.write(str(temp_path), format='SAC')
    os.replace(str(temp_path), str(output_path))


# --------------------------------------------------------------------------- #
# Default window per alignment marker (mirror _default_xlim_for_marker)       #
# --------------------------------------------------------------------------- #
def default_xlim_for_marker(marker):
    marker = _normalize_marker_key(marker or 't0')
    marker = f't{marker}' if marker != '' else 't0'
    if marker in ('t0', 't7'):
        return [-10.0, 70.0]
    if marker in ('t2', 't6'):
        return [-40.0, 30.0]
    if marker in ('t3', 't5'):
        return [-50.0, 20.0]
    return [-10.0, 10.0]


# --------------------------------------------------------------------------- #
# Main pipeline                                                               #
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Linear stack with RMS normalization over SAC files in a directory',
    )
    parser.add_argument('sac_dir', type=str, help='Directory containing SAC files')
    parser.add_argument('-t', dest='tmarker', type=str, default='t0',
                        help='Alignment SAC t-marker, defaults t0')
    parser.add_argument('-x', dest='xlim', default=None, nargs=2, type=float,
                        metavar=('xmin', 'xmax'),
                        help='Time window (s) relative to the alignment marker; '
                             'if omitted, use defaults for the selected marker')
    parser.add_argument('-s', dest='suffix', type=str, default='.sac',
                        help='SAC file suffix, defaults .sac')
    parser.add_argument('-o', dest='output_dir', type=str, default=None,
                        help='Output directory; defaults to <sac_dir>/stack_output')
    return parser.parse_args(argv)


def collect_sac_files(sac_dir, suffix):
    paths = []
    for name in sorted(os.listdir(sac_dir)):
        if name.lower().endswith(suffix.lower()):
            paths.append(os.path.join(sac_dir, name))
    return paths


def load_traces(sac_files):
    traces = []
    skipped = []
    for path in sac_files:
        try:
            st = obspy.read(path)
        except Exception as exc:
            skipped.append({'file': os.path.basename(path), 'reason': f'read error: {exc}'})
            continue
        if len(st) == 0:
            skipped.append({'file': os.path.basename(path), 'reason': 'empty stream'})
            continue
        tr = st[0]
        tr.stats.dephasekit_wave_name = os.path.basename(path)
        traces.append(tr)
    return traces, skipped


def _wave_name(tr):
    return getattr(tr.stats, 'dephasekit_wave_name', None) or tr.id


def main(argv=None):
    arg = parse_args(argv)
    sac_dir = arg.sac_dir
    if not os.path.isdir(sac_dir):
        print(f'No such directory: {sac_dir}', file=sys.stderr)
        return 2

    align_marker = _normalize_marker_key(arg.tmarker)
    x1, x2 = arg.xlim if arg.xlim else default_xlim_for_marker(arg.tmarker)
    if x2 <= x1:
        print(f'Invalid window: x2 ({x2}) must be greater than x1 ({x1})', file=sys.stderr)
        return 2

    sac_files = collect_sac_files(sac_dir, arg.suffix)
    if not sac_files:
        print(f'No SAC files (*{arg.suffix}) found in {sac_dir}', file=sys.stderr)
        return 2

    traces, skipped_reads = load_traces(sac_files)
    if not traces:
        print('No readable SAC traces.', file=sys.stderr)
        return 2

    # Common sampling: resample to the median delta (mirror WaveFigure self.dt).
    dt = float(np.median([tr.stats.delta for tr in traces]))
    target_fs = 1.0 / dt
    for tr in traces:
        if abs(tr.stats.sampling_rate - target_fs) > 1e-3:
            tr.resample(target_fs, window='hann')

    # Alignment reference time = SAC t{marker}.
    reference_times = []
    active_traces = []
    skipped_missing = []
    for tr in traces:
        ref = _sac_float(tr, f't{align_marker}', math.nan)
        if math.isnan(ref):
            skipped_missing.append({
                'file': _wave_name(tr),
                'reason': f'missing t{align_marker}',
            })
            continue
        tr.data = np.asarray(tr.data, dtype=float)
        reference_times.append(ref)
        active_traces.append(tr)

    if not active_traces:
        print(f'No waveforms have a t{align_marker} reference time to align on.',
              file=sys.stderr)
        return 2

    data, time_axis, wavelength = build_data_matrix(
        active_traces, reference_times, x1, x2, dt,
    )
    normalized_rows, valid_mask, skipped_reasons = normalize_rows(data)
    stack_data = linear_stack(normalized_rows)
    if stack_data is None:
        print('No valid waveforms remained after normalization.', file=sys.stderr)
        return 2

    valid_indices = np.flatnonzero(valid_mask)
    stack_align_time = float(np.mean([reference_times[i] for i in valid_indices])) \
        if len(valid_indices) else math.nan

    # Output package.
    output_root = arg.output_dir or os.path.join(sac_dir, 'stack_output')
    timestamp_tag = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    scope = os.path.basename(os.path.abspath(sac_dir.rstrip('/'))) or 'dir'
    basename = f'stack_t{align_marker}_{scope}_linear_rms_{timestamp_tag}'
    package_dir = os.path.join(output_root, basename)
    os.makedirs(package_dir, exist_ok=True)
    txt_path = os.path.join(package_dir, 'stack.txt')
    sac_path = os.path.join(package_dir, 'stack.sac')
    png_path = os.path.join(package_dir, 'preview.png')
    json_path = os.path.join(package_dir, 'meta.json')
    members_path = os.path.join(package_dir, 'members.txt')

    np.savetxt(
        txt_path,
        np.column_stack([time_axis, np.asarray(stack_data, dtype=float)]),
        fmt='%.8g',
        header='time_after_align_s stack_amplitude',
    )
    write_stack_sac(
        sac_path, active_traces[0], stack_data, dt, x1, x2,
        align_marker, stack_align_time, len(active_traces),
    )

    # Member status list.
    member_lines = ['wave_name\tstatus\tdetail']
    for idx, tr in enumerate(active_traces):
        status = 'used' if valid_mask[idx] else 'skipped_normalization'
        detail = ''
        for row_index, reason in skipped_reasons:
            if row_index == idx:
                detail = reason
                break
        member_lines.append(f'{_wave_name(tr)}\t{status}\t{detail}')
    with open(members_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(member_lines) + '\n')

    # Metadata.
    first = active_traces[0]
    metadata = {
        'sac_dir': os.path.abspath(sac_dir),
        'created_at': obspy.UTCDateTime().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'align_marker': f't{align_marker}',
        'x1': x1,
        'x2': x2,
        'dt': dt,
        'normalize': 'rms',
        'stack_type': 'linear',
        'wave_count_requested': len(traces),
        'wave_count_aligned': len(active_traces),
        'wave_count_used': int(len(valid_indices)),
        'skipped_reads': skipped_reads,
        'skipped_missing_reference': skipped_missing,
        'skipped_normalization': [
            {'wave_name': _wave_name(active_traces[row_index]), 'reason': reason}
            for row_index, reason in skipped_reasons
            if row_index < len(active_traces)
        ],
        'stack_align_time': stack_align_time,
        'event_info': {
            'nzyear': int(_safe_float(_sac_attr(first, 'nzyear', 0))),
            'nzjday': int(_safe_float(_sac_attr(first, 'nzjday', 0))),
            'nzhour': int(_safe_float(_sac_attr(first, 'nzhour', 0))),
            'nzmin': int(_safe_float(_sac_attr(first, 'nzmin', 0))),
            'nzsec': int(_safe_float(_sac_attr(first, 'nzsec', 0))),
            'evla': _sac_float(first, 'evla', 0.0),
            'evlo': _sac_float(first, 'evlo', 0.0),
            'evdp': _sac_float(first, 'evdp', 0.0),
        },
        'outputs': {
            'txt': txt_path,
            'sac': sac_path,
            'png': png_path,
            'json': json_path,
            'members': members_path,
        },
    }
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, ensure_ascii=False, indent=2, fp=handle)

    # Preview PNG.
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.5, 5.4))
        if len(normalized_rows):
            alpha = max(0.08, min(0.28, 8.0 / max(len(normalized_rows), 1)))
            for row in normalized_rows:
                ax.plot(time_axis, row, color='#9aa0a6', linewidth=0.45, alpha=alpha)
        ax.plot(time_axis, stack_data, color='#c62828', linewidth=1.8, label='Linear stack')
        ax.axvline(0.0, color='black', linewidth=0.8)
        ax.grid(color='#cccccc', linestyle='--', linewidth=0.45, alpha=0.7)
        ax.set_xlim(x1, x2)
        ax.set_xlabel(f'Time after t{align_marker} (s)')
        ax.set_ylabel('Normalized amplitude')
        ax.set_title(
            f'{scope} | t{align_marker} | linear | rms | '
            f'N={len(valid_indices)}',
            fontsize=11,
        )
        ax.legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as exc:
        print(f'Warning: failed to write preview PNG: {exc}', file=sys.stderr)

    print(json.dumps({
        'package_dir': package_dir,
        'wave_count_requested': len(traces),
        'wave_count_aligned': len(active_traces),
        'wave_count_used': int(len(valid_indices)),
        'outputs': metadata['outputs'],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
