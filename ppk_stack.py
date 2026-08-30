import argparse
import json
import sys
from os.path import exists

from PySide6.QtWidgets import QApplication

from ppk import MatplotlibWidget
from stack_system import (
    build_stack_workspace_index,
    build_stack_workspace_manifest,
    ensure_stack_workspace_dir,
    inspect_stack_event_health,
    quarantine_invalid_stack_files,
    repair_stack_event_metadata,
    resolve_stack_workspace_dir,
    stack_sac_time_window,
    write_stack_workspace_index,
)


def _default_xlim_for_marker(marker):
    marker = str(marker or 't0')
    if marker in ('t0', 't7'):
        return [-10, 70]
    if marker in ('t2', 't6'):
        return [-40, 30]
    if marker in ('t3', 't5'):
        return [-50, 20]
    return [-10, 10]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Open the DePhaseKit stack workspace for an event or stack directory",
    )
    parser.add_argument('event_path', type=str, help='Source event directory or stack workspace directory')
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
    parser.add_argument('--health', action='store_true',
                        help='Print stack workspace health as JSON and exit without opening the GUI')
    parser.add_argument('--manifest', action='store_true',
                        help='Print stack workspace manifest as JSON and exit without opening the GUI')
    parser.add_argument('--index', action='store_true',
                        help='Print stack workspace index as JSON and exit without opening the GUI')
    parser.add_argument('--refresh-index', action='store_true',
                        help='Rewrite .stack_index.json for the stack workspace')
    parser.add_argument('--init', action='store_true',
                        help='Create the stack workspace marker/directory for the source event, print JSON, and exit')
    parser.add_argument('--repair-metadata', action='store_true',
                        help='Normalize stack marker and sidecar metadata before opening the GUI')
    parser.add_argument('--quarantine-invalid', action='store_true',
                        help='Move invalid stack SAC files and sidecars into the workspace quarantine directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report repair/quarantine actions without writing changes; implies JSON report output')
    return parser.parse_args(argv)


def resolve_launcher_config(arg):
    input_path = arg.event_path
    if not exists(input_path):
        raise FileNotFoundError(f'No such directory: {input_path}')
    if getattr(arg, 'init', False):
        wavepath = str(ensure_stack_workspace_dir(input_path))
    else:
        wavepath = str(resolve_stack_workspace_dir(input_path))
    if not exists(wavepath):
        raise FileNotFoundError(f'No stack workspace available yet: {wavepath}')

    if arg.xlim:
        xlim = arg.xlim
    else:
        stack_xlim = stack_sac_time_window(wavepath, suffix=arg.suffix)
        xlim = stack_xlim if stack_xlim is not None else _default_xlim_for_marker(arg.tmarker)

    preview_phases = [item.strip() for item in arg.ta_tb.split(',') if item.strip()]
    if len(preview_phases) == 0:
        raise ValueError('At least one preview phase must be provided')
    if arg.xlim_preview is not None and len(arg.xlim_preview) != len(preview_phases) * 2:
        raise ValueError('Preview window count must match preview phase count')

    return {
        'wavepath': wavepath,
        'xlim': xlim,
        'order': arg.order,
        'tmarker': arg.tmarker,
        'suffix': arg.suffix,
        'ta_tb': arg.ta_tb,
        'xlim_preview': arg.xlim_preview,
    }


def stack_workspace_maintenance_report(arg, config=None):
    config = config or resolve_launcher_config(arg)
    wavepath = config['wavepath']
    persist = not getattr(arg, 'dry_run', False)
    report = {
        'wavepath': wavepath,
        'dry_run': not persist,
    }
    if getattr(arg, 'repair_metadata', False):
        report['repair_metadata'] = repair_stack_event_metadata(wavepath, persist=persist)
    if getattr(arg, 'quarantine_invalid', False):
        report['quarantine_invalid'] = quarantine_invalid_stack_files(wavepath, persist=persist)
    report['health'] = inspect_stack_event_health(wavepath)
    report['manifest'] = build_stack_workspace_manifest(wavepath)
    if getattr(arg, 'refresh_index', False) or getattr(arg, 'init', False):
        report['index'] = write_stack_workspace_index(wavepath)
    elif getattr(arg, 'index', False):
        report['index'] = build_stack_workspace_index(wavepath)
    return report


def stack_workspace_open_report(config):
    wavepath = config['wavepath']
    health = inspect_stack_event_health(wavepath)
    valid_count = int(health.get('valid_sac_count', 0) or 0)
    invalid_count = len(health.get('invalid_sac_files', []) or [])
    can_open = valid_count > 0
    message = 'Stack workspace is ready'
    if not can_open:
        if invalid_count:
            message = 'Stack workspace has no valid stack SAC files'
        else:
            message = 'Stack workspace has no stack SAC files yet'
    return {
        'wavepath': wavepath,
        'can_open_gui': can_open,
        'message': message,
        'health': health,
        'manifest': build_stack_workspace_manifest(wavepath),
    }


def main(argv=None):
    arg = parse_args(argv)
    config = resolve_launcher_config(arg)
    if (
        arg.init
        or arg.health
        or arg.manifest
        or arg.index
        or arg.refresh_index
        or arg.repair_metadata
        or arg.quarantine_invalid
        or arg.dry_run
    ):
        report = stack_workspace_maintenance_report(arg, config=config)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if arg.init or arg.health or arg.manifest or arg.index or arg.refresh_index or arg.dry_run:
            return 0
    open_report = stack_workspace_open_report(config)
    if not open_report['can_open_gui']:
        print(json.dumps(open_report, ensure_ascii=False, indent=2))
        return 2
    app = QApplication(sys.argv if argv is None else ['ppk_stack'])
    ui = MatplotlibWidget(
        config['wavepath'],
        xlim=config['xlim'],
        order=config['order'],
        tmarker=config['tmarker'],
        suffix=config['suffix'],
        ta_tb=config['ta_tb'],
        xlim_preview=config['xlim_preview'],
    )
    sys.exit(app.exec())


if __name__ == '__main__':
    sys.exit(main())
