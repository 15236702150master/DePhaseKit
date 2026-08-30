from __future__ import annotations

import argparse
import json
from pathlib import Path

from stack_system import (
    build_stack_workspace_index,
    inspect_stack_event_health,
    quarantine_invalid_stack_files,
    repair_stack_event_metadata,
    write_stack_workspace_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Repair legacy stack event metadata under data/stack."
    )
    parser.add_argument("stack_event_dir", help="Stack event directory, e.g. data/stack/<dataset>/<event>")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without rewriting files",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Only inspect stack event health without repairing metadata",
    )
    parser.add_argument(
        "--quarantine-invalid",
        action="store_true",
        help="Move invalid stack SAC files and their sidecars into a timestamped quarantine directory",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Print the stack workspace index without writing it",
    )
    parser.add_argument(
        "--refresh-index",
        action="store_true",
        help="Rewrite .stack_index.json for this stack workspace",
    )
    args = parser.parse_args()

    event_dir = Path(args.stack_event_dir).expanduser().resolve()
    if args.health:
        report = inspect_stack_event_health(event_dir)
    elif args.index:
        report = build_stack_workspace_index(event_dir)
    elif args.refresh_index:
        report = write_stack_workspace_index(event_dir)
    elif args.quarantine_invalid:
        report = quarantine_invalid_stack_files(event_dir, persist=not args.dry_run)
    else:
        report = repair_stack_event_metadata(event_dir, persist=not args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
