"""CLI entry point for the SFT Data Generator.

Usage:
    uv run main.py                          # Run all tasks, all pilot subjects
    uv run main.py --test                   # Test mode (1 sub_bab per subject)
    uv run main.py --task materi            # Only generate materi
    uv run main.py --task materi --task flashcard  # Multiple tasks
    uv run main.py --subject Sosiologi      # Only process Sosiologi
    uv run main.py --subject Sosiologi --subject "Bahasa Indonesia"
    uv run main.py --kelas "Kelas 12"       # Only Kelas 12
    uv run main.py --jenjang SMA            # Only SMA
    uv run main.py --kurikulum "K-13"       # Only K-13 curriculum
    uv run main.py --kurikulum KTSP --kurikulum "K-13"  # Multiple curricula
    uv run main.py --task materi --subject Sosiologi --kelas "Kelas 12" --test
    uv run main.py --dry-run                # Preview what would be generated
    uv run main.py --fail-fast-network      # Stop early if network/DNS is down
    uv run main.py --list-subjects         # Show available subjects
"""

from __future__ import annotations

import argparse
import json
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT Data Generator for Sekolah Rakyat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Task control ──
    parser.add_argument(
        "--task",
        action="append",
        choices=["materi", "flashcard", "mindmap", "pilgan", "essay", "pretest"],
        help="Task type(s) to generate. Repeatable. Default: all tasks.",
    )

    # ── Scope filters ──
    parser.add_argument(
        "--subject",
        action="append",
        help='Filter by mata_pelajaran. Repeatable. '
             'Example: --subject Sosiologi --subject "Bahasa Indonesia"',
    )
    parser.add_argument(
        "--kelas",
        action="append",
        help='Filter by kelas. Repeatable. '
             'Example: --kelas "Kelas 10" --kelas "Kelas 12"',
    )
    parser.add_argument(
        "--jenjang",
        action="append",
        help='Filter by jenjang. Repeatable. '
             'Example: --jenjang SMA',
    )
    parser.add_argument(
        "--variant",
        action="append",
        help='Filter by chunk variant. Repeatable. '
             'Example: --variant 1_chunks',
    )
    parser.add_argument(
        "--kurikulum",
        action="append",
        help='Filter by kurikulum. Repeatable. '
             'Example: --kurikulum "K-13" --kurikulum KTSP',
    )

    # ── Mode flags ──
    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode: only process 1 sub_bab per subject.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview the generation plan without making API calls.",
    )
    parser.add_argument(
        "--fail-fast-network",
        action="store_true",
        default=False,
        help="Stop early when network/DNS errors occur (after retries).",
    )
    parser.add_argument(
        "--list-subjects",
        action="store_true",
        default=False,
        help="List all available subjects from metadata and exit.",
    )
    return parser.parse_args()


def _build_filters(args: argparse.Namespace) -> dict:
    """Build a filters dict from CLI args."""
    filters = {}
    if args.subject:
        filters["subject"] = args.subject
    if args.kelas:
        filters["kelas"] = args.kelas
    if args.jenjang:
        filters["jenjang"] = args.jenjang
    if args.variant:
        filters["variant"] = args.variant
    if args.kurikulum:
        filters["kurikulum"] = args.kurikulum
    return filters or None


def list_subjects() -> None:
    """Show all available subjects from the metadata directory."""
    from src.metadata_loader import discover_metadata_files, load_metadata

    files = discover_metadata_files()

    # Collect unique (kurikulum, jenjang, kelas, mapel) combos
    rows: list[tuple[str, str, str, str, int]] = []
    seen: dict[tuple[str, str, str, str], int] = {}

    for path in files:
        entries = load_metadata(path)
        for entry in entries:
            kurikulum = entry.get("kurikulum", "?")
            jenjang = entry.get("jenjang", "?")
            kelas = entry.get("kelas", "?")
            mapel = entry.get("mata_pelajaran", "?")
            key = (kurikulum, jenjang, kelas, mapel)
            seen[key] = seen.get(key, 0) + 1

    table = Table(title="Available Subjects")
    table.add_column("#", style="dim")
    table.add_column("Kurikulum", style="bright_blue")
    table.add_column("Jenjang", style="magenta")
    table.add_column("Kelas", style="green")
    table.add_column("Mata Pelajaran", style="cyan")
    table.add_column("Sub Bab", style="dim", justify="right")

    for i, ((kurikulum, jenjang, kelas, mapel), count) in enumerate(
        sorted(seen.items()), 1
    ):
        table.add_row(str(i), kurikulum, jenjang, kelas, mapel, str(count))

    console.print(table)
    console.print(
        "\n[dim]Filter examples:[/]\n"
        '  --subject Sosiologi\n'
        '  --kelas "Kelas 12"\n'
        '  --jenjang SMA\n'
        '  --kurikulum "K-13"\n'
        '  --kurikulum KTSP --subject IPS\n'
        '  --subject Sosiologi --kelas "Kelas 12"'
    )


def dry_run(
    tasks: list[str] | None,
    filters: dict | None,
    test_mode: bool,
) -> None:
    """Show what would be generated without calling the API."""
    from src.config import LEVELS, TASK_TYPES, TASKS_WITHOUT_LEVELING
    from src.metadata_loader import iter_metadata
    from src.models import MODEL, TEMPERATURE, MAX_TOKENS

    tasks_to_run = tasks or TASK_TYPES

    console.print(
        Panel(
            "[bold yellow]🔍 DRY RUN — No API calls will be made[/]",
            style="yellow",
            expand=False,
        )
    )

    # Show config
    console.print(f"\n[bold]Model:[/] {MODEL}")
    console.print(f"[bold]Temperature:[/] {TEMPERATURE}")
    console.print(f"[bold]Max tokens:[/] {MAX_TOKENS}")
    console.print(f"[bold]Test mode:[/] {'ON' if test_mode else 'OFF'}")
    if filters:
        for key, vals in filters.items():
            console.print(f"[bold]Filter {key}:[/] {', '.join(vals)}")
    else:
        console.print("[bold]Filters:[/] pilot subjects (from config)")
    console.print()

    entries = list(iter_metadata(
        test_mode=test_mode,
        filters=filters,
    ))

    table = Table(title="Generation Plan")
    table.add_column("Task", style="cyan")
    table.add_column("Level", style="green")
    table.add_column("Kurikulum", style="bright_blue")
    table.add_column("Jenjang", style="magenta")
    table.add_column("Kelas", style="dim")
    table.add_column("Subject", style="yellow")
    table.add_column("Sub Bab", style="white")

    total_calls = 0

    for task in tasks_to_run:
        needs_leveling = task not in TASKS_WITHOUT_LEVELING
        levels = LEVELS if needs_leveling else [None]

        for level in levels:
            for _path, entry in entries:
                table.add_row(
                    task,
                    level or "—",
                    entry.get("kurikulum", ""),
                    entry.get("jenjang", ""),
                    entry.get("kelas", ""),
                    entry.get("mata_pelajaran", ""),
                    entry.get("sub_bab", ""),
                )
                total_calls += 1

    console.print(table)
    console.print(
        f"\n[bold]Total API calls:[/] {total_calls}"
    )


def main() -> None:
    args = parse_args()

    console.print(
        Panel(
            "[bold]🏫 Sekolah Rakyat — SFT Data Generator[/]",
            style="bright_blue",
            expand=False,
        )
    )

    if args.list_subjects:
        list_subjects()
        return

    filters = _build_filters(args)

    if args.dry_run:
        dry_run(args.task, filters, args.test)
        return

    # Import pipeline here (after args parsing so --help is fast)
    from src.pipeline import NetworkAbort, run_pipeline

    try:
        run_pipeline(
            tasks=args.task,
            filters=filters,
            test_mode=args.test,
            fail_fast_network=args.fail_fast_network,
        )
    except NetworkAbort as exc:
        console.print(
            Panel(
                f"[bold red]Aborted due to network error[/]\n{exc}",
                style="red",
                expand=False,
            )
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
