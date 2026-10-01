"""Command-line entry point: `uv run lja <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable

from llm_judge_audit.config import load_config


def _cmd_download(args: argparse.Namespace) -> int:
    from llm_judge_audit.data.download import download

    print(json.dumps(download(load_config(), force=args.force), indent=2))
    return 0


def _cmd_rebuild_data(args: argparse.Namespace) -> int:
    from llm_judge_audit.data import rebuild

    cfg = load_config()
    if args.check:
        rebuild.check(cfg)
        print("OK: derived artifacts are byte-identical to a fresh rebuild")
        return 0
    hashes = rebuild.rebuild(cfg)
    print(json.dumps(hashes, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lja", description="LLM Judge Audit")
    sub = parser.add_subparsers(dest="command", required=True)
    commands: dict[str, tuple[str, Callable[[argparse.Namespace], int]]] = {
        "download": ("download pinned raw sources and verify checksums", _cmd_download),
        "rebuild-data": (
            "deterministically rebuild derived data from pinned raw data",
            _cmd_rebuild_data,
        ),
    }
    for name, (help_text, func) in commands.items():
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        if name == "download":
            p.add_argument("--force", action="store_true", help="re-download even if present")
        if name == "rebuild-data":
            p.add_argument(
                "--check",
                action="store_true",
                help="verify on-disk artifacts equal a fresh rebuild; write nothing",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
