from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UMAP Atlas pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Execute full pipeline")
    p_run.add_argument("-c", "--config", type=Path, default=Path("configs/base.yaml"))

    p_val = sub.add_parser("validate-config", help="Validate configuration file")
    p_val.add_argument("-c", "--config", type=Path, default=Path("configs/base.yaml"))

    args = parser.parse_args()

    try:
        if args.command == "run":
            rows, data_path, report_path = run_pipeline(args.config)
            print(f"Rows: {len(rows)}")
            print(f"Data: {data_path}")
            print(f"Report: {report_path}")
        elif args.command == "validate-config":
            cfg = load_config(args.config)
            print(f"Config OK: {args.config} (hash={cfg.config_hash})")
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
