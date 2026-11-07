"""Command-line entry point for staged Prompt 8 preparation."""

from __future__ import annotations

import argparse
import json

from benchmarks.config import BenchmarkConfig
from benchmarks.data import refresh_distribution_statistics
from benchmarks.workflow import prepare_benchmark_data, train_models_and_calibrate
from benchmarks.runner import run_benchmark_searches


def _config(path: str | None) -> BenchmarkConfig:
    return BenchmarkConfig.load(path) if path else BenchmarkConfig()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Prepare approved Prompt 8 data and model artifacts in stages.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    write_config = commands.add_parser("write-config", help="write the approved default config")
    write_config.add_argument("--output", required=True)

    prepare = commands.add_parser("prepare-data", help="generate and validate split archives")
    prepare.add_argument("--config")
    prepare.add_argument("--overwrite", action="store_true")

    train = commands.add_parser("train-models", help="train models and calibrate D95")
    train.add_argument("--config")
    train.add_argument("--overwrite", action="store_true")

    refresh = commands.add_parser(
        "refresh-stats", help="recompute structural diagnostics from saved map archives"
    )
    refresh.add_argument("--config")

    pilot = commands.add_parser(
        "run-pilot", help="run the representative timing pilot and freeze timeout policy"
    )
    pilot.add_argument("--config")

    full = commands.add_parser(
        "run-full", help="launch the full benchmark after explicit approval"
    )
    full.add_argument("--config")
    full.add_argument(
        "--confirm-full-run",
        action="store_true",
        help="required acknowledgement; this runs all approved maps, arms, and seeds",
    )

    args = parser.parse_args()
    if args.command == "write-config":
        destination = BenchmarkConfig().save(args.output)
        print(f"Wrote approved benchmark configuration: {destination}")
        return
    config = _config(args.config)
    if args.command == "prepare-data":
        result = prepare_benchmark_data(config, overwrite=args.overwrite)
        print(json.dumps(result["manifest"], indent=2, sort_keys=True))
        return
    if args.command == "train-models":
        result = train_models_and_calibrate(config, overwrite=args.overwrite)
        print(f"Training summary: {result['summary_path']}")
        print(f"Model-ready manifest: {result['model_ready_manifest']}")
        return
    if args.command == "refresh-stats":
        manifest = refresh_distribution_statistics(
            f"{config.output_directory}/data"
        )
        print(json.dumps(manifest["distribution_statistics"], indent=2, sort_keys=True))
        return
    if args.command == "run-pilot":
        result = run_benchmark_searches(config, pilot=True)
        print(f"Pilot raw rows: {result['raw_csv']}")
        print(json.dumps(result["summary"], indent=2, sort_keys=True))
        return
    if args.command == "run-full":
        if not args.confirm_full_run:
            parser.error("run-full requires --confirm-full-run after the user approves the full benchmark.")
        saved_config = f"{config.output_directory}/benchmark_config.json"
        config = BenchmarkConfig.load(args.config or saved_config)
        result = run_benchmark_searches(config, pilot=False)
        print(f"Full raw results: {result['raw_csv']}")
        print(f"Full benchmark manifest: {result['manifest_path']}")
        return
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
