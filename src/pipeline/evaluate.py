"""Phase A/B entry point. Later phases are intentionally not executed."""
import argparse
import json
import logging
from pathlib import Path

from src.config import load_config
from src.data.loaders import download_niteroi
from src.data.preprocess import preprocess
from src.data.schema import DataError
from src.utils.logging import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/niteroi.yaml"))
    parser.add_argument("command", choices=["download", "audit", "baselines", "reports"])
    args = parser.parse_args()
    config = load_config(args.config)
    root = args.config.resolve().parent.parent
    try:
        if args.command == "download":
            LOGGER.info("Verified source: %s", download_niteroi(root, config.data))
        elif args.command == "audit":
            _, report = preprocess(root, config)
            LOGGER.info("Audit: %s", report.statistics)
        elif args.command == "reports":
            import pandas as pd
            from src.pipeline.reporting import generate_reports
            if (root / 'artifacts/latest_reliability.json').exists():
                from src.pipeline.research_reporting import generate
                generate(root)
            else:
                manifest = json.loads((root / "artifacts/latest_run.json").read_text())
                generate_reports(root, manifest, pd.read_csv(root / manifest["metrics_csv"]))
            LOGGER.info("Reports regenerated from saved run and pytest XML")
        else:
            from src.pipeline.baselines import run_baselines
            manifest = run_baselines(root, config)
            if manifest["failed_analyses"]:
                LOGGER.error("Run contains failed analyses; inspect the saved result CSV")
                raise SystemExit(1)
    except DataError as exc:
        LOGGER.error("Stopped: %s", exc)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
