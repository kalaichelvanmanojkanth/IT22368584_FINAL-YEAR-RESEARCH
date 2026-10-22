"""Run provenance, strict JSON and installed dependency records."""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions
import json
from pathlib import Path
import platform
import random
import subprocess
from uuid import uuid4

import numpy as np


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_manifest(root: Path, config: dict, source: Path) -> dict:
    now = datetime.now(timezone.utc)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    source_files = sorted((root / "src").rglob("*.py"))
    source_hash = hashlib.sha256()
    for path in source_files:
        source_hash.update(str(path.relative_to(root)).encode())
        source_hash.update(path.read_bytes())
    return {"run_id": now.strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8],
            "timestamp_utc": now.isoformat(), "python": platform.python_version(),
            "platform": platform.platform(), "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "source_code_sha256": source_hash.hexdigest(),
            "dataset_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset_file": str(source.relative_to(root)), "config": config,
            "dependencies": {d.metadata["Name"]: d.version for d in distributions()},
            "phase": "A/B", "pseudo_experiments_completed": 0, "reliability_model_trained": False}
