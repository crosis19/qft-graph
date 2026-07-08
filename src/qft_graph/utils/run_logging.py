"""Run-metadata logging: every quoted number traces to a results/<run_id>.json.

Implements the reproducibility ground rule: log (git commit, config hash,
seeds, metrics) for every run whose numbers feed a figure or table.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS_DIR = Path(__file__).resolve().parents[3] / "results"


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).resolve().parents[3],
                check=True,
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "item"):  # torch/numpy scalars
        return obj.item()
    if hasattr(obj, "tolist"):  # torch/numpy arrays
        return obj.tolist()
    return obj


def config_hash(config: Any) -> str:
    """Stable short hash of a (dataclass or dict) config."""
    payload = json.dumps(_to_jsonable(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def log_run(
    run_id: str,
    config: Any,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    results_dir: str | Path | None = None,
) -> Path:
    """Write results/<run_id>.json with full provenance.

    Args:
        run_id: Unique name for this run (becomes the filename).
        config: The config object (dataclass or dict) that drove the run.
        metrics: Result numbers to record.
        extra: Optional additional payload (e.g. dataset paths, timings).
        results_dir: Override for the results directory (mainly for tests).

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "config_hash": config_hash(config),
        "config": _to_jsonable(config),
        "metrics": _to_jsonable(metrics),
    }
    if extra:
        record["extra"] = _to_jsonable(extra)
    path = out_dir / f"{run_id}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    return path
