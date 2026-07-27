"""Persisting fitted model pipelines.

The explain step needs the fitted model, but a sklearn Pipeline cannot live
in LangGraph state -- it is not serialisable and state is checkpointed on
every transition. So the training node writes the model to disk and passes a
path, which is the same discipline datasets already follow.

This also populates `Experiment.artifact_path`, making a trained model a real
artifact rather than something that vanishes when the request ends.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings
from app.core.logging import get_logger
from app.services.storage import resolve

logger = get_logger(__name__)


def save_model(pipeline: Any, run_id: str, model_name: str) -> str:
    """Persist a fitted pipeline; return its path relative to STORAGE_DIR."""
    dest_dir = settings.STORAGE_DIR / "models" / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{model_name}.joblib"

    joblib.dump(pipeline, dest, compress=3)
    relative = str(dest.relative_to(settings.STORAGE_DIR)).replace("\\", "/")
    logger.info(
        "Model saved",
        extra={"path": relative, "kb": round(dest.stat().st_size / 1024, 1)},
    )
    return relative


def load_model(relative_path: str) -> Any:
    """Load a persisted pipeline. Path is validated against the storage root."""
    path: Path = resolve(relative_path)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact missing: {relative_path}")
    return joblib.load(path)


def delete_run_models(run_id: str) -> None:
    import shutil

    target = settings.STORAGE_DIR / "models" / run_id
    if target.is_dir():
        shutil.rmtree(target, ignore_errors=True)
