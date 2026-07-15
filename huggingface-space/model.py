"""Resolve and cache the twelve packaged classification models."""

from functools import lru_cache
from pathlib import Path

import joblib


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
SUPPORTED_MODELS = ("lr", "rf", "xgb")
SUPPORTED_TASKS = ("sif", "sgf")
SUPPORTED_REPRESENTATIONS = ("morgan", "avalon")

WEIGHT_FILES = {
    ("lr", "sif", "morgan"): "lr_sif_Morgan(1024)_maxiter160.pkl",
    ("lr", "sgf", "morgan"): "lr_sgf_Morgan(1024)_maxiter160.pkl",
    ("rf", "sif", "morgan"): "rf_sif_Morgan(1024)_nest37_jobs-1.pkl",
    ("rf", "sgf", "morgan"): "rf_sgf_Morgan(1024)_nest37_jobs-1.pkl",
    ("xgb", "sif", "morgan"): "xgb_sif_Morgan(1024)_depth5_lr0.1_nest200.pkl",
    ("xgb", "sgf", "morgan"): "xgb_sgf_Morgan(1024)_depth5_lr0.1_nest200.pkl",
    ("lr", "sif", "avalon"): "lr_sif_Avalon(512)_maxiter160.pkl",
    ("lr", "sgf", "avalon"): "lr_sgf_Avalon(512)_maxiter160.pkl",
    ("rf", "sif", "avalon"): "rf_sif_Avalon(512)_nest9_jobs-1.pkl",
    ("rf", "sgf", "avalon"): "rf_sgf_Avalon(512)_nest9_jobs-1.pkl",
    ("xgb", "sif", "avalon"): "xgb_sif_Avalon(512)_depth6_lr0.1_nest100.pkl",
    ("xgb", "sgf", "avalon"): "xgb_sgf_Avalon(512)_depth6_lr0.1_nest100.pkl",
}


def _normalize(value: str, name: str, supported: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip().lower()
    if normalized not in supported:
        choices = ", ".join(supported)
        raise ValueError(f"Unsupported {name} '{value}'; choose: {choices}")
    return normalized


@lru_cache(maxsize=len(WEIGHT_FILES))
def load_model(model_name: str, task: str, representation: str):
    """Load one requested model and reuse it for subsequent predictions."""
    normalized_model = _normalize(model_name, "model", SUPPORTED_MODELS)
    normalized_task = _normalize(task, "task", SUPPORTED_TASKS)
    normalized_representation = _normalize(
        representation,
        "representation",
        SUPPORTED_REPRESENTATIONS,
    )

    filename = WEIGHT_FILES[
        (normalized_model, normalized_task, normalized_representation)
    ]
    path = WEIGHTS_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Model weight not found: {path}")

    loaded = joblib.load(path)
    if hasattr(loaded, "n_jobs"):
        loaded.n_jobs = 1
    return loaded
