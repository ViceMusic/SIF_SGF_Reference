"""Configurable inference entrypoint.

Loads a single matching weight file by model + task + representation prefix
and performs SIF/SGF prediction from an input CSV containing a ``SMILES`` column.
"""

from pathlib import Path
import warnings
import re

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extraction import PeptideFeaturizer


warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = (PROJECT_ROOT / "data" / "weights").resolve()
TASKS = ["SIF", "SGF"]

REPRESENTATION_ALIASES = {
    "morgan": "Morgan(1024)",
    "avalon": "Avalon(512)",
    "molt5": "MolT5(768)",
    "molformer": "molFormer(768)",
    "avalon+molformer": "Avalon(512)_molFormer(768)",
    "avalon+molt5": "Avalon(512)_MolT5(768)",
}

DEFAULT_BITS = {
    "morgan_bits": 1024,
    "avalon_bits": 512,
    "molformer_bits": 768,
    "molt5_bits": 768,
}


def normalize_representation_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "")
    if key not in REPRESENTATION_ALIASES:
        supported = ", ".join(REPRESENTATION_ALIASES.keys())
        raise ValueError(f"Unsupported representation '{name}'. Supported: {supported}")
    return REPRESENTATION_ALIASES[key]


def _select_bits_from_prints(print_name: str):
    flags = {
        "morgan_bits": 0,
        "avalon_bits": 0,
        "molformer_bits": 0,
        "molt5_bits": 0,
    }
    if "Morgan" in print_name:
        flags["morgan_bits"] = DEFAULT_BITS["morgan_bits"]
    if "Avalon" in print_name:
        flags["avalon_bits"] = DEFAULT_BITS["avalon_bits"]
    if re.search(r"molFormer", print_name, re.IGNORECASE):
        flags["molformer_bits"] = DEFAULT_BITS["molformer_bits"]
    if re.search(r"MolT5", print_name, re.IGNORECASE):
        flags["molt5_bits"] = DEFAULT_BITS["molt5_bits"]
    return flags


def _resolve_model_path(model_name: str, task: str, feature_name: str) -> Path:
    prefix = f"{model_name}_{task.lower()}_{feature_name}_"
    matches = sorted(MODEL_DIR.glob(f"{prefix}*.pkl"))
    if not matches:
        raise FileNotFoundError(
            f"No weight file matched prefix '{prefix}' in {MODEL_DIR}"
        )
    plain_suffix_matches = []
    for path in matches:
        suffix = path.stem[len(prefix):]
        if "(" not in suffix and ")" not in suffix:
            plain_suffix_matches.append(path)
    if plain_suffix_matches:
        matches = plain_suffix_matches
    if len(matches) > 1:
        matched = ", ".join(path.name for path in matches)
        raise RuntimeError(
            f"Expected exactly one weight file for prefix '{prefix}', found: {matched}"
        )
    return matches[0]


def _load_model_for_task(model_name: str, task: str, feature_name: str):
    model_path = _resolve_model_path(model_name, task, feature_name)
    model = joblib.load(model_path)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    print(f"模型加载成功: {model_path}")
    return model


def _featurize_smiles_list(smiles_list, featurizer: PeptideFeaturizer, desc: str = None):
    X = []

    try:
        dim = len(featurizer.get_feature_names())
    except Exception:
        dim = 0

    iterator = enumerate(smiles_list)
    if desc is not None:
        iterator = tqdm(iterator, total=len(smiles_list), desc=desc)

    for _, smiles in iterator:
        try:
            vec, ok = featurizer.featurize(smiles)
            if not ok or vec is None:
                X.append(np.zeros(dim, dtype=np.float32))
            else:
                X.append(np.array(vec, dtype=np.float32))
        except Exception:
            X.append(np.zeros(dim, dtype=np.float32))

    if not X:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(X)


def predict_from_csv(csv_path, model_name: str = "lr", representation: str = "morgan"):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    if model_name not in {"lr", "rf", "xgb"}:
        raise ValueError("model_name must be one of: lr, rf, xgb")

    feature_name = normalize_representation_name(representation)

    df = pd.read_csv(csv_path)
    if "SMILES" not in df.columns:
        raise ValueError('Input CSV must contain a SMILES column named exactly "SMILES"')

    smiles_list = df["SMILES"].astype(str).tolist()
    bits = _select_bits_from_prints(feature_name)
    featurizer = PeptideFeaturizer(**bits)
    X = _featurize_smiles_list(smiles_list, featurizer, desc=f"Featurize {feature_name}")

    models = {
        task: _load_model_for_task(model_name=model_name, task=task, feature_name=feature_name)
        for task in TASKS
    }

    results = [{"SMILES": s, "SIF": None, "SGF": None} for s in smiles_list]

    for task in TASKS:
        try:
            preds = models[task].predict(X)
        except Exception as exc:
            raise RuntimeError(f"Prediction failed for task {task}: {exc}") from exc

        for i, pred in tqdm(enumerate(preds), total=len(preds), desc=f"Assign {task}"):
            try:
                results[i][task] = int(pred)
            except Exception:
                results[i][task] = None

    return results
