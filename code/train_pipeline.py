import json
from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from feature_extraction import PeptideFeaturizer
from feature_extraction.utils import extract_molecular_features


ROOT_DIR = Path(__file__).resolve().parent.parent
TRAIN_RESULT_DIR = ROOT_DIR / "train_result"
TASKS = ("SIF", "SGF")

DEFAULT_CONFIG = {
    "representation": "morgan",
    "model": "lr",
    "rounds": 3,
    "random_state": 42,
    "filter_monomer_only": False,
    "rf_n_estimators": 200,
    "rf_n_jobs": 1,
    "xgb_n_estimators": 200,
    "xgb_max_depth": 6,
    "xgb_learning_rate": 0.1,
    "xgb_tree_method": "hist",
    "lr_max_iter": 1000,
}

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
    if "molFormer" in print_name:
        flags["molformer_bits"] = DEFAULT_BITS["molformer_bits"]
    if "MolT5" in print_name:
        flags["molt5_bits"] = DEFAULT_BITS["molt5_bits"]
    return flags


def build_model(model_name: str, config: dict, random_state: int):
    if model_name == "lr":
        return LogisticRegression(
            max_iter=config["lr_max_iter"],
            class_weight="balanced",
            random_state=random_state,
        )
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=config["rf_n_estimators"],
            class_weight="balanced",
            n_jobs=config["rf_n_jobs"],
            random_state=random_state,
        )
    if model_name == "xgb":
        return XGBClassifier(
            n_estimators=config["xgb_n_estimators"],
            max_depth=config["xgb_max_depth"],
            learning_rate=config["xgb_learning_rate"],
            random_state=random_state,
            tree_method=config["xgb_tree_method"],
            eval_metric="logloss",
        )
    raise ValueError(f"Unsupported model '{model_name}'")


def _validate_input_frame(df: pd.DataFrame, csv_path: Path):
    required_columns = {"SMILES", "SIF", "SGF"}
    missing = required_columns - set(df.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"{csv_path} missing required columns: {joined}")


def _filter_and_prepare_dataset(csv_path: Path, filter_monomer_only: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    _validate_input_frame(df, csv_path)

    df = df[["SMILES", "SIF", "SGF"]].copy()
    df = df[df["SMILES"].notna()].copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[df["SMILES"] != ""].copy()

    feature_records = []
    for _, row in df.iterrows():
        feature_records.append(extract_molecular_features(row["SMILES"]))
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(feature_records)], axis=1)

    if filter_monomer_only and "is_monomer" in df.columns:
        df = df[df["is_monomer"] == True].copy()

    for task in TASKS:
        df[task] = pd.to_numeric(df[task], errors="coerce")
    df = df[df["SIF"].isin([0, 1]) & df["SGF"].isin([0, 1])].copy()
    df.reset_index(drop=True, inplace=True)
    return df


def _featurize_smiles_list(smiles_list, featurizer: PeptideFeaturizer):
    vectors = []
    try:
        dim = len(featurizer.get_feature_names())
    except Exception:
        dim = 0

    for smiles in smiles_list:
        try:
            vec, ok = featurizer.featurize(smiles)
            if not ok or vec is None:
                vectors.append(np.zeros(dim, dtype=np.float32))
            else:
                vectors.append(np.array(vec, dtype=np.float32))
        except Exception:
            vectors.append(np.zeros(dim, dtype=np.float32))

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def _build_feature_matrices(train_df: pd.DataFrame, test_df: pd.DataFrame, representation: str):
    print_name = normalize_representation_name(representation)
    bits = _select_bits_from_prints(print_name)
    featurizer = PeptideFeaturizer(**bits)
    X_train = _featurize_smiles_list(train_df["SMILES"].tolist(), featurizer)
    X_test = _featurize_smiles_list(test_df["SMILES"].tolist(), featurizer)
    return print_name, X_train, X_test


def _compute_metrics(y_true, y_pred, y_prob):
    metrics = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "AUPRC": float(average_precision_score(y_true, y_prob)),
    }
    return metrics


def _model_file_name(task: str, model_name: str, representation: str) -> str:
    safe_representation = normalize_representation_name(representation)
    return f"{model_name}_{task.lower()}_{safe_representation}_best.pkl"


def _score_tuple(metrics: dict):
    return (metrics["AUPRC"], metrics["F1"], metrics["ACC"])


def train_and_evaluate(train_csv: str, test_csv: str, config: dict | None = None):
    cfg = deepcopy(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    train_path = Path(train_csv).resolve()
    test_path = Path(test_csv).resolve()
    if not train_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    TRAIN_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = _filter_and_prepare_dataset(train_path, cfg["filter_monomer_only"])
    test_df = _filter_and_prepare_dataset(test_path, cfg["filter_monomer_only"])
    if train_df.empty:
        raise ValueError("No valid training samples remain after filtering")
    if test_df.empty:
        raise ValueError("No valid test samples remain after filtering")

    print_name, X_train, X_test = _build_feature_matrices(
        train_df=train_df,
        test_df=test_df,
        representation=cfg["representation"],
    )

    results = {}
    for task in TASKS:
        y_train = train_df[task].astype(int).to_numpy()
        y_test = test_df[task].astype(int).to_numpy()
        if len(np.unique(y_train)) < 2:
            raise ValueError(f"Training labels for {task} need both 0 and 1")
        if len(np.unique(y_test)) < 2:
            raise ValueError(f"Test labels for {task} need both 0 and 1")

        best_result = None
        best_model = None
        rounds = []
        for round_idx in range(cfg["rounds"]):
            random_state = cfg["random_state"] + round_idx
            model = build_model(cfg["model"], cfg, random_state)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = _compute_metrics(y_true=y_test, y_pred=y_pred, y_prob=y_prob)
            round_result = {
                "round": round_idx + 1,
                "random_state": random_state,
                "metrics": metrics,
            }
            rounds.append(round_result)

            if best_result is None or _score_tuple(metrics) > _score_tuple(best_result["best_metrics"]):
                best_result = {
                    "task": task,
                    "model": cfg["model"],
                    "representation": print_name,
                    "train_samples": int(len(y_train)),
                    "test_samples": int(len(y_test)),
                    "best_round": round_idx + 1,
                    "best_random_state": random_state,
                    "best_metrics": metrics,
                    "all_rounds": rounds.copy(),
                }
                best_model = model

        model_path = TRAIN_RESULT_DIR / _model_file_name(
            task=task,
            model_name=cfg["model"],
            representation=cfg["representation"],
        )
        json_path = TRAIN_RESULT_DIR / f"{task.lower()}_metrics.json"

        joblib.dump(best_model, model_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(best_result, f, ensure_ascii=False, indent=2)

        best_result["model_path"] = str(model_path)
        best_result["json_path"] = str(json_path)
        results[task] = best_result

    return results
