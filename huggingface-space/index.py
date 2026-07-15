"""Unified batch prediction entry point for SIF/SGF stability."""

from collections.abc import Sequence

from extractor import extract_features
from model import load_model


def predict(
    smiles_values: Sequence[str],
    task: str,
    representation: str,
    model_name: str,
) -> list[int]:
    """Map a batch of SMILES strings to 0 (unstable) or 1 (stable).

    Parameters are case-insensitive. Supported tasks are SIF/SGF,
    representations are Morgan/Avalon, and models are lr/rf/xgb.
    """
    features = extract_features(smiles_values, representation)
    if features.shape[0] == 0:
        return []

    classifier = load_model(model_name, task, representation)
    expected_features = getattr(classifier, "n_features_in_", None)
    if expected_features is not None and features.shape[1] != expected_features:
        raise RuntimeError(
            "Feature/model mismatch: "
            f"model expects {expected_features}, extractor produced {features.shape[1]}"
        )

    predictions = classifier.predict(features)
    result = [int(value) for value in predictions]
    if any(value not in (0, 1) for value in result):
        raise RuntimeError("The classifier returned a value other than 0 or 1")
    return result


if __name__ == "__main__":
    example = ["CCO", "CCN"]
    print(predict(example, task="SIF", representation="Morgan", model_name="lr"))
