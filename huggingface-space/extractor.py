"""Batch SMILES feature extraction for the packaged prediction models.

The existing trained models use 24 molecular descriptors followed by either
a 1024-bit Morgan fingerprint or a 512-bit Avalon fingerprint.  The public
fingerprint helpers return only the fingerprint, while ``extract_features``
returns the complete model-ready matrix.
"""

from collections.abc import Iterable, Sequence

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)


MORGAN_BITS = 1024
AVALON_BITS = 512
BASE_FEATURES = 24
SUPPORTED_REPRESENTATIONS = ("morgan", "avalon")
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=MORGAN_BITS,
    includeChirality=True,
)


def _validate_smiles(smiles_values: Iterable[str]) -> tuple[list[str], list[Chem.Mol]]:
    if isinstance(smiles_values, (str, bytes)):
        raise TypeError("smiles_values must be an iterable of SMILES strings")
    try:
        values = list(smiles_values)
    except TypeError as exc:
        raise TypeError("smiles_values must be an iterable of SMILES strings") from exc
    molecules: list[Chem.Mol] = []
    invalid: list[int] = []

    for index, smiles in enumerate(values):
        if not isinstance(smiles, str) or not smiles.strip():
            invalid.append(index)
            continue
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            invalid.append(index)
            continue
        molecules.append(molecule)

    if invalid:
        positions = ", ".join(map(str, invalid))
        raise ValueError(f"Invalid SMILES at indices: {positions}")

    return values, molecules


def _bit_vector_to_array(fingerprint, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, result)
    return result


def _morgan_from_molecules(molecules: Sequence[Chem.Mol]) -> np.ndarray:
    rows = [
        _bit_vector_to_array(
            _MORGAN_GENERATOR.GetFingerprint(molecule),
            MORGAN_BITS,
        )
        for molecule in molecules
    ]
    return np.vstack(rows) if rows else np.empty((0, MORGAN_BITS), dtype=np.float32)


def _avalon_from_molecules(molecules: Sequence[Chem.Mol]) -> np.ndarray:
    rows = [
        _bit_vector_to_array(
            pyAvalonTools.GetAvalonFP(molecule, AVALON_BITS),
            AVALON_BITS,
        )
        for molecule in molecules
    ]
    return np.vstack(rows) if rows else np.empty((0, AVALON_BITS), dtype=np.float32)


def smiles_to_morgan(smiles_values: Sequence[str]) -> np.ndarray:
    """Convert a batch of SMILES strings to a ``(n, 1024)`` Morgan matrix."""
    _, molecules = _validate_smiles(smiles_values)
    return _morgan_from_molecules(molecules)


def smiles_to_avalon(smiles_values: Sequence[str]) -> np.ndarray:
    """Convert a batch of SMILES strings to a ``(n, 512)`` Avalon matrix."""
    _, molecules = _validate_smiles(smiles_values)
    return _avalon_from_molecules(molecules)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def _base_features(molecule: Chem.Mol) -> np.ndarray:
    """Reproduce the 24 descriptor values used while training the models."""
    try:
        qed = QED.properties(molecule)
        qed_features = [
            qed.MW,
            qed.ALOGP,
            qed.HBA,
            qed.HBD,
            qed.PSA,
            qed.ROTB,
            qed.AROM,
            qed.ALERTS,
        ]
    except Exception:
        qed_features = [0.0] * 8

    try:
        rotatable_bonds = float(rdMolDescriptors.CalcNumRotatableBonds(molecule))
        ring_count = float(rdMolDescriptors.CalcNumRings(molecule))
        physchem_features = [
            Descriptors.MolWt(molecule),
            Crippen.MolLogP(molecule),
            rdMolDescriptors.CalcNumHBA(molecule),
            rdMolDescriptors.CalcNumHBD(molecule),
            rdMolDescriptors.CalcTPSA(molecule),
            rotatable_bonds,
            ring_count,
            rdMolDescriptors.CalcFractionCSP3(molecule),
            Descriptors.HeavyAtomCount(molecule),
            molecule.GetNumAtoms(),
            ring_count / (1.0 + rotatable_bonds),
        ]
    except Exception:
        physchem_features = [0.0] * 11

    try:
        AllChem.ComputeGasteigerCharges(molecule)
        charges = np.asarray(
            [
                _safe_float(atom.GetProp("_GasteigerCharge"))
                for atom in molecule.GetAtoms()
            ],
            dtype=np.float64,
        )
        if charges.size == 0:
            charges = np.asarray([0.0], dtype=np.float64)
        charges = np.where(np.isfinite(charges), charges, 0.0)
        charge_features = [
            np.mean(charges),
            np.max(charges),
            np.min(charges),
            np.std(charges),
            np.sum(charges),
        ]
    except Exception:
        charge_features = [0.0] * 5

    result = np.asarray(
        qed_features + physchem_features + charge_features,
        dtype=np.float32,
    )
    if result.shape != (BASE_FEATURES,):
        raise RuntimeError(f"Expected {BASE_FEATURES} base features, got {result.size}")
    return result


def extract_features(
    smiles_values: Sequence[str],
    representation: str,
) -> np.ndarray:
    """Build the complete batch feature matrix expected by the saved models."""
    if not isinstance(representation, str):
        raise TypeError("representation must be a string")
    normalized = representation.strip().lower()
    if normalized not in SUPPORTED_REPRESENTATIONS:
        supported = ", ".join(SUPPORTED_REPRESENTATIONS)
        raise ValueError(f"Unsupported representation '{representation}'; choose: {supported}")

    _, molecules = _validate_smiles(smiles_values)
    if not molecules:
        fingerprint_size = MORGAN_BITS if normalized == "morgan" else AVALON_BITS
        return np.empty((0, BASE_FEATURES + fingerprint_size), dtype=np.float32)

    descriptors = np.vstack([_base_features(molecule) for molecule in molecules])
    if normalized == "morgan":
        fingerprints = _morgan_from_molecules(molecules)
    else:
        fingerprints = _avalon_from_molecules(molecules)
    return np.hstack((descriptors, fingerprints)).astype(np.float32, copy=False)
