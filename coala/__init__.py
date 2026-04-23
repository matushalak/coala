from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
DATADIR = REPO_ROOT / "data"

AE_logs = str(PACKAGE_ROOT / "logs" / "AE_logs")
MAE_logs = str(PACKAGE_ROOT / "logs" / "MAE_logs")
FM_logs = str(PACKAGE_ROOT / "logs" / "FM_logs")
JEPA_logs = str(PACKAGE_ROOT / "logs" / "JEPA_logs")
LeJEPA_logs = str(PACKAGE_ROOT / "logs" / "LeJEPA_logs")
COALA_logs = str(PACKAGE_ROOT / "logs" / "COALA_logs")
RNN_logs = str(PACKAGE_ROOT / "logs" / "RNN")
hcRNN_logs = str(PACKAGE_ROOT / "logs" / "hcRNN")
rCNN_logs = str(PACKAGE_ROOT / "logs" / "rCNN")
lrRNN_logs = str(PACKAGE_ROOT / "logs" / "lrRNN")
Head_logs = str(PACKAGE_ROOT / "logs" / "TaskHeads_logs")
Classifier_logs = str(PACKAGE_ROOT / "logs" / "TaskHeads_logs" / "classifier")
Reconstruction_logs = str(PACKAGE_ROOT / "logs" / "TaskHeads_logs" / "reconstruction")

_DATASET_LOG_ALIASES = {
    "cifar10": "cifar",
}


def dataset_log_name(dataset_name: str) -> str:
    from .datasets.registry import resolve_dataset_name

    canonical_name = resolve_dataset_name(dataset_name)
    return _DATASET_LOG_ALIASES.get(canonical_name, canonical_name)


def dataset_log_dir(log_root: str, dataset_name: str) -> str:
    return str(Path(log_root).resolve() / dataset_log_name(dataset_name))


def dataset_lightning_logs_dir(log_root: str, dataset_name: str) -> str:
    return str(Path(dataset_log_dir(log_root, dataset_name)) / "lightning_logs")


AE_LIGHTNING_LOGS = dataset_lightning_logs_dir(AE_logs, "mnist")
MAE_LIGHTNING_LOGS = dataset_lightning_logs_dir(MAE_logs, "mnist")
FM_LIGHTNING_LOGS = dataset_lightning_logs_dir(FM_logs, "mnist")
JEPA_LIGHTNING_LOGS = dataset_lightning_logs_dir(JEPA_logs, "mnist")
LeJEPA_LIGHTNING_LOGS = dataset_lightning_logs_dir(LeJEPA_logs, "mnist")
COALA_LIGHTNING_LOGS = dataset_lightning_logs_dir(COALA_logs, "mnist")
CLASSIFIER_LIGHTNING_LOGS = dataset_lightning_logs_dir(Classifier_logs, "mnist")
RECONSTRUCTION_LIGHTNING_LOGS = dataset_lightning_logs_dir(Reconstruction_logs, "mnist")

__all__ = [
    "PACKAGE_ROOT",
    "REPO_ROOT",
    "DATADIR",
    "AE_logs",
    "MAE_logs",
    "FM_logs",
    "JEPA_logs",
    "LeJEPA_logs",
    "COALA_logs",
    "RNN_logs",
    "hcRNN_logs",
    "rCNN_logs",
    "lrRNN_logs",
    "Head_logs",
    "Classifier_logs",
    "Reconstruction_logs",
    "AE_LIGHTNING_LOGS",
    "MAE_LIGHTNING_LOGS",
    "FM_LIGHTNING_LOGS",
    "JEPA_LIGHTNING_LOGS",
    "LeJEPA_LIGHTNING_LOGS",
    "COALA_LIGHTNING_LOGS",
    "CLASSIFIER_LIGHTNING_LOGS",
    "RECONSTRUCTION_LIGHTNING_LOGS",
    "dataset_log_name",
    "dataset_log_dir",
    "dataset_lightning_logs_dir",
]
