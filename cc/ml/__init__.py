import os

_ML_DIR = os.path.dirname(__file__)

AE_logs = os.path.join(_ML_DIR, "logs", "AE_logs")
MAE_logs = os.path.join(_ML_DIR, "logs", "MAE_logs")
FM_logs = os.path.join(_ML_DIR, "logs", "FM_logs")
JEPA_logs = os.path.join(_ML_DIR, "logs", "JEPA_logs")
LeJEPA_logs = os.path.join(_ML_DIR, "logs", "LeJEPA_logs")
COALA_logs = os.path.join(_ML_DIR, "logs", "COALA_logs")
Head_logs = os.path.join(_ML_DIR, "logs", "TaskHeads_logs")
Classifier_logs = os.path.join(Head_logs, "classifier")
Reconstruction_logs = os.path.join(Head_logs, "reconstruction")

_DATASET_LOG_ALIASES = {
    "cifar10": "cifar",
}


def dataset_log_name(dataset_name: str) -> str:
    from cc.datasets.registry import resolve_dataset_name

    canonical_name = resolve_dataset_name(dataset_name)
    return _DATASET_LOG_ALIASES.get(canonical_name, canonical_name)


def dataset_log_dir(log_root: str, dataset_name: str) -> str:
    return os.path.join(os.path.abspath(log_root), dataset_log_name(dataset_name))


def dataset_lightning_logs_dir(log_root: str, dataset_name: str) -> str:
    return os.path.join(dataset_log_dir(log_root, dataset_name), "lightning_logs")


AE_LIGHTNING_LOGS = dataset_lightning_logs_dir(AE_logs, "mnist")
MAE_LIGHTNING_LOGS = dataset_lightning_logs_dir(MAE_logs, "mnist")
FM_LIGHTNING_LOGS = dataset_lightning_logs_dir(FM_logs, "mnist")
JEPA_LIGHTNING_LOGS = dataset_lightning_logs_dir(JEPA_logs, "mnist")
LeJEPA_LIGHTNING_LOGS = dataset_lightning_logs_dir(LeJEPA_logs, "mnist")
COALA_LIGHTNING_LOGS = dataset_lightning_logs_dir(COALA_logs, "mnist")
CLASSIFIER_LIGHTNING_LOGS = dataset_lightning_logs_dir(Classifier_logs, "mnist")
RECONSTRUCTION_LIGHTNING_LOGS = dataset_lightning_logs_dir(Reconstruction_logs, "mnist")

__all__ = [
    "AE_logs",
    "MAE_logs",
    "FM_logs",
    "JEPA_logs",
    "LeJEPA_logs",
    "COALA_logs",
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
