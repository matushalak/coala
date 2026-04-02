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

AE_LIGHTNING_LOGS = os.path.join(AE_logs, "lightning_logs")
MAE_LIGHTNING_LOGS = os.path.join(MAE_logs, "lightning_logs")
FM_LIGHTNING_LOGS = os.path.join(FM_logs, "lightning_logs")
JEPA_LIGHTNING_LOGS = os.path.join(JEPA_logs, "lightning_logs")
LeJEPA_LIGHTNING_LOGS = os.path.join(LeJEPA_logs, "lightning_logs")
COALA_LIGHTNING_LOGS = os.path.join(COALA_logs, "lightning_logs")
CLASSIFIER_LIGHTNING_LOGS = os.path.join(Classifier_logs, "lightning_logs")
RECONSTRUCTION_LIGHTNING_LOGS = os.path.join(Reconstruction_logs, "lightning_logs")

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
    "RECONSTRUCTION_LIGHTNING_LOGS"
]
