import os
AE_logs = os.path.join(os.path.dirname(__file__), "AE_logs", 'lightning_logs')
MAE_logs = os.path.join(os.path.dirname(__file__), "MAE_logs", 'lightning_logs')
COALA_logs = os.path.join(os.path.dirname(__file__), "COALA_logs", 'lightning_logs')
Head_logs = os.path.join(os.path.dirname(__file__), "TaskHeads_logs")
Classifier_logs = os.path.join(Head_logs, "classifier", "lightning_logs")
