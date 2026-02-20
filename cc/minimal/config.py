import torch.nn as nn
# Fix this so it works as a dictionary
basic = {
    "n_features": 2,
    "n_pv": 2,
    "n_context": 2,
    "activation": nn.ReLU(),
    "lr_ff": 0.01,
    "lr_fb": 0.01,
    "lr_lat": 0.01,
    "lr_pv": 0.01,
    "pyc_decay": 0.1,
    "pv_decay": 0.25,
    "alpha": 1.0,
    "weight_decay": 0.0,
    "seed": 42,
}

# TODO: different neuron types with different predicted trajectories