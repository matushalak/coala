
import torch

def get_minimal_data_stream(*trial_patterns:list[list[int]], 
                            n_trials:int = 10, trial_length:int = 100,
                            n_inputs:int = 6
                            )->torch.Tensor:
    zeros = torch.zeros(trial_length//4, n_inputs)
    n_patterns = len(trial_patterns)
    trials = []
    for i in range(n_trials):
        trials.append(torch.cat((zeros, torch.tensor(trial_patterns[i % n_patterns]).tile(trial_length//2, 1), zeros), dim=0))
    return torch.cat(trials, dim=0)