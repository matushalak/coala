import torch
from numpy.random import shuffle as np_shuffle
from typing import Iterable

def get_minimal_data(*trial_patterns:list[list[int]], 
                    n_trials:int = 10, trial_length:int = 100,
                    n_inputs:int = 6, to_tensor:bool = True
                    )->list|torch.Tensor:
    zeros = torch.zeros(trial_length//4, n_inputs)
    n_patterns = len(trial_patterns)
    trials = []
    for i in range(n_trials):
        trials.append(torch.cat((zeros, torch.tensor(trial_patterns[i % n_patterns]).tile(trial_length//2, 1), zeros), dim=0))
    if to_tensor:
        return torch.cat(trials, dim=0)
    return trials

def get_trial_type_minimal_data(trial_dict:dict[int:torch.Tensor], 
                                trial_types:Iterable[int],
                                n_trials_per_type:int = 10, trial_length:int = 100,
                                noise_level:float = 0.1
                                )->tuple[torch.Tensor, torch.Tensor]:
    # Get all trials
    total__trials = len(trial_types) * n_trials_per_type
    patterns = [trial_dict[trial_type] for trial_type in trial_types]
    trials = get_minimal_data(*patterns, n_trials=total__trials, trial_length=trial_length, to_tensor=False)
    np_shuffle(trials) # randomly shuffle trials (in place operation)
    trials_tensor = torch.cat(trials, dim=0) 
    trials_tensor += noise_level * torch.randn_like(trials_tensor)  # add controlled-level of noise

    return trials_tensor

def get_trial_dict(*trial_patterns:list[list[int]])->dict:
    trial_dict = {}
    n_patterns = len(trial_patterns)
    for i in range(n_patterns):
        trial_dict[i] = torch.tensor(trial_patterns[i])
    return trial_dict

def get_patterns()-> tuple[list[int]]:
    '''
    Patterns designed to test the minimal model
    '''
    return ([1, 1, 1, 1, 1, 1], # activate all inputs (0)
            # activate individual pyramidal neurons (1-3)
            [1, 1, 0, 0, 0, 0], [0,0,1,1,0,0], [0,0,0,0,1,1],
            # activate alternating inputs to test PV inhibition (4-5)
            [1,0,1,0,1,0], [0,1,0,1,0,1],
            # test HVA feeedback by activating different combinations of PyCs (6-10)
            [1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [1,0,0,0,0,1])