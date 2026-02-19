# author: Matúš Halák (@matushalak)
import torch
from numpy.random import shuffle as np_shuffle
from typing import Iterable, Literal

def get_minimal_data(*trial_patterns:list[list[int]], 
                    n_trials:int = 10, 
                    stim_length:int = 100,
                    n_inputs:int = 6, to_tensor:bool = True,
                    noise_level:float | None = None
                    )->list|torch.Tensor:
    padding_length = stim_length * 4
    zeros = torch.zeros(padding_length, n_inputs)
    n_patterns = len(trial_patterns)
    trials = []
    for i in range(n_trials):
        pattern = trial_patterns[i % n_patterns]
        if not isinstance(pattern, torch.Tensor): pattern = torch.tensor(pattern)
        trials.append(torch.cat((zeros[:stim_length], pattern.tile(stim_length, 1), zeros), dim=0))
    if to_tensor:
        trials = torch.cat(trials, dim=0)
        if noise_level is not None:
            trials += noise_level * torch.randn_like(trials)
    return trials

def get_trial_type_minimal_data(trial_dict:dict[int:torch.Tensor], 
                                trial_types:Iterable[int],
                                n_trials_per_type:int = 10, stim_length:int = 100,
                                noise_level:float = 0.1
                                )->tuple[torch.Tensor, torch.Tensor]:
    # Get all trials
    total__trials = len(trial_types) * n_trials_per_type
    patterns = [trial_dict[trial_type] for trial_type in trial_types]
    trials = get_minimal_data(*patterns, n_trials=total__trials, stim_length=stim_length, to_tensor=False)
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

def get_patterns(stim_type:Literal['cell_types_and_novel', 'occlusion'] = 'cell_types_and_novel')-> tuple[list[int]]:
    '''
    Patterns designed to test the minimal model
    '''
    if stim_type == 'cell_types_and_novel':
        return ([1, 1, 1, 1, 1, 1], # activate all inputs (0)
                # activate individual pyramidal neurons (1-3)
                [1, 1, 0, 0, 0, 0], [0,0,1,1,0,0], [0,0,0,0,1,1],
                # activate alternating inputs to test PV inhibition (4-5)
                [1,0,1,0,1,0], [0,1,0,1,0,1],
                # test HVA feeedback by activating different combinations of PyCs (6-9)
                [1,0,0,1,0,0], [0,1,0,0,1,0], [0,0,1,0,0,1], [1,0,0,0,0,1],
                # one-side biased patterns to test training effects (10-13)
                [1,1,1,0,0,0], [1,1,0,1,0,0], [0,0,0,1,1,1], [0,0,1,0,1,1])
    
    elif stim_type == 'occlusion':
        # Given HVA tuning
        # HVA 0: [0.475, 0.475, 0.05], HVA 1: [0.05, 0.475, 0.475]

        # Given PyC tuning
        # Broadly tuned -> more FF adaptation, become FB experts
        # PyC 0: [0.5, 0.5, 0, 0, 0, 0], PyC 1: [0, 0, 0.5, 0.5, 0, 0], PyC 2: [0, 0, 0, 0, 0.5, 0.5]
        # Narrowly tuned
        # PyC 3: [0.99, 0.01, 0, 0, 0, 0], PyC 4: [0, 0, 0.99, 0.01, 0, 0], PyC 5: [0, 0, 0, 0, 0.01, 0.99]
        # PyC 6: [0.01, 0.99, 0, 0, 0, 0], PyC 7: [0, 0, 0.01, 0.99, 0, 0], PyC 8: [0, 0, 0, 0, 1.99, -1.99]
        return ([1, 1, 1, 1, 1, 1], # activate all inputs (0)
                # Full Images
                [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0], [0, 1, 0, 0, 0, 0])