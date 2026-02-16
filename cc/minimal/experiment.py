from typing import Literal
import torch
import numpy as np
from data import get_minimal_data, get_patterns, get_trial_dict, get_trial_type_minimal_data
from minimal import Minimal
from utils import get_hva_tuning, plot_out

def test_minimal():
    torch.manual_seed(2026)  # for reproducibility
    HVA_weights = get_hva_tuning([0.45, 0.1, 0.45], [0.05, 0.9, 0.05])
    model = Minimal(HVA_tuning=HVA_weights)
    
    trial_patterns = get_patterns() # Hardcoded trial patterns for testing
    I = get_minimal_data(*trial_patterns, n_trials=10, trial_length=50, to_tensor=True)
   
    output = model(I)
    plot_out(output, I)

def experiment(train_trials:list[int], trial_length:int = 80, 
               training_trials_per_type:int = 10, noise_level:float = 0.1,
               feedback_rule:Literal['Hebbian', 'Anti-Hebbian'] = 'Hebbian'):
    '''
    Based selection of training trials; 
        different connections will become adapted / strengthened
    
    TODO: make config and load models & save results with config info
    '''
    torch.manual_seed(2026)  # for reproducibility
    np.random.seed(2026)
    # HVA 0: border regions (Pyr 0 and Pyr 2), HVA 1: center region (Pyr 1)
    HVA_weights = get_hva_tuning([0.475, 0.475, 0.05], [0.05, 0.475, 0.475])
    model = Minimal(HVA_tuning=HVA_weights, feedback_rule=feedback_rule)
    
    trial_patterns = get_patterns() # Hardcoded trial patterns for testing
    trial_dict = get_trial_dict(*trial_patterns)

    # Get training data for specified trial types
    train_tensor = get_trial_type_minimal_data(trial_dict,
                                               trial_types=train_trials, 
                                               n_trials_per_type=10,
                                               trial_length=trial_length,
                                               noise_level=noise_level)

    # Test all trials before training
    I_all_trials = get_minimal_data(*trial_patterns, n_trials=14, trial_length=trial_length, 
                                    to_tensor=True, noise_level=noise_level)
    activity_initial_all_trials = model(I_all_trials, train=False)
    plot_out(activity_initial_all_trials, I_all_trials, title='Model Activity Before Training', save=True)

    # Train the model on specified trial types
    activity_training_trials = model(train_tensor, train=True)
    plot_out(activity_training_trials, train_tensor, title='Model Activity During Training', save=True)

    # Test all trials after training
    activity_final_all_trials = model(I_all_trials, train=False)
    plot_out(activity_final_all_trials, I_all_trials, title='Model Activity After Training', save=True)

if __name__ == "__main__":
    # test_minimal()
    # train on left-ish patterns (expect stimulus more on the left)
    # Hebbian feedback rule
    experiment(train_trials=[1, 2, 6, 10, 11], noise_level=0.1, trial_length=100, training_trials_per_type=20, 
               feedback_rule='Hebbian')
    # Anti-Hebbian feedback rule
    # experiment(train_trials=[1, 2, 6, 10, 11], noise_level=0.1, trial_length=100, training_trials_per_type=20, 
    #            feedback_rule='Anti-Hebbian')