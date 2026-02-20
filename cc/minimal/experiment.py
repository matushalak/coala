# author: Matúš Halák (@matushalak)
import torch
from pandas import DataFrame, concat as pd_concat
from cc.minimal.minimal import CCNeuron
from cc.minimal.utils import build_res, prepare_collect, collect_outputs
from cc.minimal.config import basic
from cc.minimal.visualize import visualize_experiment_results
from cc.utils import randn_reparam

def design_experimental_phase(input_mean:torch.Tensor, input_var:torch.Tensor,
                              context_mean:torch.Tensor, context_var:torch.Tensor,
                              n_steps:int = 100):
    """
    Example experiment stimulation for using the minimal CCNeuron model.
        Generates random input and context sequences.
    """
    nzeros = 10
    # Generate random input and context sequences according to provided distributions
    X = randn_reparam(size = (n_steps-nzeros,), mu = input_mean, sigma = input_var)
    C = randn_reparam(size = (n_steps-nzeros,), mu = context_mean, sigma = context_var)
    # append a few 0's to indicate initial state
    X = torch.cat((X.new_zeros((nzeros, *X.shape[1:])), X), dim=0)
    C = torch.cat((C.new_zeros((nzeros, *C.shape[1:])), C), dim=0)

    return [X, C] # Image consists of [X, C]

def run_experimental_phase(model:CCNeuron, X:torch.Tensor, C:torch.Tensor,
                           condition_name:str = 'default', 
                           update:bool = False, reset_rates:bool = True)->DataFrame:
    """
    Run the model over an experimental sequence.
    """
    # Prepare collections for output data
    data_collection = prepare_collect()
    
    if reset_rates: # reset pyc and pv rates to zero before starting the phase
        model._reset_state()

    # Run the model over the sequence and collect outputs
    for step in range(X.shape[0]):
        x, y, p, c = model(X[step], C[step])
        if update:
            model.update(x, y, p, c)
        
        # Collect the raw tensors
        data_collection = collect_outputs(step, x, y, p, c, model, data_collection)
    
    # Make data frame from collected data
    DF:DataFrame = build_res(data_collection, model)
    # broadcast condition name to new column and all rows of dataframe
    DF['condition'] = condition_name
    return DF

def run_experiment(model_config:dict, n_steps_per_phase:int = 100) -> DataFrame:
    model = CCNeuron(**model_config)

    # Image 1 ("familiar", trained on)
    X1, C1 = design_experimental_phase(input_mean=[3,0], input_var = [1, 0.05],
                                       context_mean=[-3,0], context_var=[1.5, 1.5],
                                       n_steps = n_steps_per_phase)
    # Image 2 ("novel", not trained on)
    X2, C2 = design_experimental_phase(input_mean=[0,3], input_var=[0.05, 1],
                                       context_mean=[0,-3], context_var=[1.5, 1.5],
                                       n_steps = n_steps_per_phase)
    O = torch.zeros_like(X1) # occlusion (no input)

    # Initial test on all images without updates
    DF1 = run_experimental_phase(model, X1, C1, condition_name='full_familiar_naive', update=False)
    DF2 = run_experimental_phase(model, X2, C2, condition_name='full_novel_naive', update=False)
    DFO1 = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_naive', update=False)
    DFO2 = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_naive', update=False)

    # Now run the same sequences again with updates, to see how the model learns
    DF_training_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_training', update=True)
    
    # Now test everything again without changing weights
    DF_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_expert', update=False)
    DF_novel = run_experimental_phase(model, X2, C2, condition_name='full_novel_expert', update=False)
    DFO_familiar = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_expert', update=False)
    DFO_novel = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_expert', update=False)

    df = pd_concat([DF1, DF2, DFO1, DFO2, DF_training_familiar, DF_familiar, DF_novel, DFO_familiar, DFO_novel], ignore_index=True)
    df['seed'] = model_config['seed']
    return df


if __name__ == "__main__":
    # Example usage
    model_config = basic
    df = run_experiment(model_config, n_steps_per_phase=200)
    # for now just return the long format dataframe for visualization
    df = visualize_experiment_results(df)
