# author: Matúš Halák (@matushalak)
import torch
from pandas import DataFrame, concat as pd_concat
from cc.minimal.minimal import CCNeuron
from cc.minimal.utils import build_res, prepare_collect, collect_outputs
from cc.minimal.config import *
from cc.minimal.visualize import visualize_experiment_results, visualize_transition_panel
from cc.utils import randn_reparam

def design_experimental_phase(input_mean:torch.Tensor, input_var:torch.Tensor,
                              context_mean:torch.Tensor, context_var:torch.Tensor,
                              n_steps:int = 100, n_trials:int | None = 10,
                              intertrial_sigma:float = 0.05
                              ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Example experiment stimulation for using the minimal CCNeuron model.
        Generates random input and context sequences.
    """
    nzeros = 3 * n_steps // 4
    # Generate random input and context sequences according to provided distributions
    X = randn_reparam(size = (n_steps-nzeros,), mu = input_mean, sigma = input_var)
    C = randn_reparam(size = (n_steps-nzeros,), mu = context_mean, sigma = context_var)
    intertrial = randn_reparam(size=(nzeros, *X.shape[1:]), mu=0.0, sigma=intertrial_sigma)
    
    # append a few 0's to indicate initial state
    X = torch.cat((intertrial, X), dim=0)
    C = torch.cat((intertrial, C), dim=0)
    
    if n_trials is not None:
        X = X.repeat((n_trials, 1))
        C = C.repeat((n_trials, 1))
    
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
    X1, C1 = design_experimental_phase(input_mean=[1,0], input_var = 0.05,
                                       context_mean=[1,0], context_var=0.05,
                                       n_steps = n_steps_per_phase)
    # Image 2 ("novel", not trained on)
    X2, C2 = design_experimental_phase(input_mean=[0,1], input_var=0.05,
                                       context_mean=[0,1], context_var=0.05,
                                       n_steps = n_steps_per_phase)
    O = torch.zeros_like(X1) # occlusion (no input)
    
    STIMULI = {'familiar': (X1, C1), 'novel': (X2, C2)}

    # Initial test on all images without updates
    DF1 = run_experimental_phase(model, X1, C1, condition_name='full_familiar_naive', update=False)
    DF2 = run_experimental_phase(model, X2, C2, condition_name='full_novel_naive', update=False)
    DFO1 = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_naive', update=False)
    DFO2 = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_naive', update=False)
    DFNn = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_naive', update=False)

    # Now run the same sequences again with updates, to see how the model learns
    DF_training_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_training', update=True)
    
    # Now test everything again without changing weights
    DF_familiar = run_experimental_phase(model, X1, C1, condition_name='full_familiar_expert', update=False)
    DF_novel = run_experimental_phase(model, X2, C2, condition_name='full_novel_expert', update=False)
    DFO_familiar = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_expert', update=False)
    DFO_novel = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_expert', update=False)
    DFNe = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_expert', update=False)

    df = pd_concat(
        [
            DF1,
            DF2,
            DFO1,
            DFO2,
            DFNn,
            DF_training_familiar,
            DF_familiar,
            DF_novel,
            DFO_familiar,
            DFO_novel,
            DFNe,
        ],
        ignore_index=True,
    )
    df['seed'] = model_config['seed']

    return df, STIMULI


if __name__ == "__main__":
    long_dfs_by_transition: dict[str, DataFrame] = {}
    shared_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    include_novel_no_context = True

    for cfg_name, cfg in minimal_configs.items():
        print(f"Running experiment for config: {cfg_name}")
        df, STIMULI = run_experiment(cfg, n_steps_per_phase=400)
        # for now just return the long format dataframe for visualization
        long_df = visualize_experiment_results(
            df,
            STIMULI=STIMULI,
            name=cfg_name,
            include_novel_no_context=include_novel_no_context,
            xlim = (1000,1400)
        )
        long_dfs_by_transition[cfg_name] = long_df
        if shared_stimuli is None:
            shared_stimuli = STIMULI

    if shared_stimuli is not None:
        for image_mode in ("familiar", "novel", "both"):
            visualize_transition_panel(
                long_dfs_by_transition,
                STIMULI=shared_stimuli,
                name="transition_panel",
                image_mode=image_mode,
            )
