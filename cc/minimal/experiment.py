# author: Matúš Halák (@matushalak)
import os

import torch
from pandas import DataFrame, concat as pd_concat
from cc.minimal import PLOTSDIR
from cc.minimal.minimal import CCNeuron
from cc.minimal.utils import build_res, prepare_collect, collect_outputs
from cc.minimal.config import *
from cc.minimal.ablations import ABLATION_COMPONENTS, minimal_ablation_configs
from cc.minimal.visualize import TRANSITION_LABELS, visualize_experiment_results, visualize_transition_panel
from cc.utils import randn_reparam

EXPERIMENT_METADATA_PREFIX = "_"
ABLATION_NAME_SEPARATOR = "__no_"
PRIMARY_EXPERIMENT_SERIES = "training_familiar"
OCCLUDED_ONLY_EXPERIMENT_SERIES = "training_occluded_only"


def _model_kwargs(model_config: dict) -> dict:
    return {key: value for key, value in model_config.items() if not key.startswith(EXPERIMENT_METADATA_PREFIX)}


def _resolve_plots_dir(model_config: dict) -> str:
    return model_config.get("_plots_dir", PLOTSDIR)


def _tag_experiment_series(df: DataFrame, experiment_series: str) -> DataFrame:
    tagged = df.copy()
    tagged["experiment_series"] = experiment_series
    return tagged


def _iter_experiment_series(long_df: DataFrame) -> list[tuple[str | None, str, DataFrame]]:
    if "experiment_series" not in long_df.columns:
        return [(None, "", long_df)]

    series_names = long_df["experiment_series"].dropna().unique().tolist()
    if not series_names:
        return [(None, "", long_df)]

    series_entries: list[tuple[str | None, str, DataFrame]] = []
    for idx, series_name in enumerate(series_names):
        subset = long_df.loc[long_df["experiment_series"].eq(series_name)].copy()
        if subset.empty:
            continue
        suffix = "" if idx == 0 else f"_{series_name}"
        series_entries.append((series_name, suffix, subset))

    return series_entries or [(None, "", long_df)]


def _save_grouped_ablation_transition_panels(
    long_dfs_by_transition: dict[str, DataFrame],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str,
    base_transition_order: list[str],
) -> bool:
    saved_any = False
    sample_df = next(iter(long_dfs_by_transition.values()), None)
    series_entries = _iter_experiment_series(sample_df) if sample_df is not None else [(None, "", DataFrame())]

    for series_name, series_suffix, _ in series_entries:
        for ablation_name in ABLATION_COMPONENTS:
            grouped_transitions: dict[str, DataFrame] = {}
            transition_labels: dict[str, str] = {}

            for base_name in base_transition_order:
                config_name = f"{base_name}{ABLATION_NAME_SEPARATOR}{ablation_name}"
                long_df = long_dfs_by_transition.get(config_name)
                if long_df is None:
                    continue

                if series_name is not None and "experiment_series" in long_df.columns:
                    long_df = long_df.loc[long_df["experiment_series"].eq(series_name)].copy()
                    if long_df.empty:
                        continue

                grouped_transitions[config_name] = long_df
                transition_labels[config_name] = TRANSITION_LABELS.get(base_name, base_name)

            if not grouped_transitions:
                continue

            generated_path = visualize_transition_panel(
                grouped_transitions,
                STIMULI=STIMULI,
                save_path=save_path,
                name=f"transition_panel_{ablation_name}{series_suffix}",
                image_mode="both",
                transition_order=list(grouped_transitions),
                transition_labels=transition_labels,
            )
            target_path = os.path.join(
                save_path,
                f"transition_panel_familiar_novel_{ablation_name}{series_suffix}.png",
            )
            if generated_path != target_path:
                os.replace(generated_path, target_path)
            saved_any = True

    return saved_any

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
    model = CCNeuron(**_model_kwargs(model_config))

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

    # Continue from the already-trained state, but train only on the occluded familiar stimulus.
    DF_training_familiar_occluded_only = run_experimental_phase(
        model,
        O,
        C1,
        condition_name='occlusion_familiar_training',
        update=True,
    )

    # Now test everything again without changing weights
    DF_familiar_occluded_only = run_experimental_phase(model, X1, C1, condition_name='full_familiar_expert', update=False)
    DF_novel_occluded_only = run_experimental_phase(model, X2, C2, condition_name='full_novel_expert', update=False)
    DFO_familiar_occluded_only = run_experimental_phase(model, O, C1, condition_name='occlusion_familiar_expert', update=False)
    DFO_novel_occluded_only = run_experimental_phase(model, O, C2, condition_name='occlusion_novel_expert', update=False)
    DFNe_occluded_only = run_experimental_phase(model, X2, O, condition_name='full_novel_nocontext_expert', update=False)

    df = pd_concat(
        [
            _tag_experiment_series(DF1, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF2, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO1, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO2, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFNn, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_training_familiar, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_familiar, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_novel, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO_familiar, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO_novel, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFNe, PRIMARY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF1, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF2, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO1, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO2, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFNn, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_training_familiar_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_familiar_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DF_novel_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO_familiar_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFO_novel_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
            _tag_experiment_series(DFNe_occluded_only, OCCLUDED_ONLY_EXPERIMENT_SERIES),
        ],
        ignore_index=True,
    )
    df['seed'] = model_config['seed']

    return df, STIMULI


if __name__ == "__main__":
    long_dfs_by_transition: dict[str, DataFrame] = {}
    shared_stimuli: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    shared_plots_dir: str | None = None
    include_novel_no_context = True

    # for cfg_name, cfg in minimal_configs.items():
    for cfg_name, cfg in minimal_ablation_configs.items():
        plots_dir = _resolve_plots_dir(cfg)
        print(f"Running experiment for config: {cfg_name}")
        df, STIMULI = run_experiment(cfg, n_steps_per_phase=400)
        # for now just return the long format dataframe for visualization
        long_df = visualize_experiment_results(
            df,
            STIMULI=STIMULI,
            save_path=plots_dir,
            name=cfg_name,
            include_novel_no_context=include_novel_no_context,
            xlim = (1000,1400)
        )
        long_dfs_by_transition[cfg_name] = long_df
        if shared_stimuli is None:
            shared_stimuli = STIMULI
        if shared_plots_dir is None:
            shared_plots_dir = plots_dir

    if shared_stimuli is not None and shared_plots_dir is not None:
        _save_grouped_ablation_transition_panels(
            long_dfs_by_transition,
            STIMULI=shared_stimuli,
            save_path=shared_plots_dir,
            base_transition_order=list(minimal_configs),
        )
