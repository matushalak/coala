import os

from cc.minimal.config import minimal_configs

ABLATION_PLOTSDIR = os.path.join(os.path.dirname(__file__), "plots-ablations")
os.makedirs(ABLATION_PLOTSDIR, exist_ok=True)
EXPERIMENT_PLOTS_DIR_KEY = "_plots_dir"

ABLATION_COMPONENTS = (
    "use_FF_connection",
    "FF_plasticity",
    "use_FB_connection",
    "FB_plasticity",
    "use_lat_connection",
    "lat_plasticity",
    "use_pv_connection",
    "pv_plasticity",
)


def build_ablation_configs(base_configs: dict[str, dict]) -> dict[str, dict]:
    configs_with_ablations = {
        name: {**config, EXPERIMENT_PLOTS_DIR_KEY: ABLATION_PLOTSDIR}
        for name, config in base_configs.items()
    }

    for config_name, config in base_configs.items():
        for component in ABLATION_COMPONENTS:
            ablated_config = config.copy()
            ablated_config[component] = False
            ablated_config[EXPERIMENT_PLOTS_DIR_KEY] = ABLATION_PLOTSDIR
            configs_with_ablations[f"{config_name}__no_{component}"] = ablated_config

    return configs_with_ablations


ablation_configs = build_ablation_configs(minimal_configs)
minimal_ablation_configs = ablation_configs
