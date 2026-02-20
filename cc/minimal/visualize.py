import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from cc.minimal import PLOTSDIR
from cc.figures import FigureBuilder

def visualize_experiment_results(DF:DataFrame, save_path:str = PLOTSDIR):
    long_df = wide_to_long(DF)
    pre_post_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()

    visualize_naive_expert_results(pre_post_df, save_path=save_path)
    return long_df


def visualize_naive_expert_results(pre_post_df:DataFrame, save_path:str = PLOTSDIR):
    phases = [p for p in ["naive", "expert"] if p in pre_post_df["experiment_phase"].unique()]
    image_types = sorted(pre_post_df["image_type"].dropna().unique().tolist()) if "image_type" in pre_post_df.columns else []
    conditions = sorted(pre_post_df["condition"].dropna().unique().tolist()) if "condition" in pre_post_df.columns else []
    y_df = pre_post_df[["step", "y", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    pv_df = pre_post_df[["step", "pv_value", "pv_index", "condition", "experiment_phase", "image_type"]].drop_duplicates()

    builder = FigureBuilder.from_matrix(
        [["A", "B", "D"],
         ["A", "C", "E"]],
        # [['B', 'C']], # simple 2-panel
        figsize=(20, 10),
        constrained_layout=False,
        grid_wspace=0.2,
        grid_hspace=0.2,
        subfigure_wspace=0.1,
        subfigure_hspace=0.1,
    )
    # TODO
    # A panel
    # Top 2x2 should contain simple imshow of cc/model_sketches/minimal_version1.png
    # Bottom 2x2 should contain the input stimuli 
    #   - (Top row should contain familiar image [Left X1, Right C1], 
    #   - bottom row should contain novel image [Left X2, Right C2])
    builder.update_panel("A", subgrid=(4, 2), title="Input stimuli", label="A")
    # B panel as is
    builder.update_panel("B", subgrid=(len(phases), len(conditions)), title="Y activity", label="B")
    # C panel as is
    builder.update_panel("C", subgrid=(len(phases), len(conditions)), title="PV activity", label="C")
    # D panel
    # Should contain two line plots below each other, 
    #   one showing y activity over taining, 
    #   other showing PC activity over training (separate lines for each PV neuron),
    builder.update_panel("D", subgrid=(2, 1), title="Y and PV activity over training", label="D")
    # E panel
    # Should contain four line-plots below each other, 
    # showing the evolution of the four weight matrices over training (indvidual weights different colored lines)
    builder.update_panel("E", subgrid=(4, 1), title="Weight evolution over training", label="E")

    def plot_y(ax_grid, _):
        for i, phase in enumerate(phases):
            for j, condition in enumerate(conditions):
                ax = ax_grid[i, j]
                cell = y_df[(y_df["experiment_phase"] == phase) & (y_df["condition"] == condition)]
                if cell.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=cell,
                    x="step",
                    y="y",
                    hue="image_type",
                    errorbar=None,
                    ax=ax,
                    legend=(i == 0 and j == 0),
                )
                ax.set_title(f"{phase} | {condition}")
                if i < len(phases) - 1:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

    def plot_pv(ax_grid, _):
        for i, phase in enumerate(phases):
            for j, condition in enumerate(conditions):
                ax = ax_grid[i, j]
                cell = pv_df[(pv_df["experiment_phase"] == phase) & (pv_df["condition"] == condition)]
                if cell.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=cell,
                    x="step",
                    y="pv_value",
                    hue="image_type",
                    style="pv_index",
                    errorbar=None,
                    ax=ax,
                    legend=(i == 0 and j == 0),
                )
                ax.set_title(f"{phase} | {condition}")
                if i < len(phases) - 1:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

    builder.set_plotter("B", plot_y)
    builder.set_plotter("C", plot_pv)

    os.makedirs(save_path, exist_ok=True)
    fig, _ = builder.render(save_path=os.path.join(save_path, "experiment_results.png"), show=False)
    plt.close(fig)

def wide_to_long(DF:DataFrame) -> DataFrame:
    """
    Convert the wide-format DataFrame to long-format for easier plotting with seaborn.
    """
    if "step" not in DF.columns:
        raise ValueError("Input DataFrame must contain a 'step' column.")
    n = len(DF)

    x_idx = sorted(
        int(m.group(1))
        for c in DF.columns
        for m in [re.match(r"^x_(\d+)$", c)]
        if m
    )
    pv_idx = sorted(
        int(m.group(1))
        for c in DF.columns
        for m in [re.match(r"^p_(\d+)$", c)]
        if m
    )
    if not x_idx or not pv_idx:
        return pd.DataFrame(columns=[
            "step", "y", "x_index", "x_value", "w_ff",
            "c_index", "c_value", "w_fb", "pv_index", "pv_value",
            "w_lat", "W_pv", "image_type", "condition", "experiment_phase", "seed",
        ])

    nx = len(x_idx)
    npv = len(pv_idx)
    rep = nx * npv

    step = np.repeat(DF["step"].to_numpy(), rep)
    y = np.repeat(DF["y"].to_numpy(), rep)
    x_index = np.tile(np.tile(np.array(x_idx, dtype=int), npv), n)
    pv_index = np.tile(np.repeat(np.array(pv_idx, dtype=int), nx), n)

    x_vals = DF[[f"x_{i}" for i in x_idx]].to_numpy()
    wff_vals = DF[[f"w_ff_{i}" for i in x_idx]].to_numpy()
    p_vals = DF[[f"p_{i}" for i in pv_idx]].to_numpy()
    wlat_vals = DF[[f"w_lat_{i}" for i in pv_idx]].to_numpy()

    x_value = np.tile(x_vals, (1, npv)).reshape(-1)
    w_ff = np.tile(wff_vals, (1, npv)).reshape(-1)
    pv_value = np.repeat(p_vals, nx, axis=1).reshape(-1)
    w_lat = np.repeat(wlat_vals, nx, axis=1).reshape(-1)

    c_cols = [f"c_{i}" for i in pv_idx]
    wfb_cols = [f"w_fb_{i}" for i in pv_idx]
    c_vals = DF[c_cols].to_numpy() if all(c in DF.columns for c in c_cols) else np.full((n, npv), np.nan)
    wfb_vals = DF[wfb_cols].to_numpy() if all(c in DF.columns for c in wfb_cols) else np.full((n, npv), np.nan)
    c_value = np.repeat(c_vals, nx, axis=1).reshape(-1)
    w_fb = np.repeat(wfb_vals, nx, axis=1).reshape(-1)

    wpv_grid = np.full((n, npv, nx), np.nan, dtype=float)
    for ip, p in enumerate(pv_idx):
        for ix, x in enumerate(x_idx):
            col = f"W_pv_{p}_{x}"
            if col in DF.columns:
                wpv_grid[:, ip, ix] = DF[col].to_numpy()
    W_pv = wpv_grid.reshape(-1)

    long_df = pd.DataFrame({
        "step": step,
        "y": y,
        "x_index": x_index,
        "x_value": x_value,
        "w_ff": w_ff,
        "c_index": pv_index,
        "c_value": c_value,
        "w_fb": w_fb,
        "pv_index": pv_index,
        "pv_value": pv_value,
        "w_lat": w_lat,
        "W_pv": W_pv,
    })

    if "seed" in DF.columns:
        long_df["seed"] = np.repeat(DF["seed"].to_numpy(), rep)

    if "condition" in DF.columns:
        cond = DF["condition"].astype(str).to_numpy()
        cond_rep = np.repeat(cond, rep)
        parts = pd.Series(cond_rep).str.split("_", n=2, expand=True)
        if parts.shape[1] == 3:
            long_df["image_type"] = parts[0].to_numpy()
            long_df["condition"] = parts[1].to_numpy()
            long_df["experiment_phase"] = parts[2].to_numpy()
        else:
            long_df["condition"] = cond_rep

    result_cols = [
        "step", "y", "x_index", "x_value", "w_ff",
        "c_index", "c_value", "w_fb", "pv_index", "pv_value",
        "w_lat", "W_pv", "image_type", "condition", "experiment_phase", "seed",
    ]
    return long_df[[c for c in result_cols if c in long_df.columns]]
