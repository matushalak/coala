import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
import torch
from cc.minimal import PLOTSDIR
from cc.figures import FigureBuilder

def visualize_experiment_results(DF:DataFrame, STIMULI:dict[str, tuple[torch.Tensor, torch.Tensor]], 
                                 save_path:str = PLOTSDIR, name:str = None)->DataFrame:
    long_df = wide_to_long(DF)
    # DF.to_csv(os.path.join(save_path, f"experiment_results_wide_{name}.csv"), index=False)   
    # long_df.to_csv(os.path.join(save_path, f"experiment_results_long_{name}.csv"), index=False)
    visualize_naive_expert_results(long_df, STIMULI=STIMULI, save_path=save_path, name=name)
    return long_df


def visualize_naive_expert_results(long_df:DataFrame, STIMULI:dict[str, tuple[torch.Tensor, torch.Tensor]], 
                                   save_path:str = PLOTSDIR, name:str = None) -> None:
    pre_post_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    phases = [p for p in ["naive", "expert"] if p in pre_post_df["experiment_phase"].unique()]
    image_types = sorted(pre_post_df["image_type"].dropna().unique().tolist()) if "image_type" in pre_post_df.columns else []
    conditions = sorted(pre_post_df["condition"].dropna().unique().tolist()) if "condition" in pre_post_df.columns else []
    y_df = pre_post_df[["step", "y", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    pv_df = pre_post_df[["step", "pv_value", "pv_index", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    training_rows = long_df.loc[long_df["experiment_phase"].eq("training")].copy()
    if training_rows.empty:
        training_rows = pre_post_df.copy()
    if {"image_type", "condition", "experiment_phase"}.issubset(long_df.columns):
        weight_rows = long_df.loc[
            long_df["image_type"].eq("full")
            & long_df["condition"].eq("familiar")
            & long_df["experiment_phase"].eq("training")
        ].copy()
    else:
        weight_rows = pd.DataFrame()
    if weight_rows.empty:
        weight_rows = training_rows.copy()

    builder = FigureBuilder.from_matrix(
        [["B", "B", "D"],
         ["C", "C", "E"]],
        figsize=(20, 15),
        constrained_layout=False,
        grid_wspace=0.25,
        grid_hspace=0.15,
        subfigure_wspace=0.15,
        subfigure_hspace=0.2,
    )
    # builder.update_panel("A", subgrid=(3, 2), title=None, label=None)
    builder.update_panel("B", subgrid=(len(phases), len(conditions)), title="Y activity", label="B")
    builder.update_panel("C", subgrid=(len(phases), len(conditions)), title="PV activity", label="C")
    builder.update_panel("D", subgrid=(2, 1), title="Y and PV activity over training", label="D")
    builder.update_panel("E", subgrid=(4, 1), title="Weight evolution over training", label="E")

    x_colors = {0: "green", 1: "gold"}
    c_colors = {0: "magenta", 1: "navy"}
    image_colors = {"full": "black", "occlusion": "red", "novel_no_context": "cyan"}
    pv_colors = {0: "red", 1: "pink"}

    def _to_np(ts: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(ts, torch.Tensor):
            arr = ts.detach().cpu().numpy()
        else:
            arr = np.asarray(ts)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
            arr = arr.T
        return arr

    def _get_stim_pair(name: str) -> tuple[np.ndarray, np.ndarray]:
        default = (np.zeros((1, 2), dtype=float), np.zeros((1, 2), dtype=float))
        pair = STIMULI.get(name, default)
        return _to_np(pair[0]), _to_np(pair[1])

    X1, C1 = _get_stim_pair("familiar")
    X2, C2 = _get_stim_pair("novel")

    def _ensure_two_channels(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[1] < 2:
            arr = np.hstack([arr, np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=float)])
        elif arr.shape[1] > 2:
            arr = arr[:, :2]
        return arr

    X1 = _ensure_two_channels(X1)
    C1 = _ensure_two_channels(C1)
    X2 = _ensure_two_channels(X2)
    C2 = _ensure_two_channels(C2)

    def plot_y(ax_grid, _):
        for i in range(len(phases)):
            for j in range(len(conditions)):
                if i == 0 and j == 0:
                    continue
                ax_grid[i, j].sharey(ax_grid[0, 0])
        for i, phase in enumerate(phases):
            for j, condition in enumerate(conditions):
                ax = ax_grid[i, j]
                cell = y_df[(y_df["experiment_phase"] == phase) & (y_df["condition"] == condition)]
                cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
                if cell.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=cell,
                    x="step",
                    y="y",
                    hue="image_type",
                    hue_order=[k for k in ["full", "occlusion", "novel_no_context"] if k in image_types],
                    style="image_type",
                    palette=image_colors,
                    errorbar=None,
                    ax=ax,
                    legend=(i == 0 and j == 0),
                )
                ax.set_title(f"{phase} | {condition}")
                if 'un_un' in name:
                    ax.set_ylim(0, 0.3)
                if i < len(phases) - 1:
                    ax.set_xlabel("")
                    ax.tick_params(labelbottom=False)
                if j > 0:
                    ax.set_ylabel("")
                    ax.tick_params(labelleft=False)

    def plot_pv(ax_grid, _):
        for i in range(len(phases)):
            for j in range(len(conditions)):
                if i == 0 and j == 0:
                    continue
                ax_grid[i, j].sharey(ax_grid[0, 0])
        for i, phase in enumerate(phases):
            for j, condition in enumerate(conditions):
                ax = ax_grid[i, j]
                cell = pv_df[(pv_df["experiment_phase"] == phase) & (pv_df["condition"] == condition)]
                cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
                if cell.empty:
                    ax.set_visible(False)
                    continue
                sns.lineplot(
                    data=cell,
                    x="step",
                    y="pv_value",
                    hue="image_type",
                    hue_order=[k for k in ["novel_no_context","full", "occlusion"] if k in image_types],
                    palette=image_colors,
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

    def plot_panel_a(ax_grid, _):
        sketch_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_sketches")
        sketch_path = os.path.join(sketch_dir, "minimal_version1_small.png")
        img = plt.imread(sketch_path) if os.path.exists(sketch_path) else None
        left = ax_grid[0, 0]
        right = ax_grid[0, 1]
        pos_l = left.get_position()
        pos_r = right.get_position()
        x0 = min(pos_l.x0, pos_r.x0)
        y0 = min(pos_l.y0, pos_r.y0)
        x1 = max(pos_l.x1, pos_r.x1)
        y1 = max(pos_l.y1, pos_r.y1)
        right.remove()
        left.set_position([x0, y0, x1 - x0, y1 - y0])
        left.axis("off")
        if img is not None:
            left.imshow(img)
        left.text(
            -0.12,
            1.08,
            "A",
            transform=left.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            clip_on=False,
        )

        ax_grid[1, 1].sharey(ax_grid[1, 0])
        ax_grid[2, 1].sharey(ax_grid[2, 0])

        stim_cells = [
            (1, 0, X1, "X1", "x"),
            (1, 1, C1, "C1", "c"),
            (2, 0, X2, "X2", "x"),
            (2, 1, C2, "C2", "c"),
        ]
        for r, c, series, title, kind in stim_cells:
            ax = ax_grid[r, c]
            colors = x_colors if kind == "x" else c_colors
            n_steps = series.shape[0]
            ax.plot(np.arange(n_steps), series[:, 0], color=colors[0], lw=1.5, label=f"{title.lower()}_0")
            ax.plot(np.arange(n_steps), series[:, 1], color=colors[1], lw=1.5, label=f"{title.lower()}_1")
            ax.set_title(title)
            ax.set_xlim(0, max(1, n_steps - 1))
            if r == 1:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("step")

        for r, left_series, right_series in [(1, X1, C1), (2, X2, C2)]:
            row_vals = np.concatenate([left_series.ravel(), right_series.ravel()])
            y_min = float(np.nanmin(row_vals))
            y_max = float(np.nanmax(row_vals))
            span = y_max - y_min
            pad = 0.05 * span if span > 0 else max(0.1, 0.05 * max(abs(y_min), abs(y_max), 1.0))
            ax_grid[r, 0].set_ylim(y_min - pad, y_max + pad)
            ax_grid[r, 1].set_ylim(y_min - pad, y_max + pad)
            ax_grid[r, 1].tick_params(labelleft=False)

    def plot_training_activity(ax_grid, _):
        step_familiar = np.arange(X1.shape[0])
        for idx in range(min(2, X1.shape[1])):
            ax_grid[0, 0].plot(step_familiar, X1[:, idx], color=x_colors[idx], lw=1.5, label=f"x_{idx}")
        for idx in range(min(2, C1.shape[1])):
            ax_grid[0, 0].plot(step_familiar, C1[:, idx], color=c_colors[idx], linestyle='--', lw=1.5, label=f"c_{idx}")
        ax_grid[0, 0].set_title("Training (familiar) input/context (X1, C1)")
        ax_grid[0, 0].set_xlabel("")
        ax_grid[0, 0].tick_params(labelbottom=False)

        y_train = training_rows[["step", "y"]].drop_duplicates().groupby("step", as_index=False)["y"].mean()
        pv_train = (
            training_rows[["step", "pv_index", "pv_value"]]
            .drop_duplicates()
            .groupby(["step", "pv_index"], as_index=False)["pv_value"]
            .mean()
        )
        ax_grid[1, 0].plot(y_train["step"], y_train["y"], color="black", lw=1.6, label="y")
        for pv_idx, cell in pv_train.groupby("pv_index", sort=True):
            ax_grid[1, 0].plot(
                cell["step"],
                cell["pv_value"],
                color=pv_colors.get(int(pv_idx), None),
                lw=1.4,
                label=f"pv_{pv_idx}",
            )
        # ax_grid[1,0].set_yscale("log")
        ax_grid[1, 0].set_title("Training Y and PV activity")
        ax_grid[1, 0].set_xlabel("step")

    def plot_weight_evolution(ax_grid, _):
        wff = (
            weight_rows[["step", "x_index", "w_ff"]]
            .drop_duplicates()
            .dropna(subset=["w_ff"])
            .sort_values(["x_index", "step"])
        )
        wfb = (
            weight_rows[["step", "c_index", "w_fb"]]
            .drop_duplicates()
            .dropna(subset=["w_fb"])
            .sort_values(["c_index", "step"])
        )
        wlat = (
            weight_rows[["step", "pv_index", "w_lat"]]
            .drop_duplicates()
            .dropna(subset=["w_lat"])
            .sort_values(["pv_index", "step"])
        )
        wpv = (
            weight_rows[["step", "pv_index", "x_index", "W_pv"]]
            .drop_duplicates()
            .dropna(subset=["W_pv"])
            .sort_values(["pv_index", "x_index", "step"])
        )
        wpv["pair"] = "pv" + wpv["pv_index"].astype(str) + "-x" + wpv["x_index"].astype(str)

        sns.lineplot(
            data=wff,
            x="step",
            y="w_ff",
            hue="x_index",
            hue_order=[0, 1],
            palette=x_colors,
            errorbar=None,
            ax=ax_grid[0, 0],
        )
        ax_grid[0, 0].set_title("Training w_ff evolution")
        sns.lineplot(
            data=wfb,
            x="step",
            y="w_fb",
            hue="c_index",
            hue_order=[0, 1],
            palette=c_colors,
            errorbar=None,
            ax=ax_grid[1, 0],
        )
        ax_grid[1, 0].set_title("Training w_fb evolution")
        sns.lineplot(
            data=wlat,
            x="step",
            y="w_lat",
            hue="pv_index",
            hue_order=[0, 1],
            palette=pv_colors,
            errorbar=None,
            ax=ax_grid[2, 0],
        )
        ax_grid[2, 0].set_title("Training w_lat evolution")
        sns.lineplot(data=wpv, x="step", y="W_pv", hue="pair", errorbar=None, ax=ax_grid[3, 0])
        ax_grid[3, 0].set_title("Training W_pv evolution")

        for i in range(ax_grid.shape[0]):
            # ax_grid[i, 0].set_yscale("log")
            if i < ax_grid.shape[0] - 1:
                ax_grid[i, 0].set_xlabel("")
                ax_grid[i, 0].tick_params(labelbottom=False)

    # builder.set_plotter("A", plot_panel_a)
    builder.set_plotter("B", plot_y)
    builder.set_plotter("C", plot_pv)
    builder.set_plotter("D", plot_training_activity)
    builder.set_plotter("E", plot_weight_evolution)

    os.makedirs(save_path, exist_ok=True)
    fig, _ = builder.render(save_path=os.path.join(save_path, f"experiment_results_{name}.png"), show=False)
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
        parts = pd.Series(cond_rep).str.rsplit("_", n=1, expand=True)
        if parts.shape[1] == 2:
            prefix = parts[0]
            long_df["experiment_phase"] = parts[1].to_numpy()
            long_df["condition"] = np.where(
                prefix.str.contains("_novel_", regex=False),
                "novel",
                prefix.str.split("_", n=1).str[1],
            )
            long_df["image_type"] = np.where(
                prefix.eq("full_novel_nocontext"),
                "novel_no_context",
                prefix.str.split("_", n=1).str[0],
            )
        else:
            long_df["condition"] = cond_rep

    result_cols = [
        "step", "y", "x_index", "x_value", "w_ff",
        "c_index", "c_value", "w_fb", "pv_index", "pv_value",
        "w_lat", "W_pv", "image_type", "condition", "experiment_phase", "seed",
    ]
    return long_df[[c for c in result_cols if c in long_df.columns]]
