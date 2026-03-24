import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas import DataFrame
import torch
from typing import Literal
from cc.minimal import PLOTSDIR
from cc.figures import FigureBuilder

PLOT_CONDITION_LABELS = {
    "full": "Nonoccluded",
    "occlusion": "Occluded",
    "novel_no_context": "No feedback",
}
PLOT_CONDITION_ORDER = ["full", "occlusion", "novel_no_context"]
PLOT_COLORS = {"Nonoccluded": "black", "Occluded": "red", "No feedback": "blue"}
TRANSITION_ORDER = [
    "un_un",
    "un_FF",
    "un_FB",
    "FF_un",
    "FF_FF",
    "FF_FB_broad",
    "FF_FB_narrow_familiar",
    "FF_FB_narrow_novel",
    "FB_FB",
]
TRANSITION_LABELS = {
    "un_un": "un -> un",
    "un_FF": "un -> FF",
    "un_FB": "un -> FB",
    "FF_un": "FF -> un",
    "FF_FF": "FF -> FF",
    "FF_FB_broad": "FF -> FB\n(broad)",
    "FF_FB_narrow_familiar": "FF -> FB\n(narrow fam)",
    "FF_FB_narrow_novel": "FF -> FB\n(narrow nov)",
    "FB_FB": "FB -> FB",
}
TRACE_COLORS = {"full": "black", "occlusion": "red"}
TRACE_LABELS = {"full": "Nonoccluded", "occlusion": "Occluded"}
IMAGE_LABELS = {"familiar": "Familiar Image", "novel": "Novel Image"}


def _add_plot_condition_labels(df: DataFrame) -> DataFrame:
    styled = df.copy()
    if "image_type" in styled.columns:
        styled["plot_condition"] = styled["image_type"].map(PLOT_CONDITION_LABELS).fillna(styled["image_type"])
    return styled


def _plot_condition_order(image_types: list[str]) -> list[str]:
    return [PLOT_CONDITION_LABELS[k] for k in PLOT_CONDITION_ORDER if k in image_types]


def _to_np_2d(ts: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(ts, torch.Tensor):
        arr = ts.detach().cpu().numpy()
    else:
        arr = np.asarray(ts)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] != 2:
        arr = arr.T
    return arr


def _ensure_two_channels(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.shape[1] < 2:
        arr = np.hstack([arr, np.zeros((arr.shape[0], 2 - arr.shape[1]), dtype=float)])
    elif arr.shape[1] > 2:
        arr = arr[:, :2]
    return arr


def _resolve_image_mode(
    image_mode: Literal["familiar", "novel", "both"] | None,
    include_novel_image: bool | None,
) -> list[str]:
    if image_mode is None:
        image_mode = "both" if include_novel_image else "familiar"
    if image_mode not in {"familiar", "novel", "both"}:
        raise ValueError("image_mode must be one of 'familiar', 'novel', or 'both'.")
    if image_mode == "both":
        return ["familiar", "novel"]
    return [image_mode]


def _infer_trial_layout(stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray]) -> dict[str, int]:
    x = _ensure_two_channels(_to_np_2d(stim_pair[0]))
    c = _ensure_two_channels(_to_np_2d(stim_pair[1]))
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    total_steps = int(stimulus_strength.shape[0])
    if total_steps == 0:
        return {"trial_len": 1, "stim_onset": 0, "stim_offset": 1}

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return {"trial_len": total_steps, "stim_onset": 0, "stim_offset": total_steps}

    active_threshold = max(0.2 * peak, 0.15)
    active = stimulus_strength > active_threshold
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1

    if onsets.size >= 2:
        trial_len = int(np.round(np.median(np.diff(onsets))))
    else:
        trial_len = total_steps
    trial_len = max(1, min(trial_len, total_steps))

    stim_onset = int(onsets[0]) if onsets.size else 0
    stim_onset = max(0, min(stim_onset, trial_len - 1))

    if offsets.size:
        stim_offset = int(offsets[0])
        if stim_offset > trial_len:
            stim_offset = trial_len
        if stim_offset <= stim_onset:
            stim_offset = min(trial_len, stim_onset + max(1, trial_len // 2))
    else:
        stim_offset = min(trial_len, stim_onset + max(1, trial_len // 2))

    return {"trial_len": trial_len, "stim_onset": stim_onset, "stim_offset": stim_offset}


def _extract_repeated_y(
    long_df: DataFrame,
    condition: str,
    phase: str,
    image_type: str,
) -> np.ndarray:
    cell = long_df.loc[
        long_df["condition"].eq(condition)
        & long_df["experiment_phase"].eq(phase)
        & long_df["image_type"].eq(image_type),
        ["step", "y"],
    ].drop_duplicates()
    if cell.empty:
        return np.asarray([], dtype=float)
    return cell.sort_values("step")["y"].to_numpy(dtype=float)


def _extract_trace_window(
    long_df: DataFrame,
    condition: str,
    phase: str,
    image_type: str,
    step_window: tuple[int, int],
) -> DataFrame:
    start, end = step_window
    cell = long_df.loc[
        long_df["condition"].eq(condition)
        & long_df["experiment_phase"].eq(phase)
        & long_df["image_type"].eq(image_type),
        ["step", "y"],
    ].drop_duplicates()
    if cell.empty:
        return cell
    return cell.loc[(cell["step"] > start) & (cell["step"] < end)].sort_values("step")


def _summarize_repeated_trace(
    series: np.ndarray,
    trial_len: int,
    baseline_stop: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = np.asarray(series, dtype=float).reshape(-1)
    if series.size == 0:
        return (
            np.arange(max(1, trial_len), dtype=float),
            np.zeros(max(1, trial_len), dtype=float),
            np.zeros(max(1, trial_len), dtype=float),
        )

    trial_len = max(1, int(trial_len))
    n_trials = max(1, series.size // trial_len)
    trimmed = series[: n_trials * trial_len].reshape(n_trials, trial_len)

    baseline_stop = max(0, min(int(baseline_stop), trial_len))
    if baseline_stop > 0:
        baseline = trimmed[:, :baseline_stop].mean(axis=1, keepdims=True)
        trimmed = trimmed - baseline

    mean_trace = trimmed.mean(axis=0)
    if trimmed.shape[0] > 1:
        sem_trace = trimmed.std(axis=0, ddof=1) / np.sqrt(trimmed.shape[0])
    else:
        sem_trace = np.zeros_like(mean_trace)

    return np.arange(trial_len, dtype=float), mean_trace, sem_trace


def _find_stimulus_interval_in_window(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    step_window: tuple[int, int],
) -> tuple[float, float] | None:
    x = _ensure_two_channels(_to_np_2d(stim_pair[0]))
    c = _ensure_two_channels(_to_np_2d(stim_pair[1]))
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return None

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return None

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    start, end = step_window

    for onset, offset in zip(onsets, offsets, strict=False):
        if onset < end and offset > start:
            return float(max(onset, start)), float(min(offset, end))

    return None


def _expand_window_to_event_bounds(
    stim_pair: tuple[torch.Tensor, torch.Tensor] | tuple[np.ndarray, np.ndarray],
    focus_window: tuple[int, int],
) -> tuple[int, int]:
    x = _ensure_two_channels(_to_np_2d(stim_pair[0]))
    c = _ensure_two_channels(_to_np_2d(stim_pair[1]))
    stimulus_strength = np.maximum(np.abs(x).max(axis=1), np.abs(c).max(axis=1))
    if stimulus_strength.size == 0:
        return focus_window

    peak = float(np.nanmax(stimulus_strength))
    if not np.isfinite(peak) or peak <= 0:
        return focus_window

    active = stimulus_strength > max(0.2 * peak, 0.15)
    onsets = np.flatnonzero(np.diff(np.r_[0, active.astype(int)]) == 1)
    offsets = np.flatnonzero(np.diff(np.r_[active.astype(int), 0]) == -1) + 1
    if onsets.size == 0 or offsets.size == 0:
        return focus_window

    start, end = focus_window
    best_idx = None
    best_overlap = -1
    for idx, (onset, offset) in enumerate(zip(onsets, offsets, strict=False)):
        overlap = min(end, offset) - max(start, onset)
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx = idx

    if best_idx is None or best_overlap <= 0:
        return focus_window

    current_onset = int(onsets[best_idx])
    current_offset = int(offsets[best_idx])

    if best_idx > 0:
        prev_offset = int(offsets[best_idx - 1])
        expanded_start = int(round(0.5 * (prev_offset + current_onset)))
    else:
        expanded_start = 0

    if best_idx < len(onsets) - 1:
        next_onset = int(onsets[best_idx + 1])
        expanded_end = int(round(0.5 * (current_offset + next_onset)))
    else:
        expanded_end = int(stimulus_strength.size)

    return expanded_start, expanded_end


def visualize_transition_panel(
    long_dfs_by_transition: dict[str, DataFrame],
    STIMULI: dict[str, tuple[torch.Tensor, torch.Tensor]],
    save_path: str = PLOTSDIR,
    name: str = "transition_panel",
    image_mode: Literal["familiar", "novel", "both"] | None = None,
    include_novel_image: bool | None = None,
    transition_order: list[str] | None = None,
    transition_labels: dict[str, str] | None = None,
    trace_types: tuple[str, ...] = ("full", "occlusion"),
    step_window: tuple[int, int] = (1000, 1350),
) -> str:
    selected_conditions = _resolve_image_mode(image_mode=image_mode, include_novel_image=include_novel_image)
    if not long_dfs_by_transition:
        raise ValueError("long_dfs_by_transition must contain at least one transition result.")

    ordered_transitions = transition_order or TRANSITION_ORDER
    ordered_transitions = [name for name in ordered_transitions if name in long_dfs_by_transition]
    if not ordered_transitions:
        raise ValueError("No requested transitions were found in long_dfs_by_transition.")

    labels = TRANSITION_LABELS.copy()
    if transition_labels is not None:
        labels.update(transition_labels)

    display_windows = [
        _expand_window_to_event_bounds(STIMULI[condition], focus_window=step_window)
        for condition in selected_conditions
        if condition in STIMULI
    ]
    if display_windows:
        plot_window = (
            min(window[0] for window in display_windows),
            max(window[1] for window in display_windows),
        )
    else:
        plot_window = step_window

    stim_windows = {
        condition: _find_stimulus_interval_in_window(STIMULI[condition], step_window=plot_window)
        for condition in selected_conditions
        if condition in STIMULI
    }
    if not stim_windows:
        raise ValueError("STIMULI must contain at least one of the requested conditions.")

    column_specs = [(phase, condition) for phase in ["naive", "expert"] for condition in selected_conditions]
    n_rows = len(ordered_transitions)
    n_cols = len(column_specs)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.4 * n_cols + 1.8, 1.15 * n_rows + 1.9),
        squeeze=False,
        sharex=True,
        sharey=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.89, bottom=0.05, wspace=0.12, hspace=0.18)

    legend_handles = [
        Line2D([0], [0], color=TRACE_COLORS[trace_type], lw=1.6, label=TRACE_LABELS.get(trace_type, trace_type))
        for trace_type in trace_types
        if trace_type in TRACE_COLORS
    ]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.985),
            frameon=False,
            ncol=len(legend_handles),
            handlelength=2.0,
            columnspacing=1.2,
        )

    for col_idx, (phase, condition) in enumerate(column_specs):
        axes[0, col_idx].set_title(IMAGE_LABELS.get(condition, condition.title()), fontsize=12, pad=12)

    for phase_idx, phase in enumerate(["naive", "expert"]):
        start_col = phase_idx * len(selected_conditions)
        end_col = start_col + len(selected_conditions) - 1
        x_center = 0.5 * (axes[0, start_col].get_position().x0 + axes[0, end_col].get_position().x1)
        fig.text(x_center, 0.945, phase.title(), ha="center", va="center", fontsize=20)

    for row_idx, transition_name in enumerate(ordered_transitions):
        long_df = long_dfs_by_transition[transition_name]
        row_bounds: list[tuple[float, float]] = []

        for col_idx, (phase, condition) in enumerate(column_specs):
            ax = axes[row_idx, col_idx]
            stim_interval = stim_windows.get(condition)
            if condition not in STIMULI:
                ax.set_visible(False)
                continue

            if stim_interval is not None:
                ax.axvspan(stim_interval[0], stim_interval[1], color="0.9", zorder=0)
            ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)

            for trace_type in trace_types:
                cell = _extract_trace_window(
                    long_df,
                    condition=condition,
                    phase=phase,
                    image_type=trace_type,
                    step_window=plot_window,
                )
                if cell.empty:
                    continue
                ax.plot(
                    cell["step"].to_numpy(dtype=float),
                    cell["y"].to_numpy(dtype=float),
                    color=TRACE_COLORS.get(trace_type, "black"),
                    lw=1.4,
                )
                row_bounds.append(
                    (
                        float(cell["y"].min()),
                        float(cell["y"].max()),
                    )
                )

            ax.set_xlim(*plot_window)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        label_ax = axes[row_idx, 0]
        label_ax.text(
            -0.12,
            0.5,
            labels.get(transition_name, transition_name),
            transform=label_ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
        if row_bounds:
            row_min = min(bound[0] for bound in row_bounds)
            row_max = max(bound[1] for bound in row_bounds)
            span = row_max - row_min
            if span <= 0:
                span = max(abs(row_min), abs(row_max), 0.1)
            pad = 0.12 * span
            for ax in axes[row_idx, :]:
                if ax.get_visible():
                    ax.set_ylim(row_min - pad, row_max + pad)

    os.makedirs(save_path, exist_ok=True)
    out_path = os.path.join(save_path, f"{name}_{'_'.join(selected_conditions)}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_experiment_results(DF:DataFrame, STIMULI:dict[str, tuple[torch.Tensor, torch.Tensor]], 
                                 save_path:str = PLOTSDIR, name:str = None,
                                 include_novel_no_context: bool = False)->DataFrame:
    long_df = wide_to_long(DF)
    # DF.to_csv(os.path.join(save_path, f"experiment_results_wide_{name}.csv"), index=False)   
    # long_df.to_csv(os.path.join(save_path, f"experiment_results_long_{name}.csv"), index=False)
    panel_a_name = f"{name}panel_A" if name is not None else "panel_A"
    visualize_naive_expert_results(
        long_df,
        STIMULI=STIMULI,
        save_path=save_path,
        name=name,
        include_novel_no_context=include_novel_no_context,
    )
    visualize_naive_expert_results(
        long_df,
        STIMULI=STIMULI,
        save_path=save_path,
        name=panel_a_name,
        full_plots=False,
        include_novel_no_context=include_novel_no_context,
    )
    # visualize_novel_condition_quickplot(long_df, save_path=save_path, name=name)
    return long_df


def visualize_naive_expert_results(long_df:DataFrame, STIMULI:dict[str, tuple[torch.Tensor, torch.Tensor]], 
                                   save_path:str = PLOTSDIR, name:str = None,
                                   full_plots: bool = True,
                                   include_novel_no_context: bool = False) -> None:
    pre_post_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    phases = [p for p in ["naive", "expert"] if p in pre_post_df["experiment_phase"].unique()]
    image_types = sorted(pre_post_df["image_type"].dropna().unique().tolist()) if "image_type" in pre_post_df.columns else []
    conditions = [c for c in ["familiar", "novel"] if c in pre_post_df["condition"].dropna().unique()] if "condition" in pre_post_df.columns else []
    hue_order = _plot_condition_order(image_types)
    y_df = _add_plot_condition_labels(
        pre_post_df[["step", "y", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    )
    pv_df = _add_plot_condition_labels(
        pre_post_df[["step", "pv_value", "pv_index", "condition", "experiment_phase", "image_type"]].drop_duplicates()
    )
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

    if full_plots:
        builder = FigureBuilder.from_matrix(
            [["A", "A", "A", "A"],
             ["B", "B", "B", "B"],
             ["C", "C", "D", "D"]],
            figsize=(24, 18),
            height_ratios=[1.0, 1.0, 1.4],
            constrained_layout=False,
            grid_wspace=0.25,
            grid_hspace=0.15,
            subfigure_wspace=0.15,
            subfigure_hspace=0.2,
        )
        builder.update_panel(
            "A",
            subgrid=(1, max(1, len(conditions) * len(phases))),
            title=None,
            label="A",
            wspace=0.18,
        )
        builder.update_panel("B", subgrid=(1, len(conditions) * len(phases)), title="PV Activity", label="B")
        builder.update_panel("C", subgrid=(2, 1), title="Y and PV activity over training", label="C")
        builder.update_panel("D", subgrid=(4, 1), title="Weight evolution over training", label="D")
    else:
        builder = FigureBuilder.from_matrix(
            [["A", "A", "A", "A"]],
            figsize=(24, 5.5),
            height_ratios=[1.0],
            constrained_layout=False,
            grid_wspace=0.25,
            grid_hspace=0.15,
            subfigure_wspace=0.15,
            subfigure_hspace=0.2,
        )
        builder.update_panel("A", subgrid=(1, len(conditions) * len(phases)), title=None, label="A")

    x_colors = {0: "green", 1: "gold"}
    c_colors = {0: "magenta", 1: "navy"}
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

    activity_layout = [(condition, phase) for condition in conditions for phase in phases]
    stim_windows = {
        condition: _find_stimulus_interval_in_window(STIMULI[condition], step_window=(1000, 1350))
        for condition in conditions
        if condition in STIMULI
    }

    def plot_y(ax_grid, _):
        flat_axes = np.asarray(ax_grid).reshape(-1)
        if flat_axes.size == 0:
            return
        for ax in flat_axes[1:]:
            ax.sharey(flat_axes[0])
        for idx, (condition, phase) in enumerate(activity_layout):
            ax = flat_axes[idx]
            cell = y_df[(y_df["experiment_phase"] == phase) & (y_df["condition"] == condition)]
            cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
            if cell.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=cell,
                x="step",
                y="y",
                hue="plot_condition",
                hue_order=hue_order,
                style="plot_condition",
                palette=PLOT_COLORS,
                errorbar=None,
                ax=ax,
                legend=(idx == 0),
            )
            ax.set_title(f"{condition.title()} | {phase.title()}")
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Neural Activity")
            if name and 'un_un' in name:
                ax.set_ylim(0, 0.3)
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(None)
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    def plot_pv(ax_grid, _):
        flat_axes = np.asarray(ax_grid).reshape(-1)
        if flat_axes.size == 0:
            return
        for ax in flat_axes[1:]:
            ax.sharey(flat_axes[0])
        for idx, (condition, phase) in enumerate(activity_layout):
            ax = flat_axes[idx]
            cell = pv_df[(pv_df["experiment_phase"] == phase) & (pv_df["condition"] == condition)]
            cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
            if cell.empty:
                ax.set_visible(False)
                continue
            sns.lineplot(
                data=cell,
                x="step",
                y="pv_value",
                hue="plot_condition",
                hue_order=hue_order,
                palette=PLOT_COLORS,
                style="pv_index",
                errorbar=None,
                ax=ax,
                legend=(idx == 0),
            )
            ax.set_title(f"{condition.title()} | {phase.title()}")
            ax.set_xlabel("Time steps")
            ax.set_ylabel("Neural Activity")
            legend = ax.get_legend()
            if legend is not None:
                legend.set_title(None)
            if idx > 0:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

    def plot_panel_a(ax_grid, _):
        flat_axes = np.asarray(ax_grid).reshape(-1)
        if flat_axes.size == 0:
            return

        ref_ax = flat_axes[0]
        for ax in flat_axes[1:]:
            ax.sharex(ref_ax)
            ax.sharey(ref_ax)

        legend_handles = [
            Line2D([0], [0], color="black", lw=2.0, label="Nonoccluded (NO)"),
            Line2D([0], [0], color="red", lw=2.0, label="Occluded (O)"),
        ]
        if include_novel_no_context and "novel" in conditions and "novel_no_context" in image_types:
            legend_handles.append(Line2D([0], [0], color="blue", lw=2.0, label="No feedback"))

        global_y_bounds: list[tuple[float, float]] = []
        for idx, (condition, phase) in enumerate(activity_layout):
            ax = flat_axes[idx]
            allowed_image_types = ["full", "occlusion"]
            if include_novel_no_context and condition == "novel":
                allowed_image_types.append("novel_no_context")

            cell = y_df[
                (y_df["experiment_phase"] == phase)
                & (y_df["condition"] == condition)
                & (y_df["image_type"].isin(allowed_image_types))
            ].copy()
            cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
            if cell.empty:
                ax.set_visible(False)
                continue

            trace_specs = [("full", "black"), ("occlusion", "red")]
            if include_novel_no_context and condition == "novel":
                trace_specs.append(("novel_no_context", "blue"))

            for image_type, color in trace_specs:
                trace = cell.loc[cell["image_type"].eq(image_type)]
                if trace.empty:
                    continue
                ax.plot(trace["step"], trace["y"], color=color, lw=2.0)
                global_y_bounds.append((float(trace["y"].min()), float(trace["y"].max())))

            stim_interval = stim_windows.get(condition)
            if stim_interval is not None:
                ax.axvspan(
                    stim_interval[0],
                    stim_interval[1],
                    ymin=0.02,
                    ymax=0.055,
                    color="#8c5a2b",
                    clip_on=False,
                    zorder=3,
                )

            ax.set_xlim(1000, 1350)
            ax.set_title(f"{condition.title()} | {phase.title()}", fontsize=16, pad=10)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(1.8)
            ax.spines["bottom"].set_linewidth(1.8)
            ax.tick_params(width=1.4, length=4)
            ax.set_xlabel("Time steps")
            if idx == 0:
                ax.set_ylabel("Neural Activity")
                ax.legend(
                    handles=legend_handles,
                    loc="upper left",
                    frameon=False,
                    handlelength=2.0,
                    borderaxespad=0.2,
                )
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

        if global_y_bounds:
            y_min = min(bound[0] for bound in global_y_bounds)
            y_max = max(bound[1] for bound in global_y_bounds)
            span = y_max - y_min
            if span <= 0:
                span = max(abs(y_min), abs(y_max), 0.1)
            pad = 0.08 * span
            for ax in flat_axes:
                if ax.get_visible():
                    ax.set_ylim(y_min - pad, y_max + pad)

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
        ax_grid[1, 0].set_xlabel("Time steps")

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

    builder.set_plotter("A", plot_panel_a)
    if full_plots:
        builder.set_plotter("B", plot_pv)
        builder.set_plotter("C", plot_training_activity)
        builder.set_plotter("D", plot_weight_evolution)

    os.makedirs(save_path, exist_ok=True)
    fig, _ = builder.render(save_path=os.path.join(save_path, f"experiment_results_{name}.png"), show=False)
    plt.close(fig)


def visualize_novel_condition_quickplot(long_df: DataFrame, save_path: str = PLOTSDIR, name: str = None) -> None:
    pre_post_df = long_df.loc[long_df["experiment_phase"].isin(["naive", "expert"])].copy()
    novel_df = pre_post_df.loc[pre_post_df["condition"].eq("novel")].copy()
    if novel_df.empty:
        return

    phases = [p for p in ["naive", "expert"] if p in novel_df["experiment_phase"].unique()]
    image_types = sorted(novel_df["image_type"].dropna().unique().tolist()) if "image_type" in novel_df.columns else []
    hue_order = _plot_condition_order(image_types)
    y_df = _add_plot_condition_labels(
        novel_df[["step", "y", "experiment_phase", "image_type"]].drop_duplicates()
    )

    fig, axes = plt.subplots(
        1,
        max(1, len(phases)),
        figsize=(6 * max(1, len(phases)), 4.5),
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    fig.suptitle("Novel stimulus responses")

    for idx, phase in enumerate(phases):
        ax = axes[idx]
        cell = y_df[y_df["experiment_phase"] == phase]
        cell = cell.loc[(cell.step > 1000) & (cell.step < 1350)]
        if cell.empty:
            ax.set_visible(False)
            continue

        sns.lineplot(
            data=cell,
            x="step",
            y="y",
            hue="plot_condition",
            hue_order=hue_order,
            style="plot_condition",
            palette=PLOT_COLORS,
            errorbar=None,
            ax=ax,
            legend=(idx == len(phases) - 1),
        )
        ax.set_title("Naive" if phase == "naive" else "Expert")
        ax.set_xlabel("Time steps")
        if name and "un_un" in name:
            ax.set_ylim(0, 0.3)
        if idx > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("Neural Activity")
    legend = axes[-1].get_legend()
    if legend is not None:
        legend.set_title(None)

    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, f"experiment_results_{name}_novel_only.png"), bbox_inches="tight")
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
