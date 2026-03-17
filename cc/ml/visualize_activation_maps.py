import torch


SIGNAL_NAMES = ("Y", "y_FF", "y_FB")


def _select_time_indices(total_t: int, max_time_steps: int) -> torch.Tensor:
    if total_t <= max_time_steps:
        return torch.arange(total_t)
    return torch.linspace(0, total_t - 1, steps=max_time_steps).round().long()


def create_activation_input_figure(
    inputs: torch.Tensor,
    sample_idx: int,
    max_time_steps: int,
):
    import matplotlib.pyplot as plt

    time_idx = _select_time_indices(inputs.shape[1], max_time_steps)
    frames = inputs[sample_idx, time_idx].detach().cpu()
    fig, axes = plt.subplots(1, len(time_idx), figsize=(1.8 * len(time_idx), 2.2), squeeze=False)
    vmin = float(frames.min().item())
    vmax = float(frames.max().item())

    for col, t_idx in enumerate(time_idx.tolist()):
        ax = axes[0, col]
        ax.imshow(frames[col, 0], cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(f"t={t_idx}")
        ax.axis("off")

    fig.suptitle(f"Masked input sequence | sample {sample_idx}")
    return fig


def create_activation_map_figure(
    activation_maps: dict[str, dict[str, torch.Tensor]],
    layer_name: str,
    sample_idx: int = 0,
    max_time_steps: int = 12,
    mode: str | None = None,
):
    import matplotlib.pyplot as plt

    ref = activation_maps["Y"][layer_name]
    time_idx = _select_time_indices(ref.shape[1], max_time_steps)
    fig, axes = plt.subplots(
        len(SIGNAL_NAMES),
        len(time_idx),
        figsize=(1.9 * len(time_idx), 2.2 * len(SIGNAL_NAMES)),
        squeeze=False,
    )

    for row, signal_name in enumerate(SIGNAL_NAMES):
        maps = activation_maps[signal_name][layer_name][sample_idx, time_idx].detach().cpu()
        vmin = float(maps.min().item())
        vmax = float(maps.max().item())
        if vmin == vmax:
            vmax = vmin + 1e-6

        for col, t_idx in enumerate(time_idx.tolist()):
            ax = axes[row, col]
            im = ax.imshow(maps[col], cmap="magma", interpolation="nearest", vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(f"t={t_idx}")
            if col == 0:
                ax.set_ylabel(signal_name)
            ax.axis("off")

        fig.colorbar(im, ax=axes[row, :].tolist(), fraction=0.02, pad=0.01)

    title_bits = [layer_name, f"sample {sample_idx}"]
    if mode is not None:
        title_bits.insert(0, mode)
    fig.suptitle(" | ".join(title_bits))
    return fig
