# author: Matúš Halák (@matushalak)
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

def get_hva_tuning(*args)->torch.Tensor:
    '''
    len(args) = n_hva, len(args[0]) = n_pyramidal
    
    Each argument contains the weights each HVA neuron assigns 
        to each of the pyramidal neurons. 

    Returns a tensor of shape (n_hva, n_pyramidal) 
        where each row corresponds to the tuning pattern for one HVA neuron.
    '''
    tuning = torch.zeros(len(args), len(args[0]))
    for i, pattern in enumerate(args):
        tuning[i, :] = torch.tensor(pattern)
    return tuning

# TODO:
def get_pv_tuning(*args)->torch.Tensor:
    raise NotImplementedError

def get_pyc_tuning(*args)->torch.Tensor:
    raise NotImplementedError


def plot_out(out: dict, I: torch.Tensor | None = None, 
             figsize: tuple[int, int] = (12, 8),
             title: str = "Model Activity Over Time", 
             show:bool = True, 
             save:bool = False, savedir:str = 'plots') -> tuple[plt.Figure, np.ndarray]:
    """
    Plot model activity over time with per-neuron color palettes:
    - input: rainbow palette (stacked channels)
    - PV activations: red-brown-orange palette
    - pyramidal activations: rainbow palette
    - HVA activations: blue-purple palette
    """
    sns.set_theme(style="whitegrid", context="talk")

    time = out["Time"].detach().cpu().numpy()
    pyramidal = out["Pyramidal"].detach().cpu().numpy()
    pv = out["PV"].detach().cpu().numpy()
    hva = out["HVA"].detach().cpu().numpy()
    inp = I.detach().cpu().numpy() if I is not None else None

    has_input = inp is not None
    nrows = 4 if has_input else 3
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figsize,
        sharex=True,
    )
    if nrows == 1:  # keep a consistent indexable structure
        axes = [axes]

    row = 0
    if has_input:
        n_input = inp.shape[1]
        input_colors = sns.color_palette("hsv", n_input)
        channel_std = inp.std(axis=0, ddof=0)
        base_scale = float(np.median(channel_std[channel_std > 0])) if np.any(channel_std > 0) else 1.0
        spacing = 5 * base_scale
        offsets = np.arange(n_input) * spacing
        for i in range(n_input):
            axes[row].plot(
                time,
                inp[:, i] + offsets[i],
                color=input_colors[i],
                alpha=0.95,
                linewidth=1.4,
            )
        axes[row].set_yticks(offsets)
        axes[row].set_yticklabels([f"In {i}" for i in range(n_input)])
        axes[row].set_ylabel("Input")
        axes[row].set_title("Input Channels")
        row += 1

    pv_colors = sns.blend_palette(["#8b0000", "#8b4513", "#ff8c00"], n_colors=pv.shape[0])
    for i in range(pv.shape[0]):
        axes[row].plot(time, pv[i], color=pv_colors[i], linestyle = '--' if i == 1 else '-', label=f"PV {i}",
                       alpha=0.9, linewidth=1.8)
    axes[row].set_ylabel("PV")
    # make y axis log-scale
    # axes[row].set_ylim(bottom=-0.1, top=5)
    axes[row].set_title("PV Activations")
    axes[row].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, ncol=1, frameon=False)
    row += 1

    pyramidal_colors = sns.color_palette("Set1", pyramidal.shape[0])
    for i in range(pyramidal.shape[0]):
        axes[row].plot(
            time,
            pyramidal[i],
            color=pyramidal_colors[i],
            alpha=0.9,
            linewidth=1.9,
            linestyle = '--' if i == 2 else '-',
            label=f"Pyr {i}",
        )
    axes[row].set_ylabel("Pyramidal")
    # axes[row].set_ylim(bottom=-0.1, top=2)
    axes[row].set_title("Pyramidal Activations")
    axes[row].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, ncol=1, frameon=False)
    row += 1

    hva_colors = sns.blend_palette(["#1f4fff", "#7b2cbf"], n_colors=hva.shape[0])
    for i in range(hva.shape[0]):
        axes[row].plot(time, hva[i], color=hva_colors[i], alpha=0.9, linewidth=1.8, label=f"HVA {i}")
    axes[row].set_ylabel("HVA")
    # axes[row].set_ylim(bottom = -0.1, top=1)
    axes[row].set_title("HVA Activations")
    axes[row].set_xlabel("Time")
    axes[row].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, ncol=1, frameon=False)

    axes[row].set_xlim(float(time[0]), float(time[-1]))
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 1])
    if save:
        os.makedirs(savedir, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.png"
        fig.savefig(os.path.join(savedir, filename), dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    return fig, axes

