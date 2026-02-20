# General code for producing scientific-style figures with matplotlib, using a simple declarative layout system.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


PlotTarget = Axes | np.ndarray
PlotFn = Callable[[PlotTarget, "PanelSpec"], None]


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    subgrid: tuple[int, int] | None = None
    title: str | None = None
    label: str | None = None
    projection: str | None = None
    wspace: float | None = None
    hspace: float | None = None


def _validate_matrix(matrix: list[list[str]]) -> tuple[int, int]:
    if not matrix or not matrix[0]:
        raise ValueError("matrix must be a non-empty 2D list of panel ids")
    nrows = len(matrix)
    ncols = len(matrix[0])
    if any(len(r) != ncols for r in matrix):
        raise ValueError("all matrix rows must have the same length")
    return nrows, ncols


def panel_specs_from_matrix(matrix: list[list[str]]) -> dict[str, PanelSpec]:
    nrows, ncols = _validate_matrix(matrix)
    positions: dict[str, list[tuple[int, int]]] = {}
    for r in range(nrows):
        for c in range(ncols):
            panel_id = matrix[r][c]
            positions.setdefault(panel_id, []).append((r, c))

    specs: dict[str, PanelSpec] = {}
    for panel_id, coords in positions.items():
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        expected = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        if set(coords) != expected:
            raise ValueError(f"panel '{panel_id}' must occupy one rectangular region in matrix")
        specs[panel_id] = PanelSpec(
            panel_id=panel_id,
            row=r0,
            col=c0,
            rowspan=r1 - r0 + 1,
            colspan=c1 - c0 + 1,
        )
    return specs


class FigureBuilder:
    """
    Build multi-panel scientific figures with a simple declarative layout.

    Example:
        matrix = [
            ["A", "A", "B"],
            ["C", "D", "D"],
        ]

        builder = FigureBuilder.from_matrix(matrix, figsize=(10, 6))
        builder.set_plotter("A", lambda ax, _: ax.plot([1, 2, 3], [1, 4, 9]))
        builder.set_plotter("B", lambda ax, _: ax.hist(np.random.randn(1_000), bins=30))
        fig, axes = builder.render(show=True)
    """

    def __init__(
        self,
        nrows: int,
        ncols: int,
        *,
        figsize: tuple[float, float] = (12, 8),
        width_ratios: list[float] | None = None,
        height_ratios: list[float] | None = None,
        constrained_layout: bool = True,
        grid_wspace: float | None = None,
        grid_hspace: float | None = None,
        subfigure_wspace: float | None = None,
        subfigure_hspace: float | None = None,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.width_ratios = width_ratios
        self.height_ratios = height_ratios
        self.constrained_layout = constrained_layout
        self.grid_wspace = grid_wspace
        self.grid_hspace = grid_hspace
        self.subfigure_wspace = subfigure_wspace
        self.subfigure_hspace = subfigure_hspace
        self._panels: dict[str, PanelSpec] = {}
        self._plotters: dict[str, PlotFn] = {}

    @classmethod
    def from_matrix(
        cls,
        matrix: list[list[str]],
        *,
        figsize: tuple[float, float] = (12, 8),
        width_ratios: list[float] | None = None,
        height_ratios: list[float] | None = None,
        constrained_layout: bool = True,
        grid_wspace: float | None = None,
        grid_hspace: float | None = None,
        subfigure_wspace: float | None = None,
        subfigure_hspace: float | None = None,
    ) -> "FigureBuilder":
        nrows, ncols = _validate_matrix(matrix)
        builder = cls(
            nrows,
            ncols,
            figsize=figsize,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            constrained_layout=constrained_layout,
            grid_wspace=grid_wspace,
            grid_hspace=grid_hspace,
            subfigure_wspace=subfigure_wspace,
            subfigure_hspace=subfigure_hspace,
        )
        for spec in panel_specs_from_matrix(matrix).values():
            builder.add_panel(spec)
        return builder

    def add_panel(self, spec: PanelSpec) -> None:
        self._panels[spec.panel_id] = spec

    def update_panel(self, panel_id: str, **kwargs: Any) -> None:
        if panel_id not in self._panels:
            raise KeyError(f"unknown panel '{panel_id}'")
        current = self._panels[panel_id]
        self._panels[panel_id] = PanelSpec(
            panel_id=current.panel_id,
            row=kwargs.get("row", current.row),
            col=kwargs.get("col", current.col),
            rowspan=kwargs.get("rowspan", current.rowspan),
            colspan=kwargs.get("colspan", current.colspan),
            subgrid=kwargs.get("subgrid", current.subgrid),
            title=kwargs.get("title", current.title),
            label=kwargs.get("label", current.label),
            projection=kwargs.get("projection", current.projection),
            wspace=kwargs.get("wspace", current.wspace),
            hspace=kwargs.get("hspace", current.hspace),
        )

    def set_plotter(self, panel_id: str, plot_fn: PlotFn) -> None:
        if panel_id not in self._panels:
            raise KeyError(f"unknown panel '{panel_id}'")
        self._plotters[panel_id] = plot_fn

    def render(
        self,
        *,
        save_path: str | None = None,
        dpi: int = 300,
        show: bool = False,
    ) -> tuple[Figure, dict[str, PlotTarget]]:
        fig = plt.figure(figsize=self.figsize, constrained_layout=self.constrained_layout)
        gs = fig.add_gridspec(
            nrows=self.nrows,
            ncols=self.ncols,
            width_ratios=self.width_ratios,
            height_ratios=self.height_ratios,
            wspace=self.grid_wspace,
            hspace=self.grid_hspace,
        )

        rendered_axes: dict[str, PlotTarget] = {}
        for panel_id, spec in self._panels.items():
            sub = gs[spec.row : spec.row + spec.rowspan, spec.col : spec.col + spec.colspan]
            target = self._build_target(fig, sub, spec)
            rendered_axes[panel_id] = target

            if spec.title:
                if isinstance(target, np.ndarray):
                    target.flat[0].set_title(spec.title)
                else:
                    target.set_title(spec.title)

            if spec.label:
                anchor = target.flat[0] if isinstance(target, np.ndarray) else target
                anchor.text(
                    -0.12,
                    1.08,
                    spec.label,
                    transform=anchor.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    va="top",
                    ha="left",
                )

            plot_fn = self._plotters.get(panel_id)
            if plot_fn is not None:
                plot_fn(target, spec)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        return fig, rendered_axes

    def _build_target(self, fig: Figure, subplotspec: Any, spec: PanelSpec) -> PlotTarget:
        if spec.subgrid is None:
            return fig.add_subplot(subplotspec, projection=spec.projection)

        nrows, ncols = spec.subgrid
        sgs = subplotspec.subgridspec(
            nrows=nrows,
            ncols=ncols,
            wspace=spec.wspace if spec.wspace is not None else self.subfigure_wspace,
            hspace=spec.hspace if spec.hspace is not None else self.subfigure_hspace,
        )
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = fig.add_subplot(sgs[r, c], projection=spec.projection)
        return axes
