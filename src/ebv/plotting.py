"""One consistent look for every figure: palette, fonts, and grid-cell maps with state outlines."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from shapely.geometry import box

from . import config
from .region import GridSpec, load_admin_boundaries

INK, GREEN, MINT, AMBER = "#1B2E25", "#2D6A4F", "#95D5B2", "#E07A1F"
GREY, BLUE, RED, PAPER = "#9AA59F", "#3B6E8F", "#B7472A", "#FFFFFF"
GREENS = LinearSegmentedColormap.from_list("greens", ["#F3F8F1", "#B7E4C7", "#52B788", "#2D6A4F", "#081C15"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#FFF4E6", "#F8C58A", "#E07A1F", "#8A3B0A"])
TERRAIN = LinearSegmentedColormap.from_list("terrain", ["#D8EDD0", "#95C47F", "#4E8B5F", "#8C6D46", "#EFE8DC"])
NEIGHBOURS = ("Andhra Pradesh", "Telangana", "Maharashtra")
LATO_DIR = Path("/usr/share/fonts/truetype/lato")


def _font_family() -> list[str]:
    """Lato when the system has it (registered with matplotlib), otherwise DejaVu Sans."""
    fonts = sorted(LATO_DIR.glob("Lato-*.ttf")) if LATO_DIR.exists() else []
    for path in fonts:
        font_manager.fontManager.addfont(str(path))
    return ["Lato", "DejaVu Sans"] if fonts else ["DejaVu Sans"]


def use_style() -> None:
    plt.rcParams.update({
        "font.family": _font_family(), "font.size": 14, "axes.titlesize": 17,
        "axes.titleweight": "bold", "axes.titlepad": 10, "axes.labelsize": 15, "axes.edgecolor": INK,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
        "savefig.bbox": "tight", "savefig.facecolor": PAPER, "figure.facecolor": PAPER,
    })


class MapCanvas:
    """Draws per-cell values as a raster, with the five study states outlined."""

    def __init__(self, cells: pd.DataFrame):
        self.spec = GridSpec.from_cells(cells)
        self.rows, self.cols = self.spec.row_col(cells)
        admin = load_admin_boundaries()
        frame = box(*self.spec.bounds).buffer(0.5)
        self.states = {n: admin[n].intersection(frame).simplify(0.01) for n in config.STATES}
        self.neighbours = [admin[n].intersection(frame).simplify(0.01) for n in NEIGHBOURS]

    def raster(self, values) -> np.ndarray:
        image = np.full((self.spec.n_rows, self.spec.n_cols), np.nan)
        image[self.rows, self.cols] = np.asarray(values, float)
        return image

    @staticmethod
    def outline(ax, geometry, **style):
        parts = getattr(geometry, "geoms", [geometry])
        for part in parts:
            if part.geom_type == "Polygon":
                x, y = part.exterior.xy
                ax.plot(x, y, **style)

    def draw(self, ax, values, title=None, cmap=GREENS, vmin=None, vmax=None, colorbar=None,
             background=None, outline_color=INK):
        """Map `values` (one per cell, NaN = no colour). `background` cells are shaded grey first."""
        w, s, e, n = self.spec.bounds
        if background is not None:
            ax.imshow(np.where(np.isnan(self.raster(background)), np.nan, 1.0), extent=(w, e, s, n),
                      cmap=LinearSegmentedColormap.from_list("bg", ["#E3E7E4"] * 2), interpolation="nearest")
        image = ax.imshow(self.raster(values), extent=(w, e, s, n), cmap=cmap, vmin=vmin, vmax=vmax,
                          interpolation="nearest")
        for geometry in self.neighbours:
            self.outline(ax, geometry, color=GREY, lw=0.5)
        for geometry in self.states.values():
            self.outline(ax, geometry, color=outline_color, lw=0.7)
        ax.set_xlim(w - 0.1, e + 0.1)
        ax.set_ylim(s - 0.1, n + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        if title:
            ax.set_title(title)
        if colorbar:
            bar = plt.colorbar(image, ax=ax, shrink=0.62, pad=0.01, aspect=22)
            bar.set_label(colorbar, fontsize=13)
            bar.outline.set_visible(False)
            bar.ax.tick_params(labelsize=12)
        return image
