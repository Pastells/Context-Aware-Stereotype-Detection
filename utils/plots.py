import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import config


def init_plot_params(font_scale=1.8, **kwargs):
    config.FIG_PARAMS.update(**kwargs)
    plt.rcParams.update(config.FIG_PARAMS)
    sns.set_theme(rc=config.FIG_PARAMS)
    sns.set_style("white")
    sns.set_context("paper", font_scale=font_scale)
    sns.set_palette(config.PALETTE)


def savefig(name, suffix=None):
    sns.despine()
    if suffix:
        name = name + "_" + suffix
    plt.tight_layout()
    plt.savefig(
        os.path.join(config.BASE_DIR, "results/figures", name + ".png"),
        bbox_inches="tight",
    )
    plt.clf()


def barplot(
    fig: plt.Figure,
    ax: plt.Axes,
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    y_std: str = "",
    y_minmax="",  # two strings
    ttest=False,
    legend=True,
    order=None,
    **kwargs,
):
    """
    Seaborn barplot with errorbars
    takes fig and ax from plt as input
    either use:
    - a single row with y_std
    or
    - two rows with y_min and y_max
    """
    bar = sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, order=order, **kwargs)

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    def yerr(row):
        if y_minmax:
            return [
                [row[y] - row[y_minmax[0]]],
                [row[y_minmax[1]] - row[y]],
            ]
        elif y_std:
            return row[y_std]
        else:
            return None

    def ymax(row) -> float:
        if y_minmax:
            return row[y_minmax[1]]
        elif y_std:
            return row[y] + row[y_std]
        else:
            return row[y]

    if order:
        xs = order
    else:
        xs = data[x].unique().tolist()

    n_x = len(xs)
    if hue is not None:
        hues = data[hue].unique().tolist()
    else:
        hues = []

    n_hues = len(hues)
    bar_width = 1 / (1 + n_hues)
    space_width = bar_width * (n_hues + 1)

    hatches = sorted([" ", "..", "/", "o", "+"][:n_hues] * n_x)
    for i, thisbar in enumerate(bar.patches):
        thisbar.set_hatch(hatches[i])

    # print(data)
    for i, xi in enumerate(xs):
        for j, yi in enumerate(hues):
            row = data[(data[x] == xi) & (data[hue] == yi)].iloc[0]
            # print(row)
            ax.errorbar(
                x=-0.5 + (j + 1) * bar_width + i * space_width,
                y=row[y],
                yerr=yerr(row),
                fmt="none",
                ecolor="black",
                capsize=4,
                elinewidth=2,
            )
            if ttest and row["pvalue"] < 0.05:
                ax.text(
                    x=-0.54 + (j + 1) * bar_width + i * space_width,
                    y=ymax(row) + 0.02,
                    s="↑" if row["ttest"] > 0 else "↓",
                    fontsize=15,
                )

    if legend:
        fig.legend(
            labels=["_no_legend_"] * n_hues * n_x + hues,
            bbox_to_anchor=(1.15, 1),
            fontsize=config.LEGEND_FONTSIZE,
        )
