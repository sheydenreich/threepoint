"""
Functions for creating nice matplotlib plots
"""

import numpy as np
from matplotlib.offsetbox import AnchoredText
import matplotlib.pyplot as plt




def initPlot(fontsize=20, titlesize=14, labelsize=18, legendsize=14, usetex=True):
    """
    Sets the basics of the plot

    Parameters
    ----------
    fontsize : int (optional)
        Fontsize of standard text in plot (default: 16)
    titlesize :  int (optional)
        Fontsize of title (default: 16)
    labelsize : int (optional)
        Fontsize of tick-labels and legend (default: 16)
    usetex : bool (optional)
        If true: Text is renderd as LaTeX (default: True)
    """

    properties =    {
        "text.usetex": usetex,
        "font.family": "sans-serif",
        "axes.labelsize": labelsize,
        "font.size": fontsize,
        "legend.fontsize": legendsize,
        "xtick.labelsize": labelsize,
        "ytick.labelsize": labelsize,
        "axes.titlesize": titlesize,
        "axes.facecolor": 'white'
    }

    plt.rcParams.update(properties)
    


def finalizePlot(ax, title="", outputFn="", showplot=True,  showlegend=True, tightlayout=True, legendcols=1, loc_legend="best", facecolor="white"):
    """
    Finalizes Plots, saves it and shows it

    Parameters
    ----------
    ax : axis object from Matplotlib
        Plot to be shown
    title : string (optional)
        Title of Plot (default: no title)
    outputFn : string (optional)
        Filename to which Plot should be saved (default: not saved)
    showplot : bool (optional)
        If true, plot is displayed after saving (default: True)
    showlegend : bool (optional)
        If true, a legend is shown (default: True)
    tightlayout : bool (optional)
        If true, matplotlibs option "tightlayout" is used (default: True)
    """

    if(title != ""):
        ax.set_title(title)


    if(showlegend):
        plt.legend(loc=loc_legend, ncol=legendcols)

    if(tightlayout):
        plt.tight_layout()

    if(outputFn != ""):
        plt.savefig(outputFn, dpi=300, facecolor=facecolor)

    if(showplot):
        plt.show()





