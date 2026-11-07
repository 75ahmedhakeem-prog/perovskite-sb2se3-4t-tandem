"""Post-style hook for paper-5 figures: one minor tick between majors,
plain top/right spines (no ticks) unless they carry major ticks, small pads."""
import numpy as np
import matplotlib.figure as mfig
from matplotlib.ticker import (AutoMinorLocator, LogLocator, NullLocator,
                               FixedLocator, FixedFormatter, NullFormatter)

_orig = mfig.Figure.savefig


def _minor(axis, scale):
    maj = axis.get_major_locator()
    if isinstance(maj, FixedLocator) and isinstance(axis.get_major_formatter(), FixedFormatter):
        axis.set_minor_locator(NullLocator()); return
    if scale == "log":
        locs = [l for l in axis.get_majorticklocs() if l > 0]
        step = np.median(np.diff(np.log10(locs))) if len(locs) > 1 else 1
        subs = (1.0,) if step >= 1.9 else (10 ** 0.5,)
        axis.set_minor_locator(LogLocator(base=10, subs=subs, numticks=100))
        axis.set_minor_formatter(NullFormatter())
    elif scale == "linear":
        axis.set_minor_locator(AutoMinorLocator(2))


def style(fig):
    for ax in fig.axes:
        if not ax.axison:
            continue
        _minor(ax.xaxis, ax.get_xscale()); _minor(ax.yaxis, ax.get_yscale())
        ypos = ax.yaxis.get_ticks_position(); xpos = ax.xaxis.get_ticks_position()
        if ypos == "right":
            ax.tick_params(axis="y", which="both", left=False, right=True)
        else:
            ax.tick_params(axis="y", which="both", left=True, right=False)
        if xpos == "top":
            ax.tick_params(axis="x", which="both", bottom=False, top=True)
        else:
            ax.tick_params(axis="x", which="both", bottom=True, top=False)
        ax.tick_params(axis="y", which="major", pad=3); ax.tick_params(axis="x", which="major", pad=getattr(ax, "_xpad", 5))
        ax.xaxis.labelpad = 2; ax.yaxis.labelpad = 2


def savefig(self, *a, **k):
    style(self)
    return _orig(self, *a, **k)


mfig.Figure.savefig = savefig
