import numpy as np
from matplotlib.offsetbox import (
    AnchoredOffsetbox,
    AuxTransformBox,
    HPacker,
    TextArea,
    VPacker,
)
from matplotlib.patches import FancyArrowPatch, Rectangle


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.1,
        borderpad=0.1,
        sep=2,
        prop=None,
        barcolor="black",
        barwidth=0.4,
        **kwargs,
    ):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        bars = AuxTransformBox(transform)
        if sizex:
            arr = FancyArrowPatch(
                (0, 0),
                (sizex, 0),
                shrinkA=0,
                shrinkB=0,
                ec=barcolor,
                lw=barwidth,
                fc="none",
                arrowstyle="|-|",
            )
            arr.set_capstyle("butt")

            bars.add_artist(arr)

        if sizey:
            bars.add_artist(
                Rectangle(
                    (0, 0), 0, sizey, ec=barcolor, lw=barwidth, fc="none"
                )
            )

        if sizex and labelx:
            self.xlabel = TextArea(labelx)
            bars = VPacker(
                children=[bars, self.xlabel], align="center", pad=0, sep=sep
            )
        if sizey and labely:
            self.ylabel = TextArea(labely)
            bars = HPacker(
                children=[self.ylabel, bars], align="center", pad=0, sep=sep
            )

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs,
        )


def add_scalebar(
    ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs
):
    """Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """

    def f(axis):
        locs = axis.get_majorticklocs()
        return len(locs) > 1 and (locs[1] - locs[0])

    if matchx:
        kwargs["sizex"] = f(ax.xaxis)
        kwargs["labelx"] = str(kwargs["sizex"])
    if matchy:
        kwargs["sizey"] = f(ax.yaxis)
        kwargs["labely"] = str(kwargs["sizey"])

    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)
    if hidex and hidey:
        ax.set_axis_off()

    return sb


def add_scalebar_frac(
    ax, frac_len=0.333, eps=0.5, only_x=True, fix_aspect=True
):
    assert only_x, "Only usage in for this specific use case for now."

    if fix_aspect:
        ax.set_aspect("equal", adjustable="datalim")

    xmin, xmax, *_ = ax.axis()
    val = round_pow10((xmax - xmin) * frac_len)
    lbl = f"{val:g}"
    return add_scalebar(
        ax,
        matchx=False,
        matchy=False,
        sizey=0,
        sizex=val,
        labelx=lbl,
        hidex=True,
        hidey=True,
    )


def round_to_1(x, /):
    return round(x, -int(np.floor(np.log10(abs(x)))))


def round_pow10(x, /):
    """Round to the nearest power of ten.

    See https://ubuntuforums.org/showthread.php?t=816175.

    """
    return 10 ** int(np.floor(np.log10(2 * x)))


def round_nice(x, eps=0.5, nice_numbers=[1, 5]):
    n_digits = -int(np.floor(np.log10(abs(x))))
    x_ = x * 10**n_digits
    x_nice = x
    for n in nice_numbers:
        if x_ * (1 - eps) <= n <= x_ * (1 + eps):
            x_nice = n / 10**n_digits

    return round(x_nice, n_digits)
