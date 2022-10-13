import string
import subprocess

import numpy as np

from .scalebar import add_scalebar_frac

__export__ = [add_scalebar_frac]


def get_default_metadata():
    meta = dict(
        Author="Jan Niklas Böhm",
    )

    git_proc = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--broken"],
        encoding="utf-8",
        capture_output=True,
    )

    if git_proc.returncode == 0:
        creator_str = f"cnexp {git_proc.stdout.strip()}"

        meta["Creator"] = creator_str

    return meta


def flip_maybe(
    other,
    *,
    anchor,
    return_anchor=False,
):
    """flip `other` along x- or y-axis, depending on covariance to `anchor`."""

    flipx = np.cov(anchor[:, 0], other[:, 0])[0, 1]
    flipy = np.cov(anchor[:, 1], other[:, 1])[0, 1]

    flip = np.sign([flipx, flipy], dtype=other.dtype)

    if return_anchor:
        return anchor, other * flip
    else:
        return other * flip


def get_lettering_fprops():
    return dict(
        fontsize="x-large",
        fontweight="bold",
    )


def add_lettering(ax, letter, loc="left", fontdict=None, **kwargs):
    fontdict = get_lettering_fprops() if fontdict is None else fontdict

    other_title = ax.get_title("center")
    newlines = len(other_title.split("\n")) - 1
    return ax.set_title(
        letter + newlines * "\n", loc=loc, fontdict=fontdict, **kwargs
    )


def add_letters(axs, *, loc="left", **kwargs):
    fprops = get_lettering_fprops()

    return [
        add_lettering(ax, letter, loc=loc, fontdict=fprops, **kwargs)
        for ax, letter in zip(np.array(list(axs)).flat, string.ascii_lowercase)
    ]
