import string
import subprocess

import numpy as np


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
