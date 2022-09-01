import string
import subprocess


def get_default_metadata():
    meta = dict(
        Author="Jan Niklas Böhm",
    )

    git_proc = subprocess.run(
        ["git", "describe", "--always", "--dirty", "--broken"],
        encoding="utf-8",
        capture_output=True,
    )
    goredo_proc = subprocess.run(
        ["redo", "-version"],
        encoding="utf-8",
        capture_output=True,
    )
    pyredo_proc = subprocess.run(
        ["redo", "--version"],
        encoding="utf-8",
        capture_output=True,
    )

    if git_proc.returncode == 0:
        creator_str = f"cnexp {git_proc.stdout.strip()}"

        if goredo_proc.returncode == 0:
            # goredo 1.27.0 built with go1.19
            redo_v = goredo_proc.stdout.split()[1]
            redo_str = f"goredo v{redo_v}"
        elif pyredo_proc.returncode == 0:
            # 0.42
            redo_v = pyredo_proc.stdout.strip()
            redo_str = f"apenwarr redo v{redo_v}"
        else:
            redo_str = ""

        if redo_str != "":
            creator_str += f", {redo_str}"

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
        for ax, letter in zip(axs.flat, string.ascii_lowercase)
    ]
