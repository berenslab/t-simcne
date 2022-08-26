import string


def get_lettering_fprops():
    return dict(
        fontsize="x-large",
        fontweight="bold",
    )


def add_lettering(ax, letter, loc="left", **kwargs):
    other_title = ax.get_title("center")
    newlines = len(other_title.split("\n")) - 1
    return ax.set_title(letter + newlines * "\n", loc=loc, **kwargs)


def add_letters(axs, *, loc="left", **kwargs):
    fprops = get_lettering_fprops()

    return [
        add_lettering(ax, letter, loc=loc, fontdict=fprops, **kwargs)
        for ax, letter in zip(axs.flat, string.ascii_lowercase)
    ]
