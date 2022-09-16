import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def main():

    npz = np.load("c100_emb.npz")
    Y = npz["data"]
    Y -= Y.mean(0)
    Y[:, 0] += 30
    # Y[:, 1] += -10
    # print(Y.mean(0))

    labels = npz["labels"]
    names = npz["names"]

    fig, ax = plt.subplots(figsize=(3, 3), dpi=450, constrained_layout=True)

    cm = plt.get_cmap("tab10", lut=names.shape[0])
    ax.scatter(
        Y[:, 0],
        Y[:, 1],
        c=cm(labels),
        alpha=0.5,
        ec=None,
        s=1,
        marker=".",
        rasterized=True,
    )
    ax.set_axis_off()
    ax.axis("equal")
    # ax.scatter([0], [0], color="black", marker="x", s=5)

    slots = np.linspace(-np.pi, np.pi, names.shape[0], endpoint=False)

    locs = np.empty((100, 2))
    for i in range(names.shape[0]):
        mask = labels == i
        Ym = Y[mask]
        kde = gaussian_kde(Ym.T)
        locs[i, :] = Ym[kde(Ym.T).argmax()]

    norm = (locs**2).sum(1) ** 0.5
    circ = locs / norm[:, None]
    zs = circ[:, 0] + 1j * circ[:, 1]
    ixs = np.argsort(np.angle(zs))

    for i, ix in enumerate(ixs):
        slot = slots[i]
        loc = locs[ix]
        title = names[ix].replace("_", " ")
        z = 1.1 * Y.max() * (np.cos(slot) + 1j * np.sin(slot))
        ax.plot(
            [loc[0], np.real(z)],
            [loc[1], np.imag(z)],
            c="xkcd:light gray",
            # c=cm(ix),
            zorder=0.9,
            lw=0.2,
        )
        ang = slot * 180 / np.pi
        ha = "left" if -90 < ang < 90 else "right"
        ang -= 0 if -90 < ang < 90 else 180
        ax.text(
            np.real(z),
            np.imag(z),
            title,
            color=cm(ix),
            rotation=ang,
            rotation_mode="anchor",
            va="center",
            ha=ha,
            fontsize="xx-small",
        )

    fig.savefig("cifar100.labels.pdf")
    fig.savefig("cifar100.labels.png")


if __name__ == "__main__":
    main()
