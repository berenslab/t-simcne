from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


class GetFirst(object):
    def __init__(self, ar):
        super().__init__()
        self.ar = ar

    def __getitem__(self, i):
        return self.ar[i][0]


def main():
    # get the underlying dataset without augmentations.
    data_sd = torch.load("experiments/cifar/dataset.pt")
    dataset = data_sd["full_plain"].dataset
    arr = GetFirst(dataset)

    prefix = Path.home() / "tmp"
    Y = np.load(prefix / "cifar.3118.npy")
    labels = np.load(prefix / "labels.npy")

    plt.style.use("media/project.mplstyle")
    fig, ax = plt.subplots()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, marker="o")

    # create the annotations box
    im = OffsetImage(arr[0], zoom=1)
    xybox = (50.0, 50.0)
    ab = AnnotationBbox(
        im,
        (0, 0),
        xybox=xybox,
        xycoords="data",
        boxcoords="offset points",
        pad=0.3,
        arrowprops=dict(arrowstyle="->"),
    )
    # add it to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)

    def onclick(event):
        # if the mouse is over the scatter points
        if scatter.contains(event)[0]:
            # find out the index within the array from the event
            ind = scatter.contains(event)[1]["ind"][0]
            # get the figure size
            w, h = fig.get_size_inches() * fig.dpi
            ws = (event.x > w / 2.0) * -1 + (event.x <= w / 2.0)
            hs = (event.y > h / 2.0) * -1 + (event.y <= h / 2.0)
            # if event occurs in the top or right quadrant of the figure,
            # change the annotation box position relative to mouse.
            ab.xybox = (xybox[0] * ws, xybox[1] * hs)
            # make annotation box visible
            ab.set_visible(True)
            # place it at the position of the hovered scatter point
            ab.xy = (Y[ind, 0], Y[ind, 1])
            # set the image corresponding to that point
            im.set_data(arr[ind])
        else:
            # if the mouse is not over a scatter point
            ab.set_visible(False)
        fig.canvas.draw_idle()

    # add callback for mouse moves
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.show()


if __name__ == "__main__":
    main()
