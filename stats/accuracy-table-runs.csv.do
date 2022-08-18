#!/usr/bin/env python
# -*- mode: python -*-
import itertools
from pathlib import Path

import numpy as np
from cnexp import redo


def main():
    rng = np.random.default_rng(511622144)
    seeds = [None, *rng.integers(10_000, size=2)]

    prefix = Path("../experiments/cifar")

    dataloaders = [
        "dl" if s is None else f"dl:random_state={s}" for s in seeds
    ]

    models = ["model", "model:out_dim=3"]
    opt = "sgd"
    lrsched = "lrcos:n_epochs=3"
    # p = Path("model/sgd/lrcos:n_epochs=3/infonce/train")

    loss = "infonce:metric=cosine"

    # redo.redo_ifchange(
    #     [p.parent / f for f in ["model.pt", "criterion.pt", "default.run"]]
    # )

    runs = []
    for dl, model in itertools.product(dataloaders, models):
        p = prefix / dl / model / opt / lrsched / loss / "train/default.run"
        runs.append(p)

    redo.redo_ifchange_slurm(
        runs,
        partition="gpu-2080ti-preemptable",
        time_str="00:20:00",
    )

    print((p / "default.run").read_text())


if __name__ == "__main__":
    main()
