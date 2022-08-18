#!/usr/bin/env python
# -*- mode: python -*-

from pathlib import Path

from cnexp import redo


def main():
    p = Path(
        "../experiments/cifar/dl/model/sgd/lrcos:n_epochs=3/infonce/train"
    )

    redo.redo_ifchange(
        [p.parent / f for f in ["model.pt", "criterion.pt", "default.run"]]
    )

    redo.redo_ifchange_slurm(
        p / "default.run",
        partition="gpu-2080ti-preemptable",
        time_str="00:20:00",
    )

    print((p / "default.run").read_text())


if __name__ == "__main__":
    main()
