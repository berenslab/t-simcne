#!/usr/bin/env python
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cnexp import names as expnames
from cnexp import redo


def main():
    prefix = Path("../../experiments")

    parts = sys.argv[2].split(".")
    if len(parts) == 2:
        metric, dataset = parts
        dim = 128
        n_epochs = 1000
        backbone = "resnet18"

    elif len(parts) == 3:
        metric, dim_str, dataset = parts
        dim = int(dim_str[:-1])
        assert dim_str[-1] == "d"
        n_epochs = 1000
        backbone = "resnet18"

    elif len(parts) == 4:
        metric, dim_str, extra_str, dataset = parts
        dim = int(dim_str[:-1])
        n_epochs = 5000 if extra_str == "long" else 1000
        backbone = "resnet50" if extra_str == "resnet50" else "resnet18"
        assert dim_str[-1] == "d"
        assert extra_str in ["long", "resnet50"]

    else:
        raise ValueError(
            f"{sys.argv[0]}: wrong number of parts passed in "
            f"filename, got {len(parts)} parts in {sys.argv[2]!r}"
        )

    model = "model" if dim == 128 else f"model:out_dim={dim}"
    opt = "sgd"
    lrsched = "lrcos" if n_epochs == 1000 else f"lrcos:{n_epochs=}"

    if metric == "euc":
        infonce = "infonce"
        m_str = "euclidean"
    elif metric == "cos":
        infonce = "infonce:metric=cosine"
        m_str = "cosine"
    elif metric == "ft":
        infonce = "infonce"
        m_str = "ft"
        n_epochs = 1500
    elif metric == "ft-cos":
        infonce = "infonce:metric=cosine"
        m_str = "mixed"
        n_epochs = 1500
    elif metric == "ftdot":
        infonce = "infonce:metric=dot"
        m_str = "mixed"
        n_epochs = 1500
    else:
        raise ValueError(f"Unknown {metric = }")

    runs = []
    name = sys.argv[2][: -(1 + len(dataset))]
    names = []
    params = dict(
        key=[],
        seed=[],
        out_dim=[],
        metric=[],
        n_epochs=[],
        backbone=[],
        path=[],
    )

    rng = np.random.default_rng(511622144)
    seeds = [None, *rng.integers(10_000, size=2)]
    dataloaders = [
        "dl" if s is None else f"dl:random_state={s}" for s in seeds
    ]

    for (seed, dl) in zip(seeds, dataloaders):

        p = prefix / dataset / dl / model / opt / lrsched / infonce / "train"
        if metric == "ft":
            p = p / expnames.finetune()
            dim = 2
        runs.append(p)
        names.append(f"{name}" + ("" if seed is None else f" {seed}"))
        params["path"].append(p)
        params["key"].append(name)
        params["metric"].append(m_str)
        params["seed"].append(seed)
        params["out_dim"].append(dim)
        params["backbone"].append(backbone)
        params["n_epochs"].append(n_epochs)

    gpu = "2080ti" if n_epochs < 5000 and backbone == "resnet18" else "v100"

    # those times are for cifar10, so this might need to be adjusted
    # for a more complicated dataset.
    t_epoch_hr = 0.0175 if gpu == "2080ti" else 0.014
    t_epoch_hr *= 3 if backbone == "resnet50" else 1

    t_total = t_epoch_hr * n_epochs
    t_total *= 1.05  # add some buffer

    t_hr = int(t_total)
    t_min = round((t_total - t_hr) * 60)
    t_str = f"{t_hr:02d}:{t_min:02d}:00"

    redo.redo_ifchange_slurm(
        [r / "losses.csv" for r in runs],
        partition=f"gpu-{gpu}-preemptable",
        time_str=t_str,
        name=names,
    )

    # evaluation of runs
    adict_files = {
        "knn[H]": [r / "knn" for r in runs],
        "knn[Z]": [r / "knn:layer=Z" for r in runs],
        "knn[euc, H]": [r / "knn:metric=euclidean" for r in runs],
        "knn[euc, Z]": [r / "knn:metric=euclidean:layer=Z" for r in runs],
        "sklin[H]": [r / "lin" for r in runs],
        "sklin[Z]": [r / "lin:layer=Z" for r in runs],
        "lin[H]": [r / expnames.lin_aug() for r in runs],
    }

    accfiles = []
    partitions = []
    time_strs = []
    names = []
    for key, val in adict_files.items():
        n = len(val)
        accfiles += [f / "score.txt" for f in val]
        if key == "lin[H]":
            p = "gpu-2080ti"
            t = "00:45:00"
        # those times for knn/ann assume cosine metric, which is
        # faster to compute than Euclidean.  There is a buffer, but it
        # might not hold for another metric.  Esp. if `n_trees=20` is
        # changed the performance might deteriorate.
        elif key.startswith("ann"):
            p = "cpu"
            t = "00:02:30"
        else:
            # normal knn and linear take < 1 minute.
            p = "cpu"
            t = "00:05:00"
        time_strs += n * [t]
        partitions += n * [f"{p}-preemptable"]
        names += n * [key]

    redo.redo_ifchange_slurm(
        accfiles,
        partition=partitions,
        time_str=time_strs,
        name=names,
    )

    adict = {
        key: [float((f / "score.txt").read_text()) for f in val]
        for key, val in adict_files.items()
    }
    adict["time[s]"] = [float((r / "default.run").read_text()) for r in runs]
    adict["loss"] = [
        pd.read_csv(r / "losses.csv")["mean"].tail(1).item() for r in runs
    ]
    ix = pd.MultiIndex.from_frame(pd.DataFrame(params))
    df = pd.DataFrame(adict, index=ix)

    df.to_csv(sys.argv[3])


if __name__ == "__main__":
    main()
