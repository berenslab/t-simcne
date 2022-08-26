#!/usr/bin/env python
# -*- mode: python -*-
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from cnexp import names as expnames
from cnexp import redo


def main():
    rng = np.random.default_rng(511622144)
    seeds = [None, *rng.integers(10_000, size=2)]

    prefix = Path("../experiments/cifar")

    dataloaders = [
        "dl" if s is None else f"dl:random_state={s}" for s in seeds
    ]

    out_dims = [128, 3]
    models = [
        "model" if dim == 128 else f"model:out_dim={dim}" for dim in out_dims
    ]
    opt = "sgd"
    lrsched = "lrcos"
    lrsched5k = "lrcos:n_epochs=5000"
    # p = Path("model/sgd/lrcos:n_epochs=3/infonce/train")

    infonce_cos = "infonce:metric=cosine"
    infonce = "infonce"

    pretrain_for_2d = []
    runs = []
    partitions = []
    names = []
    params = dict(seed=[], out_dim=[], metric=[])
    # cosine
    for (seed, dl), (out_dim, model) in itertools.product(
        zip(seeds, dataloaders), zip(out_dims, models)
    ):
        p = prefix / dl / model / opt / lrsched / infonce_cos / "train"
        runs.append(p)
        partitions.append("gpu-2080ti-preemptable")
        names.append(f"{out_dim}d " + ("" if seed is None else f"(rs={seed})"))
        params["metric"].append("cosine")
        params["seed"].append(seed)
        params["out_dim"].append(out_dim)

        if out_dim == 128:
            pretrain_for_2d.append(p)

    # Euclidean
    out_dims = [128, 2]
    models = [
        "model" if dim == 128 else f"model:out_dim={dim}" for dim in out_dims
    ]
    model2D = models[1]

    for (seed, dl), (out_dim, model) in itertools.product(
        zip(seeds, dataloaders), zip(out_dims, models)
    ):
        p = prefix / dl / model / opt / lrsched / infonce / "train"
        runs.append(p)
        partitions.append("gpu-2080ti-preemptable")
        names.append(
            f"{out_dim}d E." + ("" if seed is None else f"(rs={seed})")
        )
        params["metric"].append("euclidean")
        params["seed"].append(seed)
        params["out_dim"].append(out_dim)

    # 5000 epoch run
    for seed, dl in zip(seeds, dataloaders):
        p = prefix / dl / model2D / opt / lrsched5k / infonce / "train"
        runs.append(p)
        partitions.append("gpu-v100-preemptable")
        names.append(
            f"{out_dim}d 5k" + ("" if seed is None else f"(rs={seed})")
        )
        params["metric"].append("euclidean (long)")
        params["seed"].append(seed)
        params["out_dim"].append(2)

    #
    # finetune model
    #
    ft_runs = []
    ft_partitions = []
    ft_names = []
    for prefix, seed, dl in zip(pretrain_for_2d, seeds, dataloaders):
        p = prefix / expnames.finetune()

        ft_runs.append(p)
        ft_partitions.append("gpu-2080ti-preemptable")
        ft_names.append("ft " + ("" if seed is None else f"({seed})"))
        params["metric"].append("finetuned")
        params["seed"].append(seed)
        params["out_dim"].append(2)

    # redo.redo_ifchange_slurm(
    #     [r / "losses.csv" for r in ft_runs],
    #     partition="gpu-2080ti-preemptable",
    #     time_str="17:20:00",
    #     name=ft_names,
    # )

    runs += ft_runs
    partitions += ft_partitions
    names += ft_names
    redo.redo_ifchange_slurm(
        [r / "losses.csv" for r in runs],
        partition=partitions,
        time_str="70:00:00",
        name=names,
    )

    # evaluation of runs
    adict_files = {
        "knn[H]": [r / "knn" for r in runs],
        "knn[Z]": [r / "knn:layer=Z" for r in runs],
        "ann[H]": [r / "ann" for r in runs],
        "ann[Z]": [r / "ann:layer=Z" for r in runs],
        "sklin[H]": [r / "lin" for r in runs],
        "sklin[Z]": [r / "lin:layer=Z" for r in runs],
        "lin[H]": [r / expnames.lin_aug() for r in runs],
    }

    redo.redo_ifchange(
        [
            r.parent / f
            for r in adict_files["lin[H]"]
            for f in ["model.pt", "criterion.pt", "default.run"]
        ]
    )

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
            t = "00:05:30"
        else:
            # normal knn and linear take < 1 minute.
            p = "cpu"
            t = "00:07:00"
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
