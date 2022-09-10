#!/usr/bin/env python
"""
The dataset can be used as an argument in stats/acctbl/, e.g. `redo
stats/acctbl/cifar.csv` will calculate the old accuracy table (various
metrics for some run configurations).  This approach is better
compartmentalized (which hopefully helps down the line).
"""
import sys

import pandas as pd
from cnexp import redo


def main():
    # rng = np.random.default_rng(511622144)
    # seeds = [None, *rng.integers(10_000, size=2)]

    # prefix = Path("../../experiments")
    dataset = sys.argv[2]

    # , "euc.2d.long" <-- start this one manually for now (so it does
    # not block for all eternity).
    keys = ["cos", "cos.3d", "euc", "euc.2d", "ft", "ft-cos"]
    df_files = [f"{key}.{dataset}.part.csv" for key in keys]

    # since "euc" is a subset of "ft" run we redo it afterwards so the
    # run has already finished and it only needs to link the files
    # from the experiment run.  The same holds for "cos" and "ft-cos".
    redo.redo_ifchange(
        [f for k, f in zip(keys, df_files) if k not in ["euc", "cos"]]
    )
    redo.redo_ifchange(
        [f for k, f in zip(keys, df_files) if k in ["euc", "cos"]]
    )

    df = pd.concat(
        [pd.read_csv(f) for f in df_files], ignore_index=True, copy=False
    )
    df.drop(columns="path", inplace=True, errors="ignore")
    df.to_csv(sys.argv[3], index=False)


if __name__ == "__main__":
    main()
