#!/usr/bin/env python
"""
The dataset can be used as an argument in stats/sampling/, e.g. `redo
stats/acctbl/cifar.csv` will calculate the old experiment table (various
losses for some negative samples).
"""
import sys

import pandas as pd
from cnexp import redo


def main():

    # prefix = Path("../../experiments")
    dataset = sys.argv[2]

    # , "euc.2d.long" <-- start this one manually for now (so it does
    # not block for all eternity).
    keys = ["infonce", "neg_sample", "nce"]
    df_files = [f"{key}.{dataset}.part.csv" for key in keys]

    redo.redo_ifchange([f for f in df_files])

    df = pd.concat(
        [pd.read_csv(f) for f in df_files], ignore_index=True, copy=False
    )
    df.drop(columns="path", inplace=True, errors="ignore")
    df.to_csv(sys.argv[3], index=False)


if __name__ == "__main__":
    main()
