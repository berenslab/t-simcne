#!/usr/bin/env python

preamble = r"""
\documentclass[margin=0.1cm]{standalone}
\usepackage{booktabs}
\usepackage{tabularx}

\begin{document}
"""
r"""
  \begin{tabularx}{\linewidth}{rX}
    \toprule
    a & b \\
    \midrule
    1 & 2 \\
    \bottomrule
  \end{tabularx}
"""
postscript = r"""
\end{document}
"""
import re
import sys

import pandas as pd
from cnexp import redo


def mean_std_df(
    df: pd.DataFrame, columns=["out_dim", "metric"]
) -> pd.DataFrame:

    df = df.drop(columns=["seed", "time[s]"])
    acc_cols = [
        c for c in df.columns if c.find("[Z]") > 0 or c.find("[H]") > 0
    ]
    new_cols = {old: old[:-1] + ", %]" for old in acc_cols}
    df[acc_cols] = df[acc_cols].apply(lambda x: x * 100)
    df = df.rename(columns=new_cols)

    dfg = df.groupby(columns)
    df_mean = dfg.mean().rename(columns=lambda x: f"μ{x}")
    df_std = dfg.std().rename(columns=lambda x: f"σ{x}")

    # https://stackoverflow.com/questions/46584736/
    # pandas-change-between-mean-std-and-plus-minus-notations
    dfj = df_mean.join(df_std)
    df1 = dfj.groupby(dfj.columns.str[1:], axis=1).apply(
        lambda x: x.round(1).astype(str).apply("±".join, 1)
    )

    return df1


def transl(key: str) -> str:
    tr = {
        "lin[H]": "Linear (augm.)",
        "knn[Z]": "$k$NN in $Z$",
        "loss": "Loss",
        "time[s]": "Time",
    }
    if key.startswith("time"):
        unit = re.match(r"time\[(\w+)\]", key)[1]
        return f"Time ({unit})"
    else:
        return tr[key]


def main():
    fname = "../../stats/accuracy-table-runs.csv"
    redo.redo_ifchange(fname)

    df: pd.DataFrame = pd.read_csv(fname)

    colkeys = ["lin[H]", "knn[Z]", "loss", "time[hr]"]
    df["time[hr]"] = df["time[s]"] / (60 * 60)
    assert all(key in df.columns for key in colkeys)

    dfg = df.groupby(["out_dim", "metric"])

    table_head = ["Training regime"] + [transl(k) for k in colkeys]
    with open(sys.argv[3], "w") as f:

        def nl():
            f.write("\n")

        def tblnl():
            f.write(r"\\" "\n")

        f.write(preamble)

        f.write(
            r"\begin{tabularx}{\linewidth}"
            f"{{X{'c' * (len(table_head) - 1)}}}"
        )
        nl()
        f.write(r"\toprule")
        nl()
        it = iter(table_head)
        f.write(f"{next(it)} ")
        [f.write(f"& {th} ") for th in it]
        tblnl()
        f.write(r"\midrule")
        nl()

        for (out_dim, metric), df1 in dfg:
            desc = metric.capitalize() + f" {out_dim}D "
            f.write(desc)

            for key in colkeys:
                mean = df1[key].mean()
                std = df1[key].std()
                # accuracy in percent
                if any(key.startswith(w) for w in ["lin", "knn", "sklin"]):
                    mean *= 100
                    std *= 100
                    f.write(f"& ${mean:.1f}\\pm{std:.1f}\\%$ ")
                else:
                    f.write(f"& ${mean:.1f}\\pm{std:.1f}$ ")
            tblnl()

        f.write(r"\bottomrule")
        nl()
        f.write(r"\end{tabularx}")
        f.write(postscript)

        f.write("\n\n------------------------------\n\n")
        f.write("the following will be ignored")
        f.write("\n\n------------------------------\n\n")

        mean_std_df(df).T.to_string(f)
        nl()


if __name__ == "__main__":
    main()
