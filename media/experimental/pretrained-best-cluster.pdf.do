#!/usr/bin/env python

import sys
from pathlib import Path

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cnexp import plot, redo
from cnexp.plot.scalebar import add_scalebar_frac


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():

    root = Path("../../experiments/")
    # prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    dname_dict = dict(
        cifar="CIFAR-10",
        cifar100="CIFAR-100",  # tiny="Tiny ImageNet"
    )

    df_files_ = []
    npz_files = []
    resnet_nums = [18, 34, 50, 101, 152]
    resnets = [f"resnet{n}.tsne" for n in resnet_nums]
    emb_str = "tsne"
    embedding = "t-SNE" if emb_str == "tsne" else "UMAP"
    npzkeynames = ["t-SimCNE"] + [
        f"{embedding}(ResNet{n})" for n in resnet_nums
    ]
    only_violin = [f"{embedding}(ResNet{n})" for n in [34, 50]]
    for dataname in dname_dict.keys():
        npz_file = (
            root.parent / f"stats/{dataname}.{emb_str}.pretrained.resnet.npz"
        )
        npz_files.append(npz_file)

        df_files = []
        for name in ["tsimcne"] + resnets:
            reps = root.parent / f"stats/cluster/{dataname}.{name}.csv"
            # redo.redo_ifchange(reps)
            df_files.append(reps)
        df_files_.append(df_files)

    df_files_ = np.array(df_files_)
    redo.redo_ifchange(npz_files + list(df_files_.flat))

    with plt.style.context(stylef):
        fig, axxs = plt.subplots(
            len(dname_dict),
            1 + len(df_files) - len(only_violin),
            figsize=(5.5, len(dname_dict) * 1.5),
            squeeze=False,
        )
        for axs, df_files, npz_file, name in zip(
            axxs, df_files_, npz_files, dname_dict.values()
        ):
            npz = np.load(npz_file)
            labels = npz["labels"]

            datakey = "ari"
            files = [
                (fname, key)
                for fname, key in zip(df_files, npzkeynames)
                if key not in only_violin
            ]
            for i, (ax, (df_fname, key)) in enumerate(zip(axs[1:], files)):
                df = pd.read_csv(df_fname)
                Y = npz[key]

                best_idx = df[datakey].argmax()
                # ami = df["ami"].loc[best_idx]
                # ari = df["ari"].loc[best_idx]
                n_clusters = df["n_clusters"].loc[best_idx]
                # best_ari = df["ari"].max()
                param_df = df[
                    [
                        "min_cluster_size",
                        "min_sample",
                        "cluster_selection_epsilon",
                    ]
                ]
                best_params = param_df.iloc[best_idx]
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=best_params[0],
                    min_samples=best_params[1],
                    cluster_selection_epsilon=best_params[2],
                )
                # title = (
                #     f"{key.replace('ResNet', 'RN')}"
                #     # f"\n({best_params[0]}, {best_params[1]}, {best_params[2]})"
                # )
                ax.set_title(key)
                preds = clusterer.fit_predict(Y)

                ax.scatter(
                    Y[:, 0],
                    Y[:, 1],
                    c=labels,
                    alpha=0.5,
                    rasterized=True,
                    zorder=4.5,
                )

                # txt = f"AMI = {ami:.2f}\nARI = {best_ari:.2f} ({ari:.2f})"
                txt = (
                    f"min_cluster_size = {best_params[0]}\n"
                    f"min_samples = {best_params[1]}\n"
                    # it's one too low in the file because we use
                    # pred_labels.max() but it starts counting from 0
                    f"{n_clusters + 1} clusters"
                )
                ax.text(
                    0,
                    -0.1,
                    txt,
                    ha="left",
                    va="top",
                    ma="left",
                    fontsize="small",
                    transform=ax.transAxes,
                )

                for cluster_idx in np.unique(preds):
                    if cluster_idx < 0:
                        continue
                    Xc = Y[preds == cluster_idx]
                    cmean = Xc.mean(axis=0)
                    ax.scatter(
                        [cmean[0]], [cmean[1]], marker="x", c="black", zorder=5
                    )
                add_scalebar_frac(ax)
                ax.margins(0)

            ax = axs[0]
            datasets = []
            default_scores = []
            xticks = list(range(len(df_files)))
            for i, fname in zip(xticks, df_files):
                df = pd.read_csv(fname)
                datasets.append(df[datakey])
                best_score = df[datakey].max()
                default_row = df[
                    (df.min_cluster_size == 5) & (df.min_sample == 5)
                ]
                default_score = default_row[datakey].item()
                default_scores.append(default_score)

                y = best_score
                ax.text(
                    i,
                    y,
                    f"{best_score:.0%}",
                    # bbox=dict(pad=5),
                    fontsize="small",
                    ha="center",
                    va="bottom",
                    label="score",
                )

                ax.update_datalim([[i, y]])

            ax.set_title(name)
            ax.violinplot(
                datasets,
                positions=xticks,
                showmeans=True,
                showmedians=True,
                points=1000,
            )

            label = (
                "default parameters\n"
                "min_cluster_size = 5\n"
                "min_samples = 5"
            )
            ax.plot(
                xticks,
                default_scores,
                c="xkcd:dark grey",
                lw=0,
                marker="o",
                label=label,
                clip_on=False,
                # label="default parameters",
            )

            # legend = ax.legend(
            #     loc="upper right",
            #     fontsize="small",
            #     markerscale=1.5,
            #     handletextpad=0.05,
            #     borderpad=0.35,
            #     borderaxespad=0,
            # )
            # [txt.set_multialignment("right") for txt in legend.get_texts()]
            tr = dict(
                ari="Adjusted Rand index", ami="Adjusted mutual information"
            )
            ax.set_ylabel(tr[datakey])
            ax.set_xticks(
                xticks,
                [s.replace("ResNet", "RN") for s in npzkeynames],
                fontsize="small",
                rotation_mode="anchor",
                rotation=45,
                ha="right",
            )

            ymin, ymax = np.round(ax.get_ylim(), decimals=1).clip(0)
            ymax = 0.8 if name == "CIFAR-10" else ymax
            yticks = [ymin]
            y = ymin + (0.2 if name == "CIFAR-10" else 0.1)
            while y < ymax:
                yticks.append(y)
                y += 0.2
            yticks.append(ymax)
            ax.set_yticks(yticks, [f"{y:.1f}" for y in yticks])

            ax.spines.left.set_bounds(ymin, ymax)
            ax.spines.bottom.set_bounds(xticks[0], xticks[-1])

            # pad the score numbers a bit, relative to the space in
            # the y-dimension.  We manually transform it into the data
            # coords so that the constrained layout can optimize the
            # position freely.
            y1, y2 = ax.get_ylim()
            ylen = y2 - y1
            pad = ylen * 0.0125
            scoretxts = [
                child
                for child in ax.get_children()
                if isinstance(child, plt.Text) and child.get_label() == "score"
            ]
            for txt in scoretxts:
                x, y = txt.get_position()
                txt.set_position((x, y + pad))

        plot.add_letters(axxs)
        plot.add_lettering(axxs[0, 0], "a", ha="right")
        plot.add_lettering(axxs[1, 0], "f", ha="right")

    metadata = plot.get_default_metadata()
    metadata["Title"] = "Visualization of best clustering t-SimCNE vs. ResNets"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
