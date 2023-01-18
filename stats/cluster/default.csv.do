#!/bin/env python

import itertools
import sys

import hdbscan
import numpy as np
import pandas as pd
from cnexp import redo
from sklearn import metrics


def main():
    parts = sys.argv[2].split(".")
    if len(parts) == 2:
        dataset = parts[0]
        assert "tsimcne" == parts[1]
        embedding = "tsne"
        load_str = "t-SimCNE"
    elif len(parts) == 3:
        dataset, resnet, embedding = parts
        resnet_str = resnet.replace("resnet", "ResNet")
        if embedding == "tsne":
            load_str = f"t-SNE({resnet_str})"
        elif embedding == "umap":
            load_str = f"UMAP({resnet_str})"
        else:
            raise RuntimeError(f"got unknown {embedding = !r}")
    else:
        raise RuntimeError(f"got {sys.argv[2]!r}, don't know how to process")

    npzfile = f"../{dataset}.{embedding}.pretrained.resnet.npz"
    redo.redo_ifchange(npzfile)
    npz = np.load(npzfile)

    labels = npz["labels"]
    # eprint(load_str)
    Y = npz[load_str].astype(float)

    min_cluster_sizes = list(range(5, 301, 10))
    min_samples = list(range(5, 151, 10))
    # min_samples = [5]
    cluster_selection_epsilons = [0]  # , 1, 2
    param_iter = itertools.product(
        min_cluster_sizes, min_samples, cluster_selection_epsilons
    )
    params = dict(
        min_cluster_size=[], min_sample=[], cluster_selection_epsilon=[]
    )

    results = dict(ari=[], ami=[], n_clusters=[])
    for param_conf in param_iter:
        min_cluster_size, min_samples, eps = param_conf
        if min_samples > min_cluster_size:
            continue

        [
            params[key].append(param)
            for key, param in zip(params.keys(), param_conf)
        ]

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=eps,
        )
        # clusterer = MiniBatchKMeans(n_classes, random_state=10110101)
        preds = clusterer.fit_predict(Y)
        ari = metrics.adjusted_rand_score(labels, preds)
        ami = metrics.adjusted_mutual_info_score(labels, preds)
        n_clusters = preds.max()
        [
            results[key].append(r)
            for key, r in zip(results.keys(), [ari, ami, n_clusters])
        ]

    idx_df = pd.DataFrame(params)
    idx = pd.MultiIndex.from_frame(idx_df)

    df = pd.DataFrame(results, index=idx)
    df.to_csv(sys.argv[3])


if __name__ == "__main__":
    main()
