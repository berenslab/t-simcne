#!/usr/bin/env python

import sys
import zipfile
from pathlib import Path

import numpy as np
import openTSNE
import torch
from cnexp import names, redo
from sklearn.decomposition import PCA


def do_pca(data, seed, n_components=50, **kwargs):
    pca = PCA(random_state=seed, n_components=n_components, **kwargs)
    return pca.fit_transform(data)


def do_tsne(data, seed, n_jobs=-1, **kwargs):
    tsne = openTSNE.TSNE(n_jobs=n_jobs, random_state=seed, **kwargs)
    return tsne.fit(data)


def main():
    root = Path("../experiments")
    prefix = root / sys.argv[2] / "dl"

    rng = np.random.default_rng(342561)

    path = prefix / names.default_train(metric="cosine")
    # redo.redo_ifchange(path / "intermediates.zip")
    with zipfile.ZipFile(path / "out/intermediates.zip") as zf:
        with zf.open("backbone_embeddings/post.npy") as f:
            data = np.load(f)
        with zf.open("labels.npy") as f:
            labels = np.load(f)

    Y_tsne = do_tsne(data, rng.integers(2**32))
    Y_pca = do_pca(data, rng.integers(2**32))

    with open(sys.argv[3], "wb") as f:
        np.savez(
            f,
            tsne=Y_tsne.astype("float16"),
            pca=Y_pca.astype("float16"),
            labels=labels,
        )


if __name__ == "__main__":
    main()
