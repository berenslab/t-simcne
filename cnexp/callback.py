import tempfile
import zipfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from .eval.ann import ann_acc


def make_callbacks(
    outdir,
    dataloader: torch.utils.data.DataLoader,
    freq: int,
    model_save_freq=None,
    embedding_save_freq=None,
    ann_evaluate=True,
    seed=None,  # for ann evaluation
) -> tuple:
    """Set up callbacks, suitable for use in `train` (train.py).

    This function creates two callbacks, one for saving a torch model
    and another one for saving the transformed output of the given
    `dataloader`.  Both results are saved to a temporary zipfile,
    which is returned alongside the list of callbacks.

    There are three modes (pre-train, epoch, and post-train) that will be
    called as follows:
           - pre-train: will be passed as the mode once before the
             training starts
           - epoch (default): will be passed during training
           - post-train: like pre-train, but after the training loop


    Parameters
    ----------
    outdir: Path
        The directory where the model and embeddings will be saved to.
    dataloader: DataLoader
        A pytorch dataloader that will be transformed by the model.
        Should not be shuffled as the corresponding labels will only
        be saved once.
    freq: int
        The frequency with which the callbacks will be executing.  Can
        be overridden separately.
    model_save_freq: int or None (default: None)
        The frequency for saving the model and its parameters.  If
        this is not None and below 0 it will disable the callback
        entirely.
    embedding_save_freq: int or None (default: None)
        The frequency for saving the embedding.  If this is not None
        and below 0 it will disable the callback entirely.

    Returns
    -------
    list of callbacks
        Returns a list of callbacks that can then be used in train.py.

    """





    model_save_freq = freq if model_save_freq is None else model_save_freq

    def model_save_callback(
        model,
        epoch,
        loss,
        *,
        device="unused",
        mode="epoch",
        infodict=None,
        **_,
    ):
        if mode == "pre-train":
            (outdir / "model").mkdir(parents=True, exist_ok=True)
            # not interested in the model weights, they are the same
            # as they were during initialization, so they have been
            # saved already.
            pass
        elif mode == "epoch" and epoch % model_save_freq == 0:
            with open(
                outdir / f"model/epoch_{epoch:d}.pt",
                mode="w",
                buffering=2**18,
            ) as f:
                sd = dict(model=model, model_sd=model.state_dict())
                torch.save(sd, f)
        elif mode == "epoch":
            # not saving embeddings this time
            pass
        elif mode == "post-train":
            # not interested in the model weights, they are saved in
            # the final model.pt.
            pass
        else:
            raise ValueError(f"Unknown callback {mode = !r}")

    embedding_save_freq = (
        freq if embedding_save_freq is None else embedding_save_freq
    )

    def embedding_save_callback(
        model,
        epoch,
        loss,
        *,
        device="cuda:0",
        mode="epoch",
        infodict=None,
        **_,
    ):
        if (
            mode in ["pre-train", "post-train"]
            or mode == "epoch"
            and epoch % embedding_save_freq == 0
        ):
            features, backbone_features, labels = to_features(
                model, dataloader, device=device
            )

            name = (
                "pre"
                if mode == "pre-train"
                else "post"
                if mode == "post-train"
                else f"epoch_{epoch:d}"
            )
            with open(
                outdir / f"embeddings/{name}.npy", "w", buffering=2**18
            ) as f:
                np.save(f, features)
            with open(
                outdir / f"backbone_embeddings/{name}.npy",
                "w",
                buffering=2**18,
            ) as f:
                np.save(f, backbone_features)

            if mode == "pre-train":
                (outdir / "embeddings").mkdir(exist_ok=True)
                (outdir / "backbone_embeddings").mkdir(exist_ok=True)
                with open(outdir / "labels.npy", "w") as f:
                    np.save(f, labels)

            if ann_evaluate and infodict is not None:
                acc = ann_evaluation(features, labels, seed=seed)
                infodict["ann"] = f"{acc:.0%}"

        elif mode == "epoch":
            # not saving embeddings this time
            pass
        else:
            raise ValueError(f"Unknown callback {mode = !r}")

    callbacks = []
    if isinstance(model_save_freq, int) and model_save_freq > 0:
        callbacks.append(model_save_callback)
    if isinstance(embedding_save_freq, int) and embedding_save_freq > 0:
        callbacks.append(embedding_save_callback)

    return callbacks


def to_features(model, dataloader, device):
    """Iterate over a whole dataset/loader and return the results.

    This will transform all of the input in `dataloader` by passing it
    through the `model`.  The model will be put into eval mode
    `model.eval()`.  The result will be numpy arrays, suitable for
    storing with `np.save()`.

    """

    feats = []
    bb_feats = []
    labels = []

    model.eval()
    with torch.no_grad():
        for (im1, _im2), lbls in dataloader:
            features, backbone_features = model(im1.to(device))

            feats.append(features.cpu().numpy())
            bb_feats.append(backbone_features.cpu().numpy())
            labels.append(lbls)

    Z = np.vstack(feats).astype(np.float16)
    H = np.vstack(bb_feats).astype(np.float16)
    labels = np.hstack(labels)

    return Z, H, labels


def ann_evaluation(
    features, labels, *, metric="euclidean", n_trees=10, seed=None
):
    split = train_test_split(
        features, labels, stratify=labels, random_state=seed
    )

    if seed is not None:
        acc = ann_acc(*split, metric=metric, n_trees=n_trees, seed=seed)
    else:
        acc = ann_acc(*split, metric=metric, n_trees=n_trees)

    return acc
