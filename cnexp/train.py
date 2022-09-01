import inspect
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import tqdm.contrib.telegram as tqdm_telegram
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from .base import ProjectBase
from .callback import make_callbacks
from .misc.telegram_send import get_token_chat_id


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@contextmanager
def elapsed_time() -> float:
    """Context manager to measure the elapsed time in seconds.

    Returns a function that should be called right after the context
    ends to measure the elapsed time.

    Example
    -------

        with elapsed_time() as t:
            sleep(1)
        elapsed_time_in_secs = t()


    Notes
    -----
    Adapted from `https://stackoverflow.com/questions/33987060/
    python-context-manager-that-measures-time`.

    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


def train(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    opt: Optimizer,
    lrsched: _LRScheduler,
    n_epochs: int = None,
    device: torch.device = "cuda:0",
    print_epoch_freq: int = 2000,
    callbacks: list = None,
    **kwargs,
):
    n_epochs = get_n_epochs(n_epochs, lrsched)
    model.to(device)

    losses = torch.empty(n_epochs, len(dataloader))

    time_keys = [
        "t_dataload",
        "t_forward",
        "t_loss",
        "t_backward",
        "t_optstep",
        "t_batch",
    ]
    timedict = {
        key: np.empty((n_epochs, len(dataloader)), dtype=np.float16)
        for key in time_keys
    }

    lrs = np.empty(n_epochs + 1)
    lrs[0] = lrsched.get_last_lr()

    infodict = dict(lr=lrsched.get_last_lr())

    memdict = {
        "active_bytes.all.peak": [],
        "allocated_bytes.all.peak": [],
        "reserved_bytes.all.peak": [],
        "reserved_bytes.all.allocated": [],
    }

    if callbacks is not None:
        [
            c(
                model,
                float("nan"),
                float("inf"),
                device=device,
                mode="pre-train",
            )
            for c in callbacks
        ]

    try:
        rc = get_token_chat_id()
        name = os.getenv("SLURM_JOB_NAME")
        epochs_iter = tqdm_telegram.trange(
            n_epochs,
            desc=name,
            unit="epoch",
            ncols=120,
            postfix=infodict,
            leave=False,
            **rc,
        )
    except:
        epochs_iter = trange(
            n_epochs, unit="epoch", ncols=80, postfix=infodict
        )
    for epoch in epochs_iter:
        batch_ret = train_one_epoch(
            dataloader, model, criterion, opt, device=device, **kwargs
        )
        losses[epoch, :] = batch_ret["batch_losses"]
        mean_loss = losses[epoch, :].mean()
        for key in time_keys:
            timedict[key][epoch, :] = batch_ret[key]

        lr = lrsched.step()
        lrs[epoch + 1] = lr

        if torch.cuda.is_available():
            info = torch.cuda.memory_stats(device)
            [memdict[k].append(info[k]) for k in memdict.keys()]

        if callbacks is not None:
            [
                c(model, epoch, mean_loss, device=device, infodict=infodict)
                for c in callbacks
            ]

        infodict["loss"] = mean_loss.item()
        infodict["lr"] = lr
        epochs_iter.set_postfix(infodict, refresh=False)
        if (epoch + 1) % print_epoch_freq == 0:
            batch_time_secs = batch_ret["t_batch"].sum() / 1e9
            eprint(
                f"({epoch + 1: 3d}/{n_epochs: 3d}) "
                f"time: {batch_time_secs:.2f} \t"
                f"loss: {losses[epoch, :].mean(): .3f}, \t"
                f"lr: {lr:.4f}"
            )

    if callbacks is not None:
        [
            c(model, epoch, losses.mean(), device=device, mode="post-train")
            for c in callbacks
        ]

    ix = pd.RangeIndex(stop=losses.size(0), name="epoch")
    cols = pd.RangeIndex(stop=losses.size(1), name="batch")
    losses_df = pd.DataFrame(losses.numpy(), index=ix, columns=cols)
    mem_df = pd.DataFrame(memdict, index=ix)

    return dict(losses=losses_df, lrs=lrs, memory=mem_df, times=timedict)


def train_one_epoch(
    dataloader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    opt: Optimizer,
    device: torch.device,
    print_batch_freq: int = 100000,
    **kwargs,
):
    if kwargs.get("readout_mode", False):
        # if we do linear readout, then the projection head is the
        # only part that is supposed to be trained.
        model.backbone.eval()
        model.projection_head.train()
    else:
        model.train()

    losses = torch.empty(len(dataloader))
    # time dictionary
    td = {
        key: np.empty(len(dataloader), dtype=np.float16)
        for key in [
            "t_dataload",
            "t_forward",
            "t_loss",
            "t_backward",
            "t_optstep",
            "t_batch",
        ]
    }

    for i, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        unit="batch",
        ncols=80,
        mininterval=0.75,
        miniters=1,
        leave=False,
    ):
        with elapsed_time() as t_batch:
            with elapsed_time() as t:
                (data1, data2), orig_label = batch
                samples = torch.vstack((data1, data2)).to(device)
            td["t_dataload"][i] = t()

            with elapsed_time() as t:
                features, backbone_features = model(samples)
            td["t_forward"][i] = t()

            with elapsed_time() as t:
                # `backbone_features` and `labels` are usually
                # discarded in the loss.
                loss = criterion(
                    features,
                    backbone_features=backbone_features,
                    labels=orig_label,
                )
            td["t_loss"][i] = t()
            with elapsed_time() as t:
                opt.zero_grad()
                loss.backward()
            td["t_backward"][i] = t()
            with elapsed_time() as t:
                opt.step()
            td["t_optstep"][i] = t()

            losses[i] = loss.item()

        td["t_batch"][i] = t_batch()

        if (i + 1) % print_batch_freq == 0:
            eprint(f"batch {i + 1:5d}/{len(dataloader)}, loss {loss:.4f}")

    return dict(batch_losses=losses, **td)


def get_n_epochs(n_epochs, lrsched):
    if n_epochs is not None:
        return n_epochs
    else:
        try:
            return lrsched.n_epochs
        except AttributeError:
            raise RuntimeError(
                "n_epochs is None and there is no `n_epochs` member in the "
                "LR scheduler.  Specify at least one value."
            )


class TrainBase(ProjectBase):
    def __init__(
        self,
        path,
        random_state=None,
        callback_freq=50,
        model_save_freq=None,
        embedding_save_freq=None,
        **kwargs,
    ):
        super().__init__(path, random_state=random_state)
        self.callback_freq = callback_freq
        self.model_save_freq = model_save_freq
        self.embedding_save_freq = embedding_save_freq
        self.kwargs: dict = kwargs

        self.memdict_keys = [
            "active_bytes.all.peak",
            "allocated_bytes.all.peak",
            "reserved_bytes.all.peak",
            "reserved_bytes.all.allocated",
        ]

    def get_deps(self):
        filedeps = [inspect.getfile(make_callbacks)]
        return filedeps + [
            self.indir / f for f in ["dataset.pt", "model.pt", "criterion.pt"]
        ]

    def load(self):
        self.dataset_dict = torch.load(self.indir / "dataset.pt")
        self.dataloader = self.dataset_dict["train_contrastive_loader"]
        self.dataloader_plain = self.dataset_dict["full_plain_loader"]

        self.state_dict = torch.load(self.indir / "model.pt")
        sd = self.state_dict
        self.model = sd["model"]
        self.opt = sd["opt"]
        self.lr = sd["lrsched"]

        self.criterion_dict = torch.load(self.indir / "criterion.pt")
        self.criterion = self.criterion_dict["criterion"]

        self.callback_seed = self.random_state.integers(2**31 - 1)
        self.callbacks, self.zipf_dict = make_callbacks(
            self.outdir,
            self.dataloader_plain,
            self.callback_freq,
            self.model_save_freq,
            self.embedding_save_freq,
            seed=self.callback_seed,
        )

    def compute(self):
        self.retdict = train(
            self.dataloader,
            self.model,
            self.criterion,
            self.opt,
            self.lr,
            callbacks=self.callbacks,
            **self.kwargs,
        )
        self.losses: pd.DataFrame = self.retdict["losses"]
        self.memory: pd.DataFrame = self.retdict["memory"]
        self.learning_rates: np.ndarray = self.retdict["lrs"]

    def save(self):
        self.save_lambda_alt(
            self.outdir / "model.pt", self.state_dict, torch.save
        )

        self.zipf_dict["zip"].close()
        tempf = self.zipf_dict["tmp"]
        (self.outdir / "intermediates.zip").unlink(missing_ok=True)
        os.link(tempf.name, self.outdir / "intermediates.zip")
        tempf.close()

        self.save_lambda(
            self.outdir / "losses.npy", self.losses.values, np.save
        )
        self.losses["mean"] = self.losses.mean(axis=1)
        self.save_lambda(
            self.outdir / "losses.csv", self.losses, lambda f, df: df.to_csv(f)
        )
        self.save_lambda(
            self.outdir / "memory.csv", self.memory, lambda f, df: df.to_csv(f)
        )

        self.save_lambda(
            self.outdir / "learning_rates.npy", self.learning_rates, np.save
        )

        self.save_lambda(
            self.outdir / "times.npz",
            self.retdict["times"],
            lambda f, d: np.savez(f, **d),
        )
