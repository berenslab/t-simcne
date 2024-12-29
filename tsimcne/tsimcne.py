import zipfile
from collections import defaultdict
from pathlib import Path

import lightning
import numpy as np
import PIL
import torch
from annoy import AnnoyIndex
from lightning.pytorch.core.mixins import HyperparametersMixin
from scipy import stats
from sklearn.model_selection import train_test_split

from .imagedistortions import (
    TransformedPairDataset,
    get_transforms_unnormalized,
)
from .losses.infonce import (
    CauchyTemp,
    CosineTemp,
    InfoNCECauchy,
    InfoNCECosine,
    InfoNCEGaussian,
)
from .lrschedule import (
    ConstantSchedule,
    CosineAnnealingSchedule,
    LinearSchedule,
)
from .models.mutate_model import mutate_model
from .models.simclr_like import make_model


class PLtSimCNE(lightning.LightningModule, HyperparametersMixin):
    def __init__(
        self,
        model=None,
        loss="infonce",
        metric=None,
        temperature=0.5,
        backbone="resnet18_sm_kernel",
        backbone_dim=None,
        projection_head="mlp",
        n_epochs=1000,
        batch_size=512,
        out_dim=128,
        anneal_to_dim=2,
        optimizer_name="sgd",
        lr_scheduler_name="cos_annealing",
        lr="auto_batch",
        dim_annealing="cos_annealing",  # or "piecewise-linear"
        dof="parameter",  # (inital_dof, final_dof), dof=1.0
        weight_decay=5e-4,
        momentum=0.9,
        warmup_epochs=10,
        batches_per_epoch=None,
        random_state=None,
        save_intermediate_feat=False,
        save_intermediate_bbf=False,
        eval_ann=True,
        eval_function=None,
    ):
        super().__init__()
        ignore_list = [
            "loss",
            "backbone",
            "projection_head",
            "eval_function",
            "model",
        ]
        self.save_hyperparameters(ignore=ignore_list)
        self.model = model
        self.loss = loss
        self.metric = metric
        self.temperature = temperature
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.projection_head = projection_head
        self.n_epochs = n_epochs
        self.out_dim = out_dim
        self.anneal_to_dim = anneal_to_dim
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.lr = lr
        self.dim_annealing = dim_annealing
        self.dof = dof
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs
        self.batches_per_epoch = batches_per_epoch
        self.random_state = random_state
        self.save_intermediate_feat = save_intermediate_feat
        self.save_intermediate_bbf = save_intermediate_bbf
        self.eval_ann = eval_ann
        self.eval_function = eval_function

        self._handle_parameters()

    def _handle_parameters(self):
        if self.model is None:
            self.model = make_model(
                backbone=self.backbone,
                projection_head=self.projection_head,
                out_dim=self.out_dim,
                backbone_dim=self.backbone_dim,
            )

        if self.loss == "infonce":
            if self.metric is None:
                self.metric = "euclidean"

            if self.metric == "euclidean":
                self.loss = InfoNCECauchy(temperature=self.temperature)
            elif self.metric == "cosine":
                self.loss = InfoNCECosine(temperature=self.temperature)
            elif self.metric == "gauss":
                self.loss = InfoNCEGaussian(temperature=self.temperature)
            else:
                raise ValueError(
                    f"Unknown {self.metric = !r} for InfoNCE loss"
                )
        elif self.loss == "infonce-temp":
            if self.metric is None:
                self.metric = "euclidean"

            if self.metric == "euclidean":
                self.loss = CauchyTemp(temperature=self.temperature)
            elif self.metric == "cosine":
                self.loss = CosineTemp(temperature=self.temperature)
            else:
                raise ValueError(
                    f"Unknown {self.metric = !r} for InfoNCE loss"
                )

        # else: assume that the loss is a proper pytorch loss function

        if self.lr == "auto_batch" and self.optimizer_name == "sgd":
            self.lr = self.lr_from_batchsize(self.batch_size)
        elif self.lr == "auto_batch" and self.optimizer_name.startswith(
            "adam"
        ):
            self.lr = 1e-3

        initial_dim = self.out_dim
        final_dim = self.anneal_to_dim
        epochs = self.n_epochs

        match self.dim_annealing:
            case "cos_annealing":
                # # cosine annealing
                sched = final_dim + 0.5 * (initial_dim + 1 - final_dim) * (
                    1
                    + torch.cos(
                        torch.pi * torch.arange(1, epochs + 1) / epochs
                    )
                )
                self._dims = sched.floor().to(int).numpy()

            case "piecewise-linear":
                # piecewise linear annealing
                a1 = np.full(epochs * 2 // 3, initial_dim, dtype=int)
                a2 = np.linspace(
                    initial_dim,
                    final_dim,
                    epochs // 8,
                    dtype=int,
                    endpoint=False,
                )
                a3 = np.full(
                    epochs - (a1.shape[0] + a2.shape[0]), final_dim, dtype=int
                )
                self._dims = np.hstack((a1, a2, a3))
            case "linear":
                self._dims = np.linspace(initial_dim, final_dim, self.n_epochs)
            case "piecewise-cos":
                # piecewise linear annealing
                a1 = np.full(epochs * 2 // 3, initial_dim, dtype=int)
                e2 = epochs // 8
                sched = final_dim + 0.5 * (initial_dim + 1 - final_dim) * (
                    1 + torch.cos(torch.pi * torch.arange(1, e2 + 1) / e2)
                )
                a2 = sched.floor().to(int).numpy()
                a3 = np.full(
                    epochs - (a1.shape[0] + a2.shape[0]), final_dim, dtype=int
                )
                self._dims = np.hstack((a1, a2, a3))
            case _:
                raise ValueError(
                    "Allowed values for dim_annealing are "
                    "'cos_annealing' and 'piecewise-linear', "
                    f"got {self.dim_annealing!r}."
                )

        self.dim_mask_schedule = [
            slice(initial_dim if d >= initial_dim else d - initial_dim)
            for d in self._dims
        ]

        self.dim_mask = self.dim_mask_schedule[0]
        self.train_embeddings = None

        match self.dof:
            case (initial_dof, final_dof):
                # final_dof = 128
                # initial_dof = 1

                n_epochs = self.n_epochs
                self.dofs = final_dof + 0.5 * (initial_dof - final_dof) * (
                    1
                    + torch.cos(
                        torch.pi * torch.arange(0, n_epochs) / n_epochs
                    )
                )

            case "parameter":
                self.dof_ = torch.nn.Parameter(
                    torch.tensor(1.0, dtype=torch.float64)
                )
                self.dofs = [self.dof_] * self.n_epochs

            case "dim-sub1":
                self.dofs = [d - 1 for d in self._dims]

            case list(dof):
                if len(dof) != self.n_epochs:
                    raise ValueError(
                        "Got list of dofs, but it is unequal to the number of "
                        f"epochs. {self.n_epochs} epochs, but {len(dof)=} "
                        "values supplied."
                    )
                self.dofs = dof

            case int(dof) | float(dof):
                self.dofs = [dof] * self.n_epochs

            case dof:
                raise ValueError(
                    f"Got unexpected value for {dof = !r}. "
                    "Please supply 'parameter', 'dim-sub1', a pair, "
                    "a list of numbers, or a constant for `dof`."
                )

        if self.optimizer_name not in ["sgd", "adam", "adamw"]:
            raise ValueError(
                "Only 'sgd', 'adam', or 'adamw' is supported as optimizer, "
                f"got {self.optimizer_name}."
            )

        self.alphas = torch.sin(
            torch.linspace(0, 1, self.n_epochs) * torch.pi / 2
        )
        self.alpha = self.alphas[0]

        if self.random_state is None:
            self.rng = np.random.default_rng()
        elif isinstance(self.random_state, np.random.RandomState):
            seed = self.random_state.randint(-(2**31), 2**31)
            self.rng = np.random.default_rng(seed)
        elif isinstance(self.random_state, np.random.Generator):
            self.rng = self.random_state

    def configure_optimizers(self):
        opts = []
        if self.optimizer_name == "sgd":
            opts.append(
                torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.lr,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay,
                )
            )
            if sum(p.numel() for p in self.loss.parameters()) > 0:
                opts.append(
                    torch.optim.SGD(
                        self.loss.parameters(),
                        lr=self.lr / 100,
                        momentum=self.momentum,
                        weight_decay=self.weight_decay,
                    )
                )
        elif self.optimizer_name == "adam":
            opts.append(
                torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    amsgrad=True,
                )
            )
            if sum(p.numel() for p in self.loss.parameters()) > 0:
                opts.append(
                    torch.optim.Adam(
                        self.loss.parameters(),
                        lr=self.lr / 100,
                        weight_decay=self.weight_decay,
                        amsgrad=True,
                    )
                )
        elif self.optimizer_name == "adamw":
            opts.append(
                torch.optim.AdamW(
                    self.model.parameters(),
                    lr=self.lr,
                    weight_decay=self.weight_decay,
                    amsgrad=False,
                    eps=1e-6,
                )
            )
            if sum(p.numel() for p in self.loss.parameters()) > 0:
                opts.append(
                    torch.optim.AdamW(
                        self.loss.parameters(),
                        lr=self.lr / 100,
                        weight_decay=self.weight_decay,
                        amsgrad=False,
                        eps=1e-6,
                    )
                )

        effective_n_epochs = (
            self.n_epochs
            if self.batches_per_epoch is None
            else self.n_epochs * self.batches_per_epoch
        )
        effective_warmup_epochs = (
            self.warmup_epochs
            if self.batches_per_epoch is None
            else self.warmup_epochs * self.batches_per_epoch
        )
        match self.lr_scheduler_name:
            case "cos_annealing":
                lrs = [
                    CosineAnnealingSchedule(
                        opt,
                        n_epochs=effective_n_epochs,
                        warmup_epochs=effective_warmup_epochs,
                    )
                    for opt in opts
                ]
            case "constant":
                lrs = [
                    ConstantSchedule(
                        opt,
                        n_epochs=effective_n_epochs,
                        warmup_epochs=effective_warmup_epochs,
                    )
                    for opt in ops
                ]
            case "linear":
                lrs = [
                    LinearSchedule(
                        opt,
                        n_epochs=effective_n_epochs,
                        warmup_epochs=effective_warmup_epochs,
                    )
                    for opt in opts
                ]
            case _:
                raise ValueError(
                    "Expected 'cos_annealing', 'constant', or 'linear', got "
                    f"{self.lr_scheduler_name=!r}"
                )

        return opts, [
            {
                "scheduler": lrsched,
                "interval": (
                    "epoch" if self.batches_per_epoch is None else "step"
                ),
            }
            for lrsched in lrs
        ]

    def on_train_epoch_start(self):
        if hasattr(self, "dof_"):
            self.log("dof", self.dof_.item(), prog_bar=True)
        else:
            self.log("dof", self.cur_dof, prog_bar=False)

        self.dim_mask = self.dim_mask_schedule[self.current_epoch]
        self.cur_dof = self.dofs[self.current_epoch]

        # reset the train embeddings for the next epoch
        self.train_embeddings = []

    def training_step(self, batch, batch_idx):
        # calls self.forward()
        features, backbone_features = self(batch)

        # assert features.size(0) % 2 == 0, f"{features.shape = } is wrong!"

        # backbone_features are not used in infonce loss
        # loss = self.loss(features, alpha=self.alpha)
        lossd = self.loss(features)

        return lossd

    @staticmethod
    def fix_ffcv_batch(batch):
        if len(batch) == 3:
            x1, lbl, x2 = batch
            x = torch.vstack((x1, x2))
            batch = x, lbl
        return batch

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(outputs, prog_bar=True)

    def on_validation_epoch_start(self):
        self.val_batches = defaultdict(list)
        self.cur_dof = self.dofs[self.current_epoch]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch = self.fix_ffcv_batch(batch)
        if dataloader_idx == 0:
            # calls self.forward()
            features, backbone_features = self(batch)

            # backbone_features are unused in infonce loss
            lossd = self.loss(features)

            return lossd

        elif dataloader_idx == 1:
            features, backbone_features = self(batch)
            labels = batch[1]
            return dict(
                Z_batch=features,
                H_batch=backbone_features,
                labels_batch=labels,
            )

    def on_validation_batch_end(
        self, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if dataloader_idx == 0:
            renamed_outputs = {f"val_{k}": v for k, v in outputs.items()}
            self.log_dict(
                renamed_outputs,
                prog_bar=False,
                add_dataloader_idx=False,
            )
        elif dataloader_idx == 1:
            [self.val_batches[k].append(v) for k, v in outputs.items()]

    def on_validation_epoch_end(self):
        if self.trainer.validating and len(self.val_batches["Z_batch"]) > 0:
            Z = torch.vstack(self.val_batches["Z_batch"]).cpu().float().numpy()
            H = torch.vstack(self.val_batches["H_batch"]).cpu().float().numpy()
            labels = (
                torch.hstack(self.val_batches["labels_batch"]).cpu().numpy()
            )

            save_any = (
                self.save_intermediate_feat or self.save_intermediate_bbf
            )
            p = Path(self.logger.log_dir)
            zname = p / "intermediate_emb.zip"
            stepsf = p / "steps.csv"
            if save_any:
                if not stepsf.exists():
                    stepsf.write_text("global_step,epoch\n")
                with stepsf.open("a") as f:
                    f.write(f"{self.global_step},{self.current_epoch}\n")

                with zipfile.ZipFile(zname, "a") as zipf:

                    if "labels.npy" not in zipf.namelist():
                        with zipf.open("labels.npy", "w") as f:
                            np.save(f, labels)

                    if self.save_intermediate_feat:
                        with zipf.open(
                            f"Z/step-{self.global_step:05d}.npy", "w"
                        ) as f:
                            np.save(f, Z)
                    if self.save_intermediate_bbf:
                        with zipf.open(
                            f"H/step-{self.global_step:05d}.npy", "w"
                        ) as f:
                            np.save(f, H)

            if self.eval_ann:
                for name, X in dict(Z=Z, H=H).items():
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        labels,
                        test_size=0.15,
                        random_state=self.rng.integers(2**32),
                    )
                    # import time

                    # t0 = time.time()
                    ann = AnnoyClassifier(15)
                    acc = ann.fit(X_train, y_train).score(X_test, y_test)
                    # t1 = time.time()
                    self.log(f"ann({name})", acc, prog_bar=True)
                    # self.log(f"t_{name}", t1 - t0)

            if callable(self.eval_function):
                edict = self.eval_function(
                    Z=Z,
                    H=H,
                    labels=labels,
                    step=self.global_step,
                    epoch=self.current_epoch,
                )
                self.log_dict(edict, add_dataloader_idx=False, prog_bar=True)

    def on_predict_epoch_start(self):
        self.dim_mask = self.dim_mask_schedule[-1]

    # def test_step(self, batch, batch_idx): ...

    def forward(self, batch):
        batch = self.fix_ffcv_batch(batch)
        x, y = batch
        f, bb = self.model(x)
        return f[:, self.dim_mask], bb

    def on_fit_end(self):
        train_dataset = self.trainer.datamodule.train_dataloader()
        self.save_hyperparameters(dict(n_batches=len(train_dataset)))

    @staticmethod
    def lr_from_batchsize(batch_size: int, /, mode="lin-bs") -> float:
        if mode == "lin-bs":
            lr = 0.03 * batch_size / 256
        elif mode == "sqrt-bs":
            lr = 0.075 * batch_size**0.5
        else:
            raise ValueError(
                f"Unknown mode for calculating the lr ({mode = !r})"
            )
        return lr


class AnnoyClassifier:
    def __init__(
        self, n_neighbors, n_trees=20, metric="euclidean", random_state=None
    ):
        self.n_neighbors = n_neighbors
        self.n_trees = n_trees
        self.metric = metric
        self.random_state = random_state

        if random_state is None:
            self.rng = np.random.default_rng()
            self.seed = self.rng.integers(-(2**31), 2**31)
        elif isinstance(random_state, np.random.RandomState):
            self.seed = self.random_state.randint(-(2**31), 2**31)
        elif isinstance(random_state, np.random.Generator):
            self.seed = self.random_state.integers(-(2**31), 2**31)

    def fit(self, X, y):
        self.y = y

        metric = self.metric if self.metric != "cosine" else "angular"
        self.nn = nn = AnnoyIndex(X.shape[1], metric)
        nn.set_seed(self.seed)

        [nn.add_item(i, x) for i, x in enumerate(X)]
        nn.build(self.n_trees)

        return self

    def predict(self, X):
        nn_ixs = [
            self.nn.get_nns_by_vector(x, self.n_neighbors + 1)[1:] for x in X
        ]
        preds, _counts = stats.mode(self.y[nn_ixs], axis=1, keepdims=False)
        return preds

    def score(self, X, y):
        preds = self.predict(X)
        return (preds == y).mean()


class TSimCNE:
    """The main entry point for fitting tSimCNE on a dataset.

    This class implements the algorithm described in Böhm et
    al. (`ICLR 2023 <https://openreview.net/forum?id=nI2HmVA0hvt>`__).
    It learns a model that will map image data points to 2D, allowing
    an entire dataset to be visualized at once in the form of each
    datum represented as a dot in the Cartesian plane.

    See also :ref:`parameter-guide` for a detailed explanation with
    examples of a selection of the parameters.

    :param None model: The model to train.  By default it will be
        constructed from the two parameters `backbone` and
        `projection_head`.

    :param "infonce" loss: The (contrastive) loss to use.  Default is
        ``"infonce"`` and currently the only supported one.  For
        alternatives, see Damrich et al. (`ICLR 2023
        <https://openreview.net/forum?id=B8a1FcY0vi>`__).

    :param None metric: The metric that is used to calculate the
        similarity.  Defaults to Euclidean metric (with the Cauchy
        kernel).  Another option is ``"cosine"`` to get the default
        SimCLR loss.

    :param "resnet18_simclr" backbone: Backbone to use for the
        contrastive model.  Defaults to ResNet18 as defined in the
        original SimCLR paper (so with a smaller kernel size).  Other
        options are ``"resnet50"``, etc. or simply pass in a torch
        model directly.

    :param "mlp" projection_head: The projection head that maps from
        the backbone features down to the ``"out_dim"``.  Also accepts
        a torch model.

        The activation function is a ReLU.  By default a multilayer
        perceptron with one hidden layer going from 512 (output
        dim. of ResNet18) → 1024 → 128.  The last layer is the output
        dimension during the first training stage, afterwards the
        model will be mutated in-place to then map 512 → 1024 → 2.

        Note that if the output dimension of the backbone was changed,
        then this needs to be appropriately reflected in the
        projection head as well.

    :param None | ``"is_included"`` | torchvision.transforms |
        ffcv.transforms data_transform: The data augmentations to
        create the differing views of the input.  By default it will
        use the same augmentations as written in Böhm et al. (2023);
        random cropping, greyscaling, color jitter, horizontal flips.
        This parameter should be changed with care.

        If ``data_transform="is_included"``, then it is assumed that
        all of the data augmentations are already included within the
        dataset object ``X`` (which is passed to `tsimcne.fit`) and
        will return the augmented samples along with a (dummy) label.

    :param [1000, 50, 450] total_epochs: A list of the number of
        epochs per training stage.  The ratio between the stages
        should be roughly preserved and it should also be exactly
        three.  You can also pass a single integer, which will then
        only fit the first stage.

    :param 512 batch_size: The number of images in one batch.  Note
        that this parameter should be set as high as the memory of the
        GPU allows, as contrastive learning benefits from larger batch
        sizes.  For each image in the batch two views will be
        generated, so by default the batch size will be ``2 * 512 =
        1024``.

    :param 2 out_dim: The number of output dimensions.  For the
        purpose of viusalization you should leave this as 2 (the
        default).  But tSimCNE can also map into an arbitrary number
        of dimensions (so it could also be used to plot a dataset in
        3D, for example).


    :param 128 pretrain_out_dim: The number of output dimensions
        during pretraining (the first stage).

    :param "sgd" optimizer: The optimizer to use.  Currently only
        ``"sgd"`` is allowed.

    :param "cos_annealing" lr_scheduler: The learning rate scheduler
        to use.  Currently only ``"cos_annealing"`` is allowed.

    :param "auto_batch" lr: The learning rate to use.  By default it
        uses a learning rate adapted to the batch size (as well as the
        training stage).

    :param "auto" warmup: The number of warmup epochs.  By default it
        will do 10 epochs of warmup (linearly interpolating from 0 to
        the initial learning rate) if the number of epochs is at least
        100, otherwise it will be 0 warmup epochs.

    :param "only_linear" freeze_schedule: The behavior for
        freezing/unfreezing the network during the different
        optimization stages.  Only change this, if you know what will
        happen to the model.  For now, only the default is allowed.

    :param tuple image_size: The size of the images in the dataset.
        If not passed will be attempted to be inferred from the
        dataset.  Required if ``use_ffcv=True`` (as the dataset will
        need to point to the beton file string and the size
        information cannot be inferred from that).

    :param 1 devices: The number of devices/accelerators to use (with
        the PL Trainer).  Will be passed on as is.  Consider this
        parameter experimental, the effects of using multiple GPUs is
        not entirely clear (but it should probably be safe to do so).

        Currently, the learning rate is not adjusted to account for
        multiple devices, please do so yourself; see the `PL
        documentation
        <https://lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html>`__
        about it.

    :param dict | None trainer_kwargs: The keyword arguments to pass
        to the Trainer, to use during training.  By default the keys
        ``gradient_clip_val=4`` and
        ``gradient_clip_algorithm="value"`` will be set, but can be
        overridden by passing in a custom dict.  The values will be
        set regardless of whether you pass in a ``dict`` or not, so if
        you want to disable gradient clipping you need to override the
        values.

    :param int=8 num_workers: The number of workers for creating the
        dataloader.  Will be passed to the pytorch DataLoader
        constructor.

    :param dict dl_kwargs: Additional keyword arguments that will be
        passed on to the pytorch dataloader constructor.

    :param bool | "auto" use_ffcv: Whether to use the ffcv-ssl library
        to load the data from disk.  If set to ``"auto"`` (default) it
        will check if the supplied argument is a filepath (to a .beton
        file) and set it to ``True``, otherwise it will be ``False``.

    :param str="medium" float32_matmul_precision: The precision to set
        for ``torch.set_float32_matmul_precision``.  By default it
        will be set to ``"medium"``.  Set to ``False`` to leave the
        default.  (This is mostly as a convenience to silence the
        warning that will otherwise be shown if the value is unset.)

    """

    def __init__(
        self,
        model=None,
        loss="infonce",
        metric=None,
        backbone="resnet18_sm_kernel",
        projection_head="mlp",
        data_transform=None,
        total_epochs=[1000],
        batch_size=512,
        out_dim=128,
        anneal_to_dim=2,
        optimizer="sgd",
        lr_scheduler="cos_annealing",
        lr="auto_batch",
        warmup="auto",
        freeze_schedule="only_linear",
        image_size=None,
        devices=1,
        trainer_kwargs=None,
        num_workers=8,
        dl_kwargs=None,
        float32_matmul_precision="medium",
    ):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.backbone = backbone
        self.projection_head = projection_head
        self.data_transform = data_transform
        self.out_dim = out_dim
        self.anneal_to_dim = anneal_to_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.lr = lr
        self.warmup = warmup
        self.freeze_schedule = freeze_schedule
        self.image_size = image_size
        self.devices = devices
        self.trainer_kwargs = trainer_kwargs
        self.num_workers = num_workers
        self.dl_kwargs = dict() if dl_kwargs is None else dl_kwargs
        self.float32_matmul_precision = float32_matmul_precision

        self._handle_parameters()

        if self.float32_matmul_precision:
            torch.set_float32_matmul_precision(self.float32_matmul_precision)

    def _handle_parameters(self):
        if isinstance(self.total_epochs, list):
            self.epoch_schedule = self.total_epochs
        elif isinstance(self.total_epochs, int):
            self.epoch_schedule = [self.total_epochs]

        self.n_stages = len(self.epoch_schedule)
        n_stages = self.n_stages

        if self.lr == "auto_batch":
            lr = self.lr_from_batchsize(self.batch_size)
            self.learning_rates = [lr, lr, lr / 1000][:n_stages]
        elif isinstance(self.lr, list):
            self.learning_rates = self.lr
        elif isinstance(self.lr, (float, int)):
            self.learning_rates = [self.lr]
        else:
            raise ValueError(
                'Expected "auto_batch" or a list of learning rates '
                f" but got {self.lr = !r}."
            )

        if self.n_stages != len(self.learning_rates):
            raise ValueError(
                f"Got {self.total_epochs} for total epochs, but "
                f"{self.learning_rates} for learning rates "
                f"(due to {self.lr = !r})."
            )

        if self.warmup == "auto":
            self.warmup_schedules = [10, 0, 10][:n_stages]
        elif isinstance(self.warmup, list):
            self.warmup_schedules = self.warmup
        else:
            raise ValueError(
                'Expected "auto" or a list of warmup epochs '
                f"but got {self.warmup = !r}."
            )

        if len(self.warmup_schedules) != self.n_stages:
            raise ValueError(
                f"Number of warmup epochs (got {len(self.warmup_schedules)}) "
                "needs to match "
                f"number of learning rates (got {len(self.learning_rates)})."
            )

        if self.freeze_schedule != "only_linear":
            raise ValueError(
                "Only 'only_linear' is supported as freeze_schedule, "
                f"but got {self.freeze_schedule}."
            )

        if self.devices != 1 and self.lr == "auto_batch":
            import warnings

            warnings.warn(
                "devices is not 1, but the learning rate has not been adjusted."
                "  Please see https://"
                "lightning.ai/docs/pytorch/stable/accelerators/gpu_faq.html "
                "for how to set the learning rate when using multiple devices"
            )

        trainer_kwargs = dict(
            gradient_clip_val=4, gradient_clip_algorithm="value"
        )
        if self.trainer_kwargs is None:
            self.trainer_kwargs = trainer_kwargs
        else:
            self.trainer_kwargs = trainer_kwargs.update(self.trainer_kwargs)

    @staticmethod
    def check_ffcv(use_ffcv):
        if use_ffcv:
            try:
                import ffcv

                ffcv.transforms.RandomGrayscale
            except ModuleNotFoundError:
                raise ValueError(
                    "`use_ffcv` is not False, but `ffcv` is not installed. "
                    "Install https://github.com/facebookresearch/FFCV-SSL"
                )
            except AttributeError:
                raise ValueError(
                    "`use_ffcv` is True, but wrong ffcv library is installed. "
                    "Install https://github.com/facebookresearch/FFCV-SSL"
                )

    def fit_transform(
        self,
        X: torch.utils.data.Dataset | str,
        data_transform=None,
        return_labels: bool = False,
        return_backbone_feat: bool = False,
    ):
        """Learn the mapping from the dataset to 2D and return it.

        :param X: The image dataset to be used for training.  Will be
            wrapped into a data loader automatically.  If
            ``use_ffcv=True``, then it needs to be a string pointing
            to the .beton file.

        :param data_transform: the data transformation to use for
            calculating the final 2D embedding.  By default it will
            not perform any data augmentation (as this is only
            relevant during training).

        :param False return_labels: Whether to return the labels that are
            part of the dataset.

        :param False return_backbone_feat: Whether to return the
            high-dimensional features of the backbone.

        """
        self.fit(X)
        return self.transform(
            X,
            data_transform=data_transform,
            return_labels=return_labels,
            return_backbone_feat=return_backbone_feat,
        )

    def fit(self, X: torch.utils.data.Dataset | str | Path):
        """Learn the mapping from the dataset ``X`` to 2D.

        :param X: The image dataset to be used for training.  Will be
            wrapped into a data loader automatically.  If
            ``use_ffcv=True``, then it needs to be a string pointing
            to the .beton file.

        """

        if self.use_ffcv == "auto":
            if isinstance(X, (str, Path)):
                self.use_ffcv = True
            else:
                self.use_ffcv = False
        self.check_ffcv(self.use_ffcv)

        train_dl = self.make_dataloader(X, True, self.data_transform)

        self.loader = train_dl
        it = zip(
            self.epoch_schedule, self.learning_rates, self.warmup_schedules
        )
        self.models = []
        self.trainers = []
        for n_stage, (n_epochs, lr, warmup_epochs) in enumerate(it):
            train_kwargs = dict(
                n_epochs=n_epochs,
                batch_size=self.batch_size,
                optimizer_name=self.optimizer,
                lr_scheduler_name=self.lr_scheduler,
                lr=lr,
                warmup=warmup_epochs,
                pretrain_out_dim=self.pretrain_out_dim,
                out_dim=self.out_dim,
                use_ffcv=self.use_ffcv,
            )
            if n_stage == 0:
                # initialize the model
                plmodel = PLtSimCNE(
                    model=self.model,
                    loss=self.loss,
                    metric=self.metric,
                    backbone=self.backbone,
                    projection_head=self.projection_head,
                    **train_kwargs,
                )

            elif n_stage == 1:
                # modify the model to map down to `out_dim` (2) dimensions
                p = plmodel
                model = mutate_model(
                    p.model,
                    change="lastlin",
                    freeze=True,
                    out_dim=self.out_dim,
                )
                plmodel = PLtSimCNE(
                    model=model,
                    loss=p.loss,
                    metric=p.metric,
                    **train_kwargs,
                )

            elif n_stage == 2:
                model = mutate_model(plmodel.model, freeze=False)
                plmodel = PLtSimCNE(
                    model=model,
                    loss=p.loss,
                    metric=p.metric,
                    **train_kwargs,
                )
            trainer = lightning.Trainer(
                max_epochs=n_epochs,
                devices=self.devices,
                **self.trainer_kwargs,
            )
            trainer.fit(model=plmodel, train_dataloaders=train_dl)
            self.models.append(plmodel)
            self.trainers.append(trainer)

        self.plmodel = plmodel
        self.trainer = trainer
        return self

    def transform(
        self,
        X: torch.utils.data.Dataset,
        data_transform=None,
        return_labels: bool = False,
        return_backbone_feat: bool = False,
    ):
        """Perform the 2D transform on the dataset, using the trained model.
        :param X: The image dataset to be used for transformation.  Will be
            wrapped into a data loader automatically.  If
            ``use_ffcv=True``, then it needs to be a string pointing
            to the .beton file.
        :param data_transform: the data transformation to use for
            calculating the final 2D embedding.  By default it will
            not perform any data augmentation (as this is only
            relevant during training).
        :param False return_labels: Whether to return the labels that are
            part of the dataset.
        :param False return_backbone_feat: Whether to return the
            high-dimensional features of the backbone.
        """
        loader = self.make_dataloader(X, False, data_transform)
        trainer = lightning.Trainer(devices=1)
        pred_batches = trainer.predict(self.plmodel, loader)
        Y = torch.vstack([x[0] for x in pred_batches]).numpy()
        backbone_features = torch.vstack([x[1] for x in pred_batches])

        if return_labels and return_backbone_feat:
            labels = torch.hstack([lbl for _, lbl in loader])
            return Y, labels, backbone_features
        elif not return_labels and return_backbone_feat:
            return Y, backbone_features
        elif return_labels and not return_backbone_feat:
            # XXX: this for some reason changes the labels; but I
            # don't know what causes this!
            labels = torch.hstack([lbl for _, lbl in loader])
            return Y, labels
        else:
            return Y

    @staticmethod
    def get_image_size_from_dataset(X):
        if isinstance(X, (str, Path)):
            raise ValueError(
                "Dataset X is a path, but self.image_size is None. "
                "The parameter is required, image size cannot be "
                "inferred from X this way."
            )
        x0 = X[0]
        if hasattr(x0, "__len__") and len(x0) == 2:
            sample_img, _lbl = x0
        else:
            sample_img = x0
        if not isinstance(sample_img, PIL.Image.Image):
            raise ValueError(
                "The dataset does not return PIL images, "
                f"got {type(sample_img)} instead."
            )
        else:
            return sample_img.size

    @staticmethod
    def get_data_transform(train_or_test, image_size, use_ffcv):
        # data augmentations for contrastive training
        if train_or_test:
            data_transform = get_transforms_unnormalized(
                size=image_size, setting="contrastive", use_ffcv=use_ffcv
            )
        else:
            data_transform = get_transforms_unnormalized(
                size=image_size, setting="none", use_ffcv=use_ffcv
            )
        return data_transform

    def make_dataloader(self, X, train_or_test, data_transform):
        if self.image_size is None:
            self.image_size = self.get_image_size_from_dataset(X)

        if data_transform is None:
            if (
                self.data_transform == "is_included"
            ):  # Keep "is_included" status if that was set prior and no data_transform was passed
                data_transform = self.data_transform
            else:
                data_transform = self.get_data_transform(
                    train_or_test, self.image_size, self.use_ffcv
                )

        if not self.use_ffcv:
            if data_transform != "is_included":
                # dataset that returns two augmented views of a given
                # datapoint (and label)
                dataset_contrastive = TransformedPairDataset(X, data_transform)
            else:
                dataset_contrastive = X

            # wrap dataset into dataloader
            loader = torch.utils.data.DataLoader(
                dataset_contrastive,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=train_or_test,
                **self.dl_kwargs,
            )
        else:
            import ffcv
            from ffcv.fields.basics import IntDecoder

            if train_or_test:
                self.data_transform2 = get_transforms_unnormalized(
                    size=self.image_size,
                    setting="contrastive",
                    use_ffcv=self.use_ffcv,
                )

                self.label_pipeline = [
                    IntDecoder(),
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.Squeeze(),
                ]

                pipelines = {
                    "image": data_transform,
                    "image_0": self.data_transform2,
                    "label": self.label_pipeline,
                }

                custom_field_mapper = {"image_0": "image"}
                order = ffcv.loader.OrderOption.QUASI_RANDOM

                loader = ffcv.Loader(
                    X,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    order=order,
                    os_cache=False,
                    drop_last=False,
                    pipelines=pipelines,
                    custom_field_mapper=custom_field_mapper,
                )
            else:
                self.label_pipeline = [
                    ffcv.fields.basics.IntDecoder(),
                    ffcv.transforms.ToTensor(),
                    ffcv.transforms.Squeeze(),
                ]

                pipelines = {
                    "image": data_transform,
                    "label": self.label_pipeline,
                }

                order = ffcv.loader.OrderOption.SEQUENTIAL

                loader = ffcv.Loader(
                    X,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    order=order,
                    os_cache=False,
                    drop_last=False,
                    pipelines=pipelines,
                )
        return loader

    @staticmethod
    def lr_from_batchsize(batch_size: int, /, mode="lin-bs") -> float:
        if mode == "lin-bs":
            lr = 0.03 * batch_size / 256
        elif mode == "sqrt-bs":
            lr = 0.075 * batch_size**0.5
        else:
            raise ValueError(
                f"Unknown mode for calculating the lr ({mode = !r})"
            )
        return lr


class DummyLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i], 0
