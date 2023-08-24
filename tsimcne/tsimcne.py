from copy import deepcopy

import PIL
import torch
from lightning import pytorch as pl

from .imagedistortions import (
    TransformedPairDataset,
    get_transforms_unnormalized,
)
from .losses.infonce import InfoNCECauchy, InfoNCECosine, InfoNCEGaussian
from .lrschedule import CosineAnnealingSchedule
from .models.mutate_model import mutate_model
from .models.simclr_like import make_model


class PLtSimCNE(pl.LightningModule):
    def __init__(
        self,
        model=None,
        loss="infonce",
        metric=None,
        backbone="resnet18",
        projection_head="mlp",
        n_epochs=100,
        batch_size=512,
        out_dim=2,
        optimizer_name="sgd",
        lr_scheduler_name="cos_annealing",
        lr="auto_batch",
        weight_decay=5e-4,
        momentum=0.9,
        warmup="auto",
    ):
        super().__init__()
        self.model = model
        self.loss = loss
        self.metric = metric
        self.backbone = backbone
        self.projection_head = projection_head
        self.n_epochs = n_epochs
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.optimizer_name = optimizer_name
        self.lr_scheduler_name = lr_scheduler_name
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmup = warmup

        self._handle_parameters()

    def _handle_parameters(self):
        if self.model is None:
            self.model = make_model(
                backbone=self.backbone,
                proj_head=self.projection_head,
                out_dim=self.out_dim,
            )

        if self.loss == "infonce":
            if self.metric is None:
                self.metric = "euclidean"

            if self.metric == "euclidean":
                self.loss = InfoNCECauchy()
            elif self.metric == "cosine":
                self.loss = InfoNCECosine()
            elif self.metric == "gauss":
                self.loss = InfoNCEGaussian()
            else:
                raise ValueError(
                    f"Unknown {self.metric = !r} for InfoNCE loss"
                )
        # else: assume that the loss is a proper pytorch loss function

        if self.lr == "auto_batch":
            self.lr = lr_from_batchsize(self.batch_size)

        if self.warmup == "auto":
            self.warmup = 10 if self.max_epochs >= 100 else 0

        if self.optimizer_name != "sgd":
            raise ValueError(
                f"Only 'sgd' is supported as optimizer, got {self.optimizer}."
            )

        if self.lr_scheduler_name != "cos_annealing":
            raise ValueError(
                "Only 'cos_annealing' is supported as learning rate "
                f"scheduler, got {self.lr_scheduler}."
            )

    def configure_optimizers(self):
        opt = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        lrsched = CosineAnnealingSchedule(
            opt, n_epochs=self.n_epochs, warmup_epochs=self.warmup
        )
        return [opt], [
            {"scheduler": lrsched, "interval": "epoch"}
        ]  # interval "step" for batch update

    def training_step(self, batch):
        (i1, i2), _lbl = batch
        samples = torch.vstack((i1, i2))

        features, backbone_features = self.model(samples)
        # backbone_features, _lbl are unused in infonce loss
        loss = self.loss(features, backbone_features, _lbl)

        self.log("train_loss", loss)
        return loss

    def forward(self, batch):
        x, y = batch
        if hasattr(x, "__len__") and len(x) == 2:
            x, _ = x
        return self.model(x)


def tsimcne_transform(
    model: pl.LightningModule,
    X: torch.utils.data.Dataset,
    data_transform=None,
    batch_size=512,
    num_workers=8,
    return_labels: bool = False,
    return_backbone_feat: bool = False,
):
    if data_transform is None:
        x0 = X[0]
        if hasattr(x0, "__len__") and len(x0) == 2:
            sample_img, _lbl = x0
        else:
            sample_img = x0
        if isinstance(sample_img, PIL.Image.Image):
            size = sample_img.size
        else:
            raise ValueError(
                "The dataset does not return PIL images, "
                f"got {type(sample_img)} instead."
            )

        data_transform_none = get_transforms_unnormalized(
            size=size, setting="none"
        )
    else:
        data_transform_none = data_transform

    # dataset that returns two augmented views of a given
    # datapoint (and label)
    dataset_contrastive = TransformedPairDataset(X, data_transform_none)
    # wrap dataset into dataloader
    loader = torch.utils.data.DataLoader(
        dataset_contrastive,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    trainer = pl.Trainer(devices=1)
    pred_batches = trainer.predict(model, loader)
    Y = torch.vstack([x[0] for x in pred_batches]).numpy()
    backbone_features = torch.vstack([x[1] for x in pred_batches])

    if return_labels and return_backbone_feat:
        labels = torch.hstack([lbl for _, lbl in loader])
        return Y, labels, backbone_features
    elif not return_labels and return_backbone_feat:
        return Y, backbone_features
    elif return_labels and not return_backbone_feat:
        labels = torch.hstack([lbl for _, lbl in loader])
        return Y, labels
    else:
        return Y


class TSimCNE:
    def __init__(
        self,
        model=None,
        loss="infonce",
        metric=None,
        backbone="resnet18",
        projection_head="mlp",
        data_transform=None,
        total_epochs=[1000, 50, 450],
        batch_size=512,
        out_dim=2,
        optimizer="sgd",
        lr_scheduler="cos_annealing",
        lr="auto_batch",
        warmup="auto",
        freeze_schedule="only_linear",
        num_workers=8,
    ):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.backbone = backbone
        self.projection_head = projection_head
        self.data_transform = data_transform
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.lr = lr
        self.warmup = warmup
        self.freeze_schedule = freeze_schedule
        self.num_workers = num_workers

        self._handle_parameters()

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

    def fit_transform(
        self,
        X: torch.utils.data.Dataset,
        data_transform=None,
        return_labels: bool = False,
        return_backbone_feat: bool = False,
    ):
        self.fit(X)
        return self.transform(
            X,
            data_transform=data_transform,
            return_labels=return_labels,
            return_backbone_feat=return_backbone_feat,
        )

    def fit(self, X: torch.utils.data.Dataset):
        if self.data_transform is None:
            x0 = X[0]
            if hasattr(x0, "__len__") and len(x0) == 2:
                sample_img, _lbl = x0
            else:
                sample_img = x0
            if isinstance(sample_img, PIL.Image.Image):
                size = sample_img.size
            else:
                raise ValueError(
                    "The dataset does not return PIL images, "
                    f"got {type(sample_img)} instead."
                )

            # data augmentations for contrastive training
            self.data_transform = get_transforms_unnormalized(
                size=size,
                setting="contrastive",
            )

            self.data_transform_none = get_transforms_unnormalized(
                size=size, setting="none"
            )

        # dataset that returns two augmented views of a given
        # datapoint (and label)
        dataset_contrastive = TransformedPairDataset(X, self.data_transform)
        # wrap dataset into dataloader
        train_dl = torch.utils.data.DataLoader(
            dataset_contrastive,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

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
            )
            if n_stage == 0:
                # initialize the model
                plmodel = PLtSimCNE(
                    model=self.model,
                    loss=self.loss,
                    metric=self.metric,
                    backbone=self.backbone,
                    projection_head=self.projection_head,
                    out_dim=128,
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
                    out_dim=self.out_dim,
                    **train_kwargs,
                )

            elif n_stage == 2:
                model = mutate_model(plmodel.model, freeze=False)
                plmodel = PLtSimCNE(
                    model=model,
                    loss=p.loss,
                    metric=p.metric,
                    out_dim=self.out_dim,
                    **train_kwargs,
                )
            trainer = pl.Trainer(max_epochs=n_epochs, devices=1)
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
        return tsimcne_transform(
            self.plmodel,
            X,
            data_transform=data_transform,
            batch_size=self.batch_size,
            return_labels=return_labels,
            return_backbone_feat=return_backbone_feat,
        )

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


class DummyLabelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i], 0
