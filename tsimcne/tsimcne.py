import PIL
import torch

from .callback import to_features
from .imagedistortions import (
    TransformedPairDataset,
    get_transforms_unnormalized,
)
from .losses.infonce import InfoNCECauchy, InfoNCECosine, InfoNCEGaussian
from .lrschedule import CosineAnnealingSchedule
from .models.mutate_model import mutate_model
from .models.simclr_like import make_model
from .optimizers import lr_from_batchsize, make_sgd
from .train import train


class TSimCNE:
    def __init__(
        self,
        model=None,
        loss="infonce",
        metric=None,
        backbone="resnet18",
        projection_head="mlp",
        mutate_model_inplace=True,
        data_transform=None,
        total_epochs=[1000, 50, 450],
        batch_size=512,
        out_dim=2,
        optimizer="sgd",
        lr_scheduler="cos_annealing",
        lr="auto_batch",
        warmup="auto",
        freeze_schedule="only_linear",
        device="cuda:0",
        num_workers=8,
    ):
        self.model = model
        self.loss = loss
        self.metric = metric
        self.backbone = backbone
        self.projection_head = projection_head
        self.mutate_model_inplace = mutate_model_inplace
        self.data_transform = data_transform
        self.out_dim = out_dim
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.total_epochs = total_epochs
        self.lr = lr
        self.warmup = warmup
        self.freeze_schedule = freeze_schedule
        self.device = device
        self.num_workers = num_workers

        self._handle_parameters()

    def _handle_parameters(self):
        if self.model is None:
            self.model = make_model(
                backbone=self.backbone, proj_head=self.projection_head
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

        if isinstance(self.total_epochs, list):
            self.epoch_schedule = self.total_epochs
        elif isinstance(self.total_epochs, int):
            self.epoch_schedule = [self.total_epochs]

        self.n_stages = len(self.epoch_schedule)
        n_stages = self.n_stages

        if self.lr == "auto_batch":
            lr = lr_from_batchsize(self.batch_size)
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

        if self.optimizer != "sgd":
            raise ValueError(
                f"Only 'sgd' is supported as optimizer, got {self.optimizer}."
            )

        if self.lr_scheduler != "cos_annealing":
            raise ValueError(
                "Only 'cos_annealing' is supported as learning rate "
                f"scheduler, got {self.lr_scheduler}."
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
        if not self.mutate_model_inplace:
            from deepcopy import copy

            self.model = copy(self.model)

        if self.data_transform is None:
            sample_img, _lbl = X[0]
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
        for n_stage, (n_epochs, lr, warmup_epochs) in enumerate(it):
            self._fit_stage(train_dl, n_epochs, lr, warmup_epochs)

            if n_stage == 0:
                mutate_model(
                    self.model,
                    change="lastlin",
                    freeze=True,
                    out_dim=self.out_dim,
                )
            elif n_stage == 1:
                mutate_model(self.model, freeze=False)

    def _fit_stage(
        self,
        X: torch.utils.data.DataLoader,
        n_epochs: int,
        lr: float,
        warmup_epochs: int,
    ):
        if self.optimizer == "sgd":
            self.opt = make_sgd(self.model, lr=lr)

        if self.lr_scheduler == "cos_annealing":
            self.lrsched = CosineAnnealingSchedule(
                self.opt, n_epochs=n_epochs, warmup_epochs=warmup_epochs
            )

        train(
            X,
            self.model,
            self.loss,
            self.opt,
            self.lrsched,
            device=self.device,
        )

    def transform(
        self,
        X: torch.utils.data.Dataset,
        data_transform=None,
        return_labels: bool = False,
        return_backbone_feat: bool = False,
    ):

        if data_transform is not None:
            self.data_transform_none = data_transform

        # dataset that returns two augmented views of a given
        # datapoint (and label)
        dataset_contrastive = TransformedPairDataset(
            X, self.data_transform_none
        )
        # wrap dataset into dataloader
        loader = torch.utils.data.DataLoader(
            dataset_contrastive,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

        Y, backbone_features, labels = to_features(
            self.model, loader, device=self.device
        )

        if return_labels and return_backbone_feat:
            return Y, labels, backbone_features
        elif not return_labels and return_backbone_feat:
            return Y, backbone_features
        elif return_labels and not return_backbone_feat:
            return Y, labels
        else:
            return Y


# def example_test_cifar10():

#     # get the cifar dataset
#     dataset_train = torchvision.datasets.CIFAR10(
#         root="experiments/cifar/out/cifar10",
#         download=True,
#         train=True,
#     )
#     dataset_test = torchvision.datasets.CIFAR10(
#         root="experiments/cifar/out/cifar10",
#         download=True,
#         train=False,
#     )
#     dataset_full = torch.utils.data.ConcatDataset(
#         [dataset_train, dataset_test]
#     )

#     # mean, std, size correspond to dataset
#     mean = (0.4914, 0.4822, 0.4465)
#     std = (0.2023, 0.1994, 0.2010)
#     size = (32, 32)

#     # data augmentations for contrastive training
#     transform = get_transforms(
#         mean,
#         std,
#         size=size,
#         setting="contrastive",
#     )
#     # transform_none just normalizes the sample
#     transform_none = get_transforms(
#         mean,
#         std,
#         size=size,
#         setting="test_linear_classifier",
#     )

#     # datasets that return two augmented views of a given datapoint (and label)
#     dataset_contrastive = TransformedPairDataset(dataset_train, transform)
#     dataset_visualize = TransformedPairDataset(dataset_full, transform_none)

#     # wrap dataset into dataloader
#     train_dl = torch.utils.data.DataLoader(
#         dataset_contrastive, batch_size=1024, shuffle=True
#     )
#     orig_dl = torch.utils.data.DataLoader(
#         dataset_visualize, batch_size=1024, shuffle=False
#     )

#     # create the object
#     tsimcne = TSimCNE(total_epochs=[3, 2, 2])
#     # train on the augmented/contrastive dataloader (this takes the most time)
#     tsimcne.fit(train_dl)
#     # fit the original images
#     Y, labels = tsimcne.transform(orig_dl, return_labels=True)

#     return Y, labels
