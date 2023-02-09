import torch

from .train import TrainBase


def eval_model(model, /, dataloader, device="cuda:0") -> float:
    n_seen: int = 0
    running_acc: float = 1.0

    model.eval()
    for batch in dataloader:
        # we ignore img2 since it's the same as img1.  A bit awkward
        # since the structure of the dataloader is set up for
        # contrastive learning, but here we do not do that.
        (img1, _img2), labels = batch
        img1 = img1.to(device)

        with torch.no_grad():
            features, _backbone_features = model(img1)

        preds = features.argmax(dim=1)
        n_batch = preds.size(0)
        correct = (preds.to(labels.device) == labels).sum().item()

        running_acc = (running_acc * n_seen + correct) / (n_seen + n_batch)
        n_seen += n_batch

    return running_acc


class SupervisedTraining(TrainBase):
    def __init__(
        self, path, model_save_freq=-1, embedding_save_freq=-1, **kwargs
    ):
        super().__init__(
            path,
            model_save_freq=model_save_freq,
            embedding_save_freq=embedding_save_freq,
            **kwargs,
        )
        # this will set up model eval/train modes correctly in `train`.
        self.kwargs.setdefault("readout_mode", True)

    def load(self):
        super().load()

        self.dataloader = self.dataset_dict["train_linear_loader"]
        self.dataloader_test = self.dataset_dict["test_linear_loader"]

    def compute(self):
        super().compute()  # fits the model

        self.kwargs.pop("readout_mode")
        self.acc = eval_model(self.model, self.dataloader_test, **self.kwargs)

    def save(self):
        super().save()

        text = bytes(f"{self.acc}\n", encoding="utf8")
        self.save_lambda(
            self.outdir / "score.txt",
            text,
            lambda f, d: f.write(d),
        )


class SupervisedFullTraining(TrainBase):
    def __init__(
        self, path, model_save_freq=1000, embedding_save_freq=1000, **kwargs
    ):
        super().__init__(
            path,
            model_save_freq=model_save_freq,
            embedding_save_freq=embedding_save_freq,
            **kwargs,
        )

    def load(self):
        super().load()

        self.save_dir.mkdir(exist_ok=True)
        self.dataloader = self.dataset_dict["train_augmented_loader"]
        self.dataloader_test = self.dataset_dict["test_linear_loader"]

    def compute(self):
        super().compute()  # fits the model

        self.acc = eval_model(self.model, self.dataloader_test, **self.kwargs)

    def save(self):
        super().save()

        text = bytes(f"{self.acc}\n", encoding="utf8")
        self.save_lambda(
            self.outdir / "score.txt",
            text,
            lambda f, d: f.write(d),
        )
