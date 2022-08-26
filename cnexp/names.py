from pathlib import Path


def lin_aug(n_classes="infer", n_epochs=100):
    if n_classes != "infer":
        readout = f"readout:out_dim={n_classes}"
    else:
        readout = "readout"

    return (
        Path(readout)
        / "sgd:lr=30"
        / f"lrcos:n_epochs={n_epochs}:warmup_epochs=0"
        / "ce_loss"
        / "suptrain"
    )


def model_opt_lr(out_dim=128, n_epochs=1000, backbone="resnet18"):
    model = "model" if out_dim == 128 else f"model:{out_dim=}"
    if backbone != "resnet18":
        model += f":{backbone=!s}"
    lr = "lrcos" if n_epochs == 1000 else f"lrcos:{n_epochs}"

    return Path(model) / "sgd" / lr


def default_train(*, metric="cosine", **kwargs):
    model_etc = model_opt_lr(**kwargs)
    loss = "infonce" if metric == "euclidean" else f"infonce:{metric=!s}"

    return model_etc / loss / "train"


def simclr_train(**kwargs):
    return default_train(**kwargs)


def finetune(llin_epochs=50, ft_epochs=450, ft_lr=1.2e-4, freeze_last_stage=0):
    last_linear_train = Path(
        "ftmodel:freeze=1:change=lastlin/sgd/"
        f"lrcos:n_epochs={llin_epochs}:warmup_epochs=0/infonce/train"
    )

    if freeze_last_stage not in [0, "backbone"]:
        raise ValueError(f"Unknown value for {freeze_last_stage = !r}")
    finetune_train = (
        last_linear_train
        / f"ftmodel:freeze={freeze_last_stage}"
        / f"sgd:lr={ft_lr}/lrcos:n_epochs={ft_epochs}/train"
    )
    return finetune_train
