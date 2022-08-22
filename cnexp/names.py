from pathlib import Path


def lin_aug(n_classes=10, n_epochs=100):
    return (
        f"readout:out_dim={n_classes}/"
        f"sgd:lr=30/lrcos:n_epochs={n_epochs}:warmup_epochs=0/"
        "ce_loss/suptrain"
    )


def model_opt_lr(out_dim=128, n_epochs=1000):
    model = "model" if out_dim == 128 else f"model:{out_dim=}"
    lr = "lrcos" if n_epochs == 1000 else f"lrcos:{n_epochs}"

    return model / "sgd" / lr


def default_train(*, metric="euclidean", **kwargs):
    model_etc = model_opt_lr(**kwargs)
    loss = "infonce" if metric == "euclidean" else f"infonce:{metric=}"

    return model_etc / loss / "train"


def finetune(llin_epochs=50, ft_epochs=450, freeze_last_stage=0):
    last_linear_train = Path(
        "ftmodel:freeze=1:change=lastlin/sgd/"
        f"lrcos:n_epochs={llin_epochs}:warmup_epochs=0/infonce/train"
    )

    if freeze_last_stage not in [0, "backbone"]:
        raise ValueError(f"Unknown value for {freeze_last_stage = !r}")
    finetune_train = (
        last_linear_train / f"ftmodel:freeze={freeze_last_stage}"
        f"sgd/lrcos:n_epochs={ft_epochs}/train"
    )
    return finetune_train
