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


def format_train(kwargs):
    train = "train"
    if kwargs is not None:
        for k, v in kwargs.items():
            train += f":{k}={v}"
    return train


def model_opt_lr(
    out_dim=128,
    n_epochs=1000,
    warmup_epochs=10,
    backbone="resnet18",
    random_state=None,
):
    model = "model"
    if out_dim != 128:
        model += f":{out_dim=}"
    if random_state is not None:
        model += f":{random_state=}"
    if backbone != "resnet18":
        model += f":{backbone=!s}"

    lr = "lrcos"
    if n_epochs != 1000:
        lr += f":{n_epochs=}"
    if warmup_epochs != 10:
        lr += f":{warmup_epochs=}"

    return Path(model) / "sgd" / lr


def default_train(
    *,
    metric="euclidean",
    train_kwargs=None,
    loss="infonce",
    random_state=None,
    **kwargs,
):
    if random_state is not None:
        kwargs["random_state"] = random_state
        if train_kwargs is not None:
            train_kwargs["random_state"] = random_state
        else:
            train_kwargs = dict(random_state=random_state)

    model_etc = model_opt_lr(**kwargs)
    loss = loss if metric == "euclidean" else f"{loss}:{metric=!s}"

    return model_etc / loss / format_train(train_kwargs)


def simclr_train(**kwargs):
    return default_train(**kwargs)


def finetune(
    llin_epochs=50,
    ft_epochs=450,
    ft_lr=1.2e-4,
    ft_loss="infonce",
    freeze_last_stage=0,
    random_state=None,
    train_kwargs=None,
):
    ftmodel = "ftmodel:freeze=1:change=lastlin"
    if random_state is not None:
        ftmodel += f":{random_state=}"
    ftmodel = Path(ftmodel)
    last_linear = (
        ftmodel / f"sgd/lrcos:n_epochs={llin_epochs}:warmup_epochs=0/{ft_loss}"
    )
    if random_state is not None and train_kwargs is not None:
        train_kwargs["random_state"] = random_state
    train = format_train(train_kwargs)

    last_linear_train = last_linear / train

    if freeze_last_stage not in [0, "backbone"]:
        raise ValueError(f"Unknown value for {freeze_last_stage = !r}")
    finetune_train = (
        last_linear_train
        / f"ftmodel:freeze={freeze_last_stage}"
        / f"sgd:lr={ft_lr}/lrcos:n_epochs={ft_epochs}"
        / train
    )
    return finetune_train
