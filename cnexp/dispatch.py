from pathlib import Path


def split_path(path, *, split_argpairs=":", split_argvals="="):
    s = str(Path(path).resolve().name)
    parts = s.split(split_argpairs)
    name = parts.pop(0)
    kwargs = dict()
    for p in parts:
        try:
            key, val = p.split(split_argvals)
        except ValueError:
            raise ValueError(
                f"malformed parameter “{p}” cannot be split in two"
            )

        # try to convert to either an int or float, otherwise pass the
        # value on as is.
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass

        # special cases for val can be handled here before being
        # passed on to the kwarg dict.

        kwargs[key] = val

    return name, kwargs


def resolve_class(name):
    """map the name to the class object."""

    if name == "cifar":
        from .dataset.cifar import CIFAR10

        return CIFAR10
    elif name == "cifar100":
        from .dataset.cifar import CIFAR100

        return CIFAR100
    elif name == "derma":
        from .dataset.medmnist import DermaMNIST

        return DermaMNIST
    elif name == "tiny":
        from .dataset.tiny_imagenet import TinyImageNet

        return TinyImageNet
    elif name == "dl":
        from .dataset.dataloader import GenericDataLoader as GDL

        return GDL
    elif name == "model":
        from .models.simclr_like import SimCLRModel

        return SimCLRModel
    elif name == "ftmodel":
        from .models.mutate_model import FinetuneSimCLRModel

        return FinetuneSimCLRModel
    elif name == "readout":
        from .models.mutate_model import ReadoutModel

        return ReadoutModel
    elif name == "sgd":
        from .optimizers import SGD

        return SGD
    elif name == "adam":
        from .optimizers import Adam

        return Adam
    elif name == "lrcos":
        from .lrschedule import CosineAnnealing as CA

        return CA
    elif name == "lrlin":
        from .lrschedule import LinearAnnealing as LA

        return LA
    elif name == "infonce":
        from .losses.infonce import InfoNCELoss

        return InfoNCELoss
    elif name == "ce_loss":
        from .losses.supervised import CELoss

        return CELoss
    elif name == "closs":
        from .losses.old_impl import SlowContrastiveLoss as SCL

        return SCL
    elif name == "train":
        from .train import TrainBase

        return TrainBase
    elif name == "suptrain":
        from .suptrain import SupervisedTraining as ST

        return ST
    elif name == "suptrain2":
        from .suptrain import SupervisedFullTraining as SFT

        return SFT
    elif name == "lin":
        from .eval.linear import LinearAcc

        return LinearAcc
    elif name == "knn":
        from .eval.knn import KNNAcc

        return KNNAcc
    elif name == "ann":
        from .eval.ann import ANNAcc

        return ANNAcc
    else:
        raise ValueError(f"Unknown class identifier “{name}”")


def from_string(path):
    path = Path(path)
    classid, kwargs = split_path(path)

    cls = resolve_class(classid)
    return cls(path, **kwargs)
