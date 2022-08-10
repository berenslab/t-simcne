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
    elif name == "model":
        from .models.simclr_like import SimCLRModel

        return SimCLRModel
    elif name == "ftmodel":
        from .models.simclr_like import FinetuneSimCLRModel

        return FinetuneSimCLRModel
    else:
        raise ValueError(f"Unknown class identifier “{name}”")


def from_string(path):
    path = Path(path)
    classid, kwargs = split_path(path)

    cls = resolve_class(classid)
    return cls(path, **kwargs)
