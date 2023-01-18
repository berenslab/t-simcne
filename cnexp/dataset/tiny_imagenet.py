import contextlib
import hashlib
import inspect
import shutil
import subprocess
import zipfile

from torch.utils.data import ConcatDataset
from torchvision.datasets import ImageFolder

from ..imagedistortions import TransformedPairDataset, get_transforms
from .base import DatasetBase


class TinyImageNet(DatasetBase):
    def get_deps(self):
        supdeps = super().get_deps()
        deps = [
            inspect.getfile(get_transforms),
            inspect.getfile(TransformedPairDataset),
        ]
        return supdeps + deps

    def compute(self):
        with open(self.outdir / "stdout.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                with contextlib.redirect_stderr(f):
                    self.data_sd = load_tiny_imagenet(
                        root=self.outdir, **self.kwargs
                    )


def get_imagenet(root):
    imagenet_dir = root / "tiny-imagenet-200"
    fname = "tiny-imagenet-200.zip"
    h_zip = "6198c8ae015e2b3e007c7841da39ec069199b9aa3bfa943a462022fe5e43c821"

    if imagenet_dir.exists():
        return imagenet_dir
    else:
        if not (root / fname).exists():
            url = f"http://cs231n.stanford.edu/{fname}"
            cmd = ["wget", url, "--output-document", str(root / fname)]
            subprocess.run(cmd, check=True, capture_output=True)
            assert h_zip == sha256_file(
                root / fname
            ), f"Hash does not match: {fname}"
        else:
            file_hash = sha256_file(root / fname)
            if file_hash != h_zip:
                (root / fname).unlink()
                return get_imagenet(root)

        if imagenet_dir.exists():
            shutil.rmtree(imagenet_dir)
        with zipfile.ZipFile(root / fname) as zf:
            zf.extractall(root)

        # add labels for val/
        lbl_map = dict()
        lbldirs = set()
        with (imagenet_dir / "val/val_annotations.txt").open() as f:
            for line in f.readlines():
                imgname, dirname, *_ = line.split("\t")
                lbl_map[imgname] = dirname
                lbldirs.add(dirname)

        [(imagenet_dir / "val" / d).mkdir() for d in lbldirs]
        for imgfile in (imagenet_dir / "val/images").iterdir():
            labeldir = lbl_map[imgfile.name]
            imgfile.rename(imagenet_dir / "val" / labeldir / imgfile.name)
        (imagenet_dir / "val/images").rmdir()

        assert imagenet_dir.exists()
        return imagenet_dir


def load_tiny_imagenet(root, **kwargs):
    # hardcode dataset mean and std
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = (64, 64)

    # make root path independent of cwd so the dataset can be loaded
    # while cwd is in other directories.
    root = root.resolve()

    path = get_imagenet(root)

    transform = get_transforms(mean, std, size=size, setting="contrastive")
    transform_lin_train = get_transforms(
        mean, std, size=size, setting="train_linear_classifier"
    )
    transform_none = get_transforms(
        mean, std, size=size, setting="test_linear_classifier"
    )

    dataset_train = ImageFolder(
        path / "train",
        **kwargs,
    )
    dataset_val = ImageFolder(
        path / "val",
        **kwargs,
    )
    # dataset_test = ImageFolder(
    #     path / "test",
    #     **kwargs,
    # )

    dataset_full = ConcatDataset([dataset_train, dataset_val])

    # need to make a dataset that returns two transforms of an image
    # fmt: off
    T = TransformedPairDataset
    test =                T(dataset_val,  transform_none)
    return dict(
        train_contrastive=T(dataset_full,  transform),
        train_linear     =T(dataset_train, transform_lin_train),
        test_linear      =test,
        train_plain      =T(dataset_train, transform_none),
        test_plain       =test,
        full_plain       =T(dataset_full,  transform_none),
    )
    # fmt: on


def sha256_file(filename, blocksize=2**20):
    m = hashlib.sha256()
    with open(filename, "rb") as f:
        while True:
            buf = f.read(blocksize)
            if not buf:
                break
            m.update(buf)
    return m.hexdigest()
