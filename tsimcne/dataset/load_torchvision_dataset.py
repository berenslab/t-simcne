from torch.utils.data import ConcatDataset

from ..imagedistortions import TransformedPairDataset, get_transforms


def load_torchvision_dataset(
    dataset_class,
    /,
    mean,
    std,
    size,
    download=True,
    train=None,
    crop_scale_lo=0.2,
    crop_scale_hi=1,
    **kwargs,
):
    transform = get_transforms(
        mean,
        std,
        size=size,
        setting="contrastive",
        crop_scale_lo=crop_scale_lo,
        crop_scale_hi=crop_scale_hi,
    )
    transform_lin_train = get_transforms(
        mean,
        std,
        size=size,
        setting="train_linear_classifier",
        crop_scale_lo=crop_scale_lo,
        crop_scale_hi=crop_scale_hi,
    )
    transform_none = get_transforms(
        mean,
        std,
        size=size,
        setting="test_linear_classifier",
        crop_scale_lo=crop_scale_lo,
        crop_scale_hi=crop_scale_hi,
    )

    dataset_train = dataset_class(
        download=download,
        train=True,
        **kwargs,
    )
    dataset_test = dataset_class(
        download=download,
        train=False,
        **kwargs,
    )
    dataset_full = ConcatDataset([dataset_train, dataset_test])

    # need to make a dataset that returns two transforms of an image
    # fmt: off
    T = TransformedPairDataset
    test =                T(dataset_test,  transform_none)
    return dict(
        train_contrastive=T(dataset_full,  transform),
        train_augmented  =T(dataset_train, transform),
        train_linear     =T(dataset_train, transform_lin_train),
        test_linear      =test,
        train_plain      =T(dataset_train, transform_none),
        test_plain       =test,
        full_plain       =T(dataset_full,  transform_none),
    )
    # fmt: on
