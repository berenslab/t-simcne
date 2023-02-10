from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


def get_transforms(
    mean, std, size, setting, crop_scale_lo=0.2, crop_scale_hi=1
):
    normalize = transforms.Normalize(mean=mean, std=std)

    crop_scale = crop_scale_lo, crop_scale_hi
    if setting == "contrastive":
        return transforms.Compose(
            [
                # transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "train_linear_classifier":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "test_linear_classifier":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unknown transformation setting {setting!r}")


def get_transforms_unnormalized(
    size, setting="contrastive", crop_scale_lo=0.2, crop_scale_hi=1
):
    crop_scale = crop_scale_lo, crop_scale_hi
    if setting == "contrastive":
        return transforms.Compose(
            [
                # transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=size, scale=crop_scale),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
            ]
        )
    elif setting == "train_linear_classifier":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
    elif setting == "none" or setting == "test_linear_classifier":
        return transforms.ToTensor()
    else:
        raise ValueError(f"Unknown transformation setting {setting!r}")


class TransformedPairDataset(Dataset):
    """Create two augmentations based on one sample from the original dataset.

    This creates a torch dataset that will take one sample from the
    original `dataset` and apply the `transform` to it twice and
    return the two resulting items instead.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        A dataset returning a (data, label) pair.
    transform : torchvision.transforms.*
        Transformations that should be applied to the data point sampled from
        the dataset.

    Returns
    -------
    torch.utils.data.Dataset
        a new torch dataset that will return a pair (transformed,
        label) where `transformed` is an augmented sample in the form
        (data', data'').  `label` corresponds to the original label.

    """

    def __init__(self, dataset: Dataset, transform, classes=None):
        self.dataset = dataset
        self.transform = transform

        # try to find the class list
        if classes is None:
            if isinstance(self.dataset, ConcatDataset):
                try:
                    self.classes = self.dataset.datasets[0].classes

                except AttributeError:
                    self.classes = None
            else:
                try:
                    self.classes = self.dataset.classes

                except AttributeError:
                    self.classes = None

        else:
            self.classes = classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        orig_item, label = self.dataset[i]

        item1 = self.transform(orig_item)
        item2 = self.transform(orig_item)

        return (item1, item2), label
