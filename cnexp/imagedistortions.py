from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(mean, std, size, setting):
    normalize = transforms.Normalize(mean=mean, std=std)

    if setting == "contrastive":
        return transforms.Compose(
            [
                # transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
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


class TransformedPairDataset(Dataset):
    """Create two augmentations based on one sample from the original dataset.

    This creates a torch dataset that will take one sample from the
    original `dataset` and apply the `transform` to it twice.  In the
    process it discards the label information, but this might be
    subject to change.

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
        orig) where `transformed` is an augmented sample in the form
        (data', data''). `orig` is the original sample and the
        `label`.

    """

    def __init__(self, dataset: Dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        orig_item, label = self.dataset[i]

        item1 = self.transform(orig_item)
        item2 = self.transform(orig_item)

        return (item1, item2), (orig_item, label)
