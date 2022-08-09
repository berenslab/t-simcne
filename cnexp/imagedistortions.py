from torch.utils.data import Dataset
from torchvision import transforms


def get_transforms(mean, std, size, setting):
    normalize = transforms.Normalize(mean=mean, std=std)

    if setting == "contrastive":
        return transforms.Compose(
            [
                # transforms.functional.to_pil_image,
                # transforms.RandomRotation(30),
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.functional.to_tensor,
                normalize,
            ]
        )
    elif setting == "train_linear_classifier":
        return transforms.Compose(
            [
                # transforms.functional.to_pil_image,
                transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif setting == "test_linear_classifier":
        return transforms.Compose(
            [
                transforms.functional.to_pil_image,
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
    process it discards the label information, but this might be subject to change.

    """

    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        item, _label = self.dataset[i]

        item1 = self.transform(item)
        item2 = self.transform(item)

        return item1, item2
