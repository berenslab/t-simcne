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
