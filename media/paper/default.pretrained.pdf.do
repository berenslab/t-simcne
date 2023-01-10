#!/usr/bin/env python

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openTSNE
import torch
import torchvision
from cnexp import plot, redo
from cnexp.plot.scalebar import add_scalebar_frac
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from torchvision import transforms
from trimap import TRIMAP
from umap import UMAP


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, i):
        x, lbl = self.data[i]
        return self.transform(x), lbl

    def __len__(self):
        return len(self.data)


def main():

    root = Path("../../experiments/")
    # prefix = root / sys.argv[2] / "dl"
    stylef = "../project.mplstyle"

    device = "cuda:0"
    model = (
        torchvision.models.resnet18(weights="IMAGENET1K_V1").to(device).eval()
    )

    loaders = []
    for dataname in ["cifar", "cifar100"]:
        prefix = root / dataname / "dl"
        redo.redo_ifchange(prefix / "dataset.pt")
        data_sd = torch.load(prefix / "dataset.pt")
        data = data_sd["full_plain"].dataset

        dataset = TransformDataset(data)

        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1024, num_workers=8
        )
        loaders.append(loader)

    def model_transform(x):
        x = model.conv1(x)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        x = model.layer4(x)
        x = model.avgpool(x)
        return x

    # model_transform = model

    with plt.style.context(stylef):
        fig, axxs = plt.subplots(2, 3, figsize=(5.5, 3))
        for axs, loader, name in zip(axxs, loaders, ["CIFAR-10", "CIFAR-100"]):
            with torch.no_grad():
                acts = [model_transform(x.to(device)) for x, lbl in loader]

            ar = np.vstack([a.squeeze().cpu().numpy() for a in acts])
            labels = np.hstack([lbl for x, lbl in loader])
            tsne = openTSNE.TSNE(n_jobs=-1, random_state=98878)
            Y_tsne = tsne.fit(ar)
            print("tsne", file=sys.stderr)
            umap = UMAP(n_jobs=-1, random_state=14141)
            Y_umap = umap.fit_transform(ar)
            print("umap", file=sys.stderr)
            trimap = TRIMAP()
            Y_trimap = trimap.fit_transform(ar)
            print("trimap", file=sys.stderr)

            for ax, Y, title in zip(
                axs, [Y_tsne, Y_umap, Y_trimap], ["t-SNE", "UMAP", "TriMap"]
            ):
                ax.scatter(
                    Y[:, 0], Y[:, 1], c=labels, alpha=0.5, rasterized=True
                )
                ax.set_title(f"{title} {name}")

                add_scalebar_frac(ax)
                knn = KNeighborsClassifier(15)
                X_train, X_test, y_train, y_test = train_test_split(
                    Y, labels, test_size=10_000, random_state=11
                )
                knn.fit(X_train, y_train)
                acc = knn.score(X_test, y_test)
                acctxt = f"$k$nn = {acc:.0%}"
                ax.set_title(acctxt, loc="right", fontsize="small")

        plot.add_letters(axxs)
    metadata = plot.get_default_metadata()
    metadata[
        "Title"
    ] = f"Visualization of a pretrained ResNet18 for {sys.argv[2]}"
    fig.savefig(sys.argv[3], format="pdf", metadata=metadata)


if __name__ == "__main__":
    main()
