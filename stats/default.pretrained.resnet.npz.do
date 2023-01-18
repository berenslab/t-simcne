#!/usr/bin/env python

import sys
import time
from pathlib import Path

import numpy as np
import openTSNE
import torch
import umap
from cnexp import names, redo
from torchvision import transforms
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)


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


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def main():

    root = Path("../experiments/")
    parts = sys.argv[2].split(".")
    if len(parts) == 2:
        dataname, algo = parts
    else:
        (dataname,) = parts
        algo = "tsne"
    prefix = root / dataname / "dl"

    device = "cuda:0"
    modelfuncs = [resnet18, resnet34, resnet50, resnet101, resnet152]
    modelnames = [m.__name__.replace("resnet", "ResNet") for m in modelfuncs]
    models = [m(weights="IMAGENET1K_V1").to(device).eval() for m in modelfuncs]

    prefix = root / dataname / "dl"
    redo.redo_ifchange(prefix / "dataset.pt")
    data_sd = torch.load(prefix / "dataset.pt")
    data = data_sd["full_plain"].dataset

    dataset = TransformDataset(data)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1024, num_workers=8
    )

    def model_transform(x, model):
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

    if dataname != "tiny":
        if dataname == "cifar":
            seed = 3118
        else:
            seed = None
        tsimcne_path = (
            prefix
            / names.default_train(random_state=seed)
            / names.finetune(random_state=seed)
        )
        tsimcne_name = "t-SimCNE"
    else:
        tsimcne_path = (
            root
            / "tiny/dl:num_workers=48/model/sgd/lrcos/infonce:metric=cosine"
            / "train/sgd:lr=0.0012/lrcos:n_epochs=100"
            / "infonce:metric=cosine:reg_coef=1:reg_radius=1/train"
            / "sgd:lr=0.0012/lrcos:n_epochs=400/infonce:temperature=200/train"
            / "ftmodel:freeze=1:change=lastlin/sgd"
            / "lrcos:n_epochs=50:warmup_epochs=0/infonce:temperature=200/train"
            / "ftmodel:freeze=0/sgd:lr=1.2e-05/lrcos:n_epochs=450/train"
        )
        tsimcne_name = "t-SimCNE v2"

    labels = np.hstack([lbl for x, lbl in loader])
    result = dict(labels=labels)
    npz = np.load(tsimcne_path / "intermediates.zip")
    tsimcne = npz["embeddings/post"]
    result[tsimcne_name] = tsimcne

    for model, mname in zip(models, modelnames):
        t0 = time.time()
        with torch.no_grad():
            acts = [
                model_transform(x.to(device), model=model) for x, lbl in loader
            ]
        t1 = time.time()
        eprint(f"{mname} transform {t1 - t0:.1f}s", end="", flush=True)

        ar = np.vstack([a.squeeze().cpu().numpy() for a in acts])
        t0 = time.time()
        if algo == "tsne":
            tsne = openTSNE.TSNE(n_jobs=-1, random_state=98878)
            Y = tsne.fit(ar)
            redname = "t-SNE"
        else:
            umap1 = umap.UMAP(random_state=98878)
            Y = umap1.fit_transform(ar)
            redname = "UMAP"
        t1 = time.time()
        eprint(f", {redname} {t1 - t0: .1f}s")

        result[f"{redname}({mname})"] = Y

    with open(sys.argv[3], "wb") as f:
        np.savez(f, **result)


if __name__ == "__main__":
    main()
