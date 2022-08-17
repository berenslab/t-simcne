import zipfile

import numpy as np
from sklearn.model_selection import train_test_split

from ..base import ProjectBase


def extract_data_from_zip(zipf, layer):
    labelf = zipf.open("labels.npy")
    labels = np.load(labelf)

    embdir = "backbone_embeddings" if layer == "H" else "embeddings"
    embf = zipf.open(f"{embdir}/post.npy")
    embedding = np.load(embf)

    return embedding, labels


class EvalBase(ProjectBase):
    def __init__(
        self, path, random_state=None, layer="H", test_size=10_000, **kwargs
    ):
        super().__init__(path, random_state=random_state)
        self.layer = layer
        self.test_size = test_size
        self.kwargs = kwargs

    def get_deps(self):
        return [self.indir / "intermediates.zip"]

    def load(self):
        self.zipfile = zipfile.ZipFile(self.indir / "intermediates.zip")
        self.data, self.labels = extract_data_from_zip(
            self.zipfile, self.layer
        )

        self.data_split = train_test_split(
            self.data,
            self.labels,
            test_size=self.test_size,
            stratify=self.labels,
            random_state=self.random_state.integers(2**32 - 1),
        )

    def save(self):
        text = bytes(f"{self.acc}\n", encoding="utf8")
        self.save_lambda(
            self.outdir / "score.txt",
            text,
            lambda f, d: f.write(d),
        )
