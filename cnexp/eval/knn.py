from sklearn.neighbors import KNeighborsClassifier

from . import EvalBase


def knn_acc(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    metric="euclidean",
    n_jobs=-1,
    **kwargs,
):
    knn = KNeighborsClassifier(
        n_neighbors, metric=metric, n_jobs=n_jobs, **kwargs
    )

    knn.fit(X_train, y_train)
    return knn.score(X_test, y_test)


class KNNAcc(EvalBase):
    def __init__(
        self, path, random_state=None, layer="Z", test_size=10_000, **kwargs
    ):
        super().__init__(path, random_state=random_state)
        self.layer = layer
        self.test_size = test_size
        self.kwargs = kwargs

    def compute(self):
        self.acc = knn_acc(*self.data_split, **self.kwargs)
