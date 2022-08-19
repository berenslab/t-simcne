from annoy import AnnoyIndex
from scipy import stats

from . import EvalBase


def ann_acc(
    X_train,
    X_test,
    y_train,
    y_test,
    n_neighbors=15,
    metric="cosine",
    n_jobs=-1,
    n_trees=20,
    seed=10106122,
    save_tree_filename=None,
    **kwargs
):
    # rename
    metric = metric if metric != "cosine" else "angular"
    nn = AnnoyIndex(X_test.shape[1], metric)
    nn.set_seed(seed)

    [nn.add_item(i, x) for i, x in enumerate(X_train)]
    nn.build(n_trees)

    if save_tree_filename is not None:
        ANNAcc.save_lambda(
            save_tree_filename, nn, lambda f, nn: nn.save(f.name)
        )

    nn_ixs = [nn.get_nns_by_vector(x, n_neighbors) for x in X_test]
    preds, _counts = stats.mode(y_train[nn_ixs], axis=1, keepdims=False)

    return (preds == y_test).mean()


class ANNAcc(EvalBase):
    def compute(self):
        seed = self.random_state.integers(-(2**31), 2**31)
        self.ann_random_state = seed
        self.acc = ann_acc(*self.data_split, seed=seed, **self.kwargs)
