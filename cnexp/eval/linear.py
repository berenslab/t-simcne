from sklearn.linear_model import SGDClassifier

from . import EvalBase


def linear_acc(
    X_train, X_test, y_train, y_test, loss="log", n_jobs=-1, **kwargs
):
    lin = SGDClassifier(loss=loss, n_jobs=n_jobs, **kwargs)

    lin.fit(X_train, y_train)
    return lin.score(X_test, y_test)


class LinearAcc(EvalBase):
    def compute(self):
        self.acc = linear_acc(*self.data_split, **self.kwargs)
