from sklearn.linear_model import LogisticRegression

from . import EvalBase


def linear_acc(
    X_train, X_test, y_train, y_test, solver="saga", n_jobs=-1, **kwargs
):
    lin = LogisticRegression(solver=solver, n_jobs=n_jobs, **kwargs)

    lin.fit(X_train, y_train)
    return lin.score(X_test, y_test)


class LinearAcc(EvalBase):
    def compute(self):
        seed = self.random_state.integers(2**32 - 1)
        self.linear_classifier_seed = seed

        self.acc = linear_acc(
            *self.data_split, random_state=seed, **self.kwargs
        )
