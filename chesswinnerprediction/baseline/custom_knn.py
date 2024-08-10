import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


class CustomKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            score_size=0.1,
            random_state=42,
            class_weight=None,
            balance_dataset_func=None,
            **kwargs,  # KNeighborsClassifier parameters
    ):
        self.class_weight = class_weight
        self.score_size = score_size
        self.random_state = random_state
        self.balance_dataset_func = balance_dataset_func
        self.knn = KNeighborsClassifier(**kwargs)

        self.predict_draws = None
        self.coef_ = None
        self.classes_ = None

        if class_weight is not None or balance_dataset_func is not None:
            if class_weight is None or balance_dataset_func is None:
                raise ValueError("class_weight and balance_dataset_func must be both None or both not None")

    def fit(self, x, y, *args, **kwargs):
        if self.balance_dataset_func is not None:
            x, y = self.balance_dataset_func(x, y, self.class_weight, self.random_state)

        self.knn.fit(x, y)
        self.classes_ = self.knn.classes_

        return self

    def predict(self, x):
        return self.knn.predict(x)

    def predict_proba(self, x):
        return self.knn.predict_proba(x)

    def score(self, x, y, use_subset=True, **kwargs):
        if use_subset:
            subset_size = int(len(x) * self.score_size)
            indices = np.random.choice(len(x), size=subset_size, replace=False)

            x = x.iloc[indices]
            y = y.iloc[indices]

        return metrics.balanced_accuracy_score(y, self.predict(x))
