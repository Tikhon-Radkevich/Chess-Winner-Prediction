import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn import metrics


class CustomKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            score_size=0.1,
            random_state=42,
            class_weight=None,
            **kwargs,  # KNeighborsClassifier parameters
    ):
        self.class_weight = class_weight
        self.score_size = score_size
        self.random_state = random_state
        self.knn = KNeighborsClassifier(**kwargs)

        self.predict_draws = None
        self.coef_ = None
        self.classes_ = None

    def fit(self, x, y, *args, **kwargs):
        if self.class_weight is not None:
            x, y = self.balance_dataset(x, y)

        self.knn.fit(x, y)
        self.classes_ = self.knn.classes_

        return self

    def balance_dataset(self, x, y):
        data_df = pd.concat([x, y], axis=1)
        base = y.value_counts().min()

        balanced_df = pd.concat([
            resample(
                data_df[y == target],
                replace=(base * weight) > y.value_counts()[target],
                n_samples=int(base * weight),
                random_state=self.random_state
            )
            for target, weight in self.class_weight.items()
        ])

        return balanced_df[x.columns], balanced_df[y.name]

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
