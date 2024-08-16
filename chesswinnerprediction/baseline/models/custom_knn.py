import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn import metrics


class CustomKNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        score_size: float = 0.1,
        random_state: int = 42,
        class_weight: dict[any, float] | None = None,
        **kwargs,  # Parameters for KNeighborsClassifier
    ):
        """
        Custom KNN Classifier that handles imbalanced classes by weighting
        them according to the provided class_weight dictionary.

        Parameters
        ----------
        score_size : float, default=0.1
            Score subset size.
            This subset is used to calculate the balanced accuracy score.

        random_state : int, default=42
            Seed used by the random number generator for reproducibility of results.

        class_weight : dict, default=None
            Dictionary containing class labels as keys and their corresponding
            weights as values. This is used to handle imbalanced datasets by
            oversampling or under-sampling the classes according to the weights provided.

        **kwargs : keyword arguments
            Parameters to pass to the underlying `KNeighborsClassifier`.

        Attributes
        ----------
        knn : KNeighborsClassifier
            The underlying KNeighborsClassifier instance used for prediction.

        classes_ : array, shape (n_classes,)
            Array of class labels.

        coef_ : None
            Need to estimate model with utils.

        predict_draws : None
            Need to estimate model with utils.
        """
        self.class_weight = class_weight
        self.score_size = score_size
        self.random_state = random_state
        self.knn = KNeighborsClassifier(**kwargs)

        # need to estimate model
        self.predict_draws = None
        self.coef_ = None
        self.classes_ = None

    def fit(
        self, X: pd.DataFrame, y: pd.Series, *args, **kwargs
    ) -> "CustomKNNClassifier":
        """
        Fit the CustomKNNClassifier model.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The input samples. Each row represents a sample, and each column represents a feature.

        y : pd.Series of shape (n_samples,)
            Target values (class labels). Must be of the same length as X.

        Returns
        -------
        self : object
            Returns the instance of the CustomKNNClassifier.
        """
        if self.class_weight is not None:
            x, y = self.balance_dataset(X, y)

        self.knn.fit(X, y)
        self.classes_ = self.knn.classes_

        return self

    def balance_dataset(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Balance the dataset by resampling based on the specified class weights.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The features of the dataset.

        y : pd.Series of shape (n_samples,)
            The target labels for the dataset. Each entry corresponds to a class label for the respective sample.

        Returns
        -------
        tuple of (pd.DataFrame, pd.Series)
            A tuple containing:

            - X : pd.DataFrame - The resampled features of the dataset.

            - y : pd.Series - The resampled target labels with balanced class distribution.
        """
        data_df = pd.concat([X, y], axis=1)
        base = y.value_counts().min()

        balanced_df = pd.concat(
            [
                resample(
                    data_df[y == target],
                    replace=(base * weight) > y.value_counts()[target],
                    n_samples=int(base * weight),
                    random_state=self.random_state,
                )
                for target, weight in self.class_weight.items()
            ]
        )

        return balanced_df[X.columns], balanced_df[y.name]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the given data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The input samples for which class labels are to be predicted.

        Returns
        -------
        y : np.ndarray of shape (n_samples, )
            The predicted class labels for the input samples.
        """
        return self.knn.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The input samples for which class probabilities are to be predicted.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            The predicted class probabilities for the input samples.
        """
        return self.knn.predict_proba(X)

    def score(
        self, X: pd.DataFrame, y: pd.Series, use_subset: bool = True, **kwargs
    ) -> float:
        """
        Compute the balanced accuracy score of the model.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The features of the dataset to be used for scoring.

        y : pd.Series of shape (n_samples,)
            The true labels corresponding to the features.

        use_subset : bool, default=True
            Whether to use a random subset of the data for scoring.
            If True, a subset of the data is used based on the proportion defined by `score_size`

        **kwargs : keyword arguments, optional
            Additional arguments passed to the scoring method.

        Returns
        -------
        score : float
            The balanced accuracy score of the model.
        """
        if use_subset:
            subset_size = int(len(X) * self.score_size)
            indices = np.random.choice(len(X), size=subset_size, replace=False)

            X = X.iloc[indices]
            y = y.iloc[indices]

        return metrics.balanced_accuracy_score(y, self.predict(X))
