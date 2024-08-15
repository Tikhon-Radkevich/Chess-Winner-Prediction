import mlflow

from sklearn import metrics


class StaticMoveBaseModel:
    def __init__(self, estimator):
        self._estimator = estimator
        self._best_scores = {}
        self.balanced_accuracy = None
        self.metric_name = "Balanced Accuracy"

    def fit(self, x, y):
        self._estimator.fit(x, y)

    def predict(self, x):
        return self._estimator.predict(x)

    def log_trial(self):
        mlflow.log_metric(self.metric_name, self.balanced_accuracy)

    def score(self, x, y):
        y_predict = self.predict(x)
        self.balanced_accuracy = metrics.balanced_accuracy_score(y, y_predict)


