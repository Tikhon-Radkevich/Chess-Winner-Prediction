import mlflow
from mlflow.data.pandas_dataset import from_pandas
import numpy as np
from sklearn import metrics

from chesswinnerprediction.constants import (
    WHITE_WIN_STR,
    BLACK_WIN_STR,
    DRAW_STR
)


class CustomModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model

    def load_context(self, context):
        pass

    def predict(self, _, model_input):
        return self.model.predict(model_input)


class NaiveModel:
    def __init__(self, predict_draws=True):
        self.predict_draws = predict_draws

        self.classes_ = [WHITE_WIN_STR, BLACK_WIN_STR] + ([DRAW_STR] if predict_draws else [])
        self.white_win_pos = 0
        self.black_win_pos = 1

    @staticmethod
    def predict(x):
        """
        Naive prediction: winner will be the player with the highest Elo rating.
        :param x: "WhiteElo" and "BlackElo" must be present in the input data
        :return: y_predict
        """
        white_win = x["WhiteElo"] > x["BlackElo"]
        return np.where(white_win, WHITE_WIN_STR, BLACK_WIN_STR)

    def predict_proba(self, x):
        n_classes = 3 if self.predict_draws else 2
        proba = np.zeros((len(x), n_classes))

        white_win = x["WhiteElo"] > x["BlackElo"]
        proba[white_win, self.white_win_pos] = 1
        proba[~white_win, self.black_win_pos] = 1

        return proba

    def score(self, x, y):
        if not self.predict_draws:
            x, y = x[y != DRAW_STR], y[y != DRAW_STR]

        y_predict = self.predict(x)
        return metrics.balanced_accuracy_score(y, y_predict)

    def mlflow_log(self, x, y):
        mlflow.set_tag("estimator_name", "NaiveModel")
        mlflow.log_param("input_data", x)
        # mlflow.log_param("predict_draws", self.predict_draws)
        mlflow.log_metric("balanced_accuracy_X_test", self.score(x, y))

        # print("Starting to log input data")
        # dataset = from_pandas(x)
        # mlflow.log_input(dataset)

        # print("Starting to log model")
        # model = CustomModelWrapper(self)
        # mlflow.pyfunc.log_model("model", python_model=model)
        # print("Model logged")
