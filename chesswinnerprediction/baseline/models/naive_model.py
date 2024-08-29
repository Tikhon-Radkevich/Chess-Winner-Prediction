import mlflow
import numpy as np
import pandas as pd

from sklearn import metrics

from chesswinnerprediction.constants import WHITE_WIN_STR, BLACK_WIN_STR, DRAW_STR


class NaiveModel:
    def __init__(self, predict_draws: bool = True) -> None:
        self.predict_draws = predict_draws

        self.classes_ = [WHITE_WIN_STR, BLACK_WIN_STR] + (
            [DRAW_STR] if predict_draws else []
        )
        self.white_win_pos = 0
        self.black_win_pos = 1

    @staticmethod
    def predict(x: pd.DataFrame) -> np.ndarray:
        """
        Naive prediction: winner will be the player with the highest Elo rating.
        :param x: "WhiteElo" and "BlackElo" must be present in the input data
        :return: y_predict
        """
        white_win = x["WhiteElo"] > x["BlackElo"]
        return np.where(white_win, WHITE_WIN_STR, BLACK_WIN_STR)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n_classes = 3 if self.predict_draws else 2
        proba = np.zeros((len(x), n_classes))

        white_win = x["WhiteElo"] > x["BlackElo"]
        proba[white_win, self.white_win_pos] = 1
        proba[~white_win, self.black_win_pos] = 1

        return proba

    def score(self, x: pd.DataFrame, y: pd.Series) -> float:
        if not self.predict_draws:
            x, y = x[y != DRAW_STR], y[y != DRAW_STR]

        y_predict = self.predict(x)
        return metrics.balanced_accuracy_score(y, y_predict)

    def log_loss(self, x: pd.DataFrame, y: pd.Series) -> float:
        if not self.predict_draws:
            x, y = x[y != DRAW_STR], y[y != DRAW_STR]

        proba = self.predict_proba(x)
        return metrics.log_loss(y, proba)

    def mlflow_log(self, x: pd.DataFrame, y: pd.Series) -> None:
        mlflow.set_tag("estimator_name", "NaiveModel")
        mlflow.log_param("input_data", x)
        mlflow.log_metric("balanced_accuracy_X_test", self.score(x, y))
