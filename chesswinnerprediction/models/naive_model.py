import mlflow
import numpy as np

from chesswinnerprediction.constants import (
    WHITE_WIN_STR,
    BLACK_WIN_STR,
    DRAW_STR
)


def estimate_prediction_by_elo(white_elo, black_elo, result, count_draws=True):
    """
    Estimates the percentage of correct predictions based on the Elo rating of the players.

    :param white_elo: "WhiteElo" column from the dataset
    :param black_elo: "BlackElo" column from the dataset
    :param result: "Result" column (with '1-0', '0-1', '1/2-1/2' values)
    :param count_draws: if True, the function will count draws as well
    :return: the percentage of correct predictions
    """
    white_elo_more_than_black = white_elo > black_elo

    white_win_condition = white_elo_more_than_black & (result == WHITE_WIN_STR)
    black_win_condition = ~white_elo_more_than_black & (result == BLACK_WIN_STR)

    right_predictions = np.sum(white_win_condition) + np.sum(black_win_condition)

    if count_draws:
        return right_predictions / len(white_elo)
    return right_predictions / np.sum(result != DRAW_STR)


def log_naive_prediction(input_data, predicted_value, accuracy, predict_draws=True):
    with mlflow.start_run(run_name="Naive Prediction"):
        mlflow.set_tag("estimator_name", "NaivePrediction")
        mlflow.log_param("input_data", input_data)
        mlflow.log_param("predicted_value", predicted_value)
        mlflow.log_param("predict_draws", predict_draws)

        dataset = mlflow.data.from_pandas(input_data.astype(np.float64))
        mlflow.log_input(dataset, context="Eval")
        mlflow.log_metric("accuracy", accuracy)


