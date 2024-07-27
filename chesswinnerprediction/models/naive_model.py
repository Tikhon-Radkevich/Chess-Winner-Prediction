import numpy as np

from chesswinnerprediction.processing.process_and_concat_raw_data import (
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
