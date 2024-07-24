import pandas as pd


def estimate_prediction_by_elo(data_df: pd.DataFrame, count_draws=True):
    white_elo_more_than_black = data_df["WhiteElo"] > data_df["BlackElo"]

    white_win_condition = white_elo_more_than_black & (data_df["WhiteWin"] == 1)
    black_win_condition = ~white_elo_more_than_black & (data_df["BlackWin"] == 1)

    right_predictions = len(data_df[white_win_condition]) + len(data_df[black_win_condition])

    if count_draws:
        return right_predictions / len(data_df)
    return right_predictions / len(data_df[data_df["Draw"] != 1])
