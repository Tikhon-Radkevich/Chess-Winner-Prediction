import ast

import pandas as pd
import numpy as np

from chesswinnerprediction.static_move.constants import STATIC_MOVE_COLUMNS
from chesswinnerprediction.constants import DRAW_STR, WHITE_WIN_STR, BLACK_WIN_STR


def sample_data(data, n, random_state, weights=None):
    draws = data[data["Result"] == DRAW_STR].sample(
        n=n, random_state=random_state, weights=weights
    )
    black_wins = data[data["Result"] == BLACK_WIN_STR].sample(
        n=n, random_state=random_state, weights=weights
    )
    white_wins = data[data["Result"] == WHITE_WIN_STR].sample(
        n=n, random_state=random_state, weights=weights
    )

    return pd.concat([draws, white_wins, black_wins]).reset_index(drop=True)


def assign_times(group):
    group = group.reset_index(drop=True)

    group["white_remaining_time"] = group["times_in_second"].where(group.index % 2 == 0)
    group["black_remaining_time"] = group["times_in_second"].where(group.index % 2 == 1)

    group.loc[0, "black_remaining_time"] = group.loc[0, "black_remaining_time"]
    return group[["white_remaining_time", "black_remaining_time"]]


def transform_data(data):
    data["GameDurations"] = data["GameDurations"].apply(lambda x: ast.literal_eval(x))
    data["evaluations_list"] = data["evaluations_list"].apply(
        lambda x: ast.literal_eval(x)
    )
    data["times_in_second"] = data["times_in_second"].apply(
        lambda x: ast.literal_eval(x)
    )

    data = data.explode(
        ["evaluations_list", "times_in_second", "GameDurations"]
    ).reset_index(drop=True)

    data["i_move"] = data.groupby("GameId").cumcount() + 1

    data["is_checkmate_countdown"] = data["evaluations_list"].str.startswith("#")

    data["eval"] = data["evaluations_list"].str.replace("#", "").astype(np.float64)
    data["GameDurations"] = data["GameDurations"].astype(np.float64)
    data["times_in_second"] = data["times_in_second"].astype(np.float64)

    times_df = (
        data.groupby("GameId")
        .apply(assign_times, include_groups=False)
        .reset_index(names=["GameId", "i_move"])
    )
    times_df["i_move"] += 1

    data = pd.merge(data, times_df, on=["GameId", "i_move"])

    data["white_remaining_time"] = data["white_remaining_time"].ffill()
    data["black_remaining_time"] = data["black_remaining_time"].ffill()

    data["time_diff"] = data["white_remaining_time"] - data["black_remaining_time"]
    data["white_remaining_time_norm"] = data["white_remaining_time"] / data["mean_base_time"]
    data["black_remaining_time_norm"] = data["black_remaining_time"] / data["mean_base_time"]

    data = data[STATIC_MOVE_COLUMNS]
    return data


def compute_sample_weights(data, threshold):
    data = data[data["i_move"] <= threshold]

    move_counts = data["i_move"].value_counts()
    inverse_freq = 1 / move_counts
    probs = inverse_freq / inverse_freq.sum()
    weights = data["i_move"].map(probs)
    return weights


def process_data(
    data_path,
    n_draw_games=24000,
    n_valid_games=3000,
    n_test_games=3000,
    n_train_draw=30000,
    n_valid_draws=10000,
    n_test_draw=10000,
):
    threshold = 120
    random_state = 42
    np.random.seed(random_state)

    data = pd.read_csv(data_path)
    data["GameId"] = data.index

    # todo move 'mean_base_time' to scripts/process_data_csv.py
    data["mean_base_time"] = data.groupby("Event")["BaseTime"].transform("mean")

    data = sample_data(data, n_draw_games, random_state)
    game_ids = data["GameId"].copy().values
    np.random.shuffle(game_ids)

    train_split_idx = len(game_ids) - n_valid_games - n_test_games
    valid_split_idx = len(game_ids) - n_test_games
    train_ids, valid_ids, test_ids = np.split(
        game_ids, indices_or_sections=[train_split_idx, valid_split_idx]
    )

    train_data = transform_data(data[data["GameId"].isin(train_ids)].copy())
    valid_data = transform_data(data[data["GameId"].isin(valid_ids)].copy())
    test_data = transform_data(data[data["GameId"].isin(test_ids)].copy())

    train_data = sample_data(
        train_data,
        n_train_draw,
        random_state,
        compute_sample_weights(train_data, threshold),
    )
    valid_data = sample_data(
        valid_data,
        n_valid_draws,
        random_state,
        compute_sample_weights(valid_data, threshold),
    )
    test_data = sample_data(
        test_data,
        n_test_draw,
        random_state,
        compute_sample_weights(test_data, threshold),
    )

    return train_data, valid_data, test_data
