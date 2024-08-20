import os
import warnings
from ast import literal_eval

import pandas as pd
import numpy as np

from chesswinnerprediction.static_move.constants import STATIC_MOVE_COLUMNS
from chesswinnerprediction.constants import DRAW_STR, WHITE_WIN_STR, BLACK_WIN_STR, INTERIM_STATIC_MOVE_SPLIT_BALANCED, INTERIM_STATIC_MOVE_SPLIT_ORIGINAL


def sample_data(data, random_state, n=None, weights=None):
    if n is None:
        n = (data["Result"] == DRAW_STR).sum()
    else:
        n //= 3
        n_min = (data["Result"] == DRAW_STR).sum()
        if n > n_min:
            warnings.warn(
                f"Number of samples {n // 3} is bigger than number of draws. "
                f"Number of samples will be: {n_min * 3}."
            )
            n = n_min

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

    group.loc[0, "black_remaining_time"] = group.loc[0, "white_remaining_time"]
    return group[["white_remaining_time", "black_remaining_time"]]


def transform_data(data, file_path):
    data["GameDurations"] = data["GameDurations"].apply(literal_eval)
    data["evaluations_list"] = data["evaluations_list"].apply(literal_eval)
    data["times_in_second"] = data["times_in_second"].apply(literal_eval)

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

    data[["white_remaining_time", "black_remaining_time"]] = (
        data[["white_remaining_time", "black_remaining_time"]].ffill()
    )

    data["time_diff"] = data["white_remaining_time"] - data["black_remaining_time"]
    norm_columns = ["time_diff", "white_remaining_time", "black_remaining_time", "GameDurations"]
    new_columns = ["time_norm_diff", "white_remaining_time_norm", "black_remaining_time_norm", "GameDurations"]

    data[new_columns] = data[norm_columns].div(data["BaseTime"], axis=0)

    # data["white_remaining_time_norm"] = data["white_remaining_time"] / data["BaseTime"]
    # data["black_remaining_time_norm"] = data["black_remaining_time"] / data["BaseTime"]
    # data["time_norm_diff"] = data["white_remaining_time_norm"] - data["black_remaining_time_norm"]
    # data["GameDurations"] /= data["BaseTime"]

    data = data[STATIC_MOVE_COLUMNS]
    data.to_csv(file_path, index=False)


def compute_sample_weights(data, threshold):
    data = data[data["i_move"] <= threshold]

    move_counts = data["i_move"].value_counts()
    inverse_freq = 1 / move_counts
    probs = inverse_freq / inverse_freq.sum()
    weights = data["i_move"].map(probs)
    return weights


# def process_data(
#     data_path,
#     n_draw_games=24000,
#     n_valid_games=3000,
#     n_test_games=3000,
#     n_train_draw=30000,
#     n_valid_draws=10000,
#     n_test_draw=10000,
# ):
#     threshold = 120
#     random_state = 42
#     np.random.seed(random_state)
#
#     data = pd.read_csv(data_path)
#     data["GameId"] = data.index
#
#     # todo move 'mean_base_time' to scripts/process_data_csv.py
#     data["mean_base_time"] = data.groupby("Event")["BaseTime"].transform("mean")
#
#     data = sample_data(data, n_draw_games, random_state)
#     game_ids = data["GameId"].copy().values
#     np.random.shuffle(game_ids)
#
#     train_split_idx = len(game_ids) - n_valid_games - n_test_games
#     valid_split_idx = len(game_ids) - n_test_games
#     train_ids, valid_ids, test_ids = np.split(
#         game_ids, indices_or_sections=[train_split_idx, valid_split_idx]
#     )
#
#     train_data = transform_data(data[data["GameId"].isin(train_ids)].copy())
#     valid_data = transform_data(data[data["GameId"].isin(valid_ids)].copy())
#     test_data = transform_data(data[data["GameId"].isin(test_ids)].copy())
#
#     train_data = sample_data(
#         train_data,
#         n_train_draw,
#         random_state,
#         compute_sample_weights(train_data, threshold),
#     )
#     valid_data = sample_data(
#         valid_data,
#         n_valid_draws,
#         random_state,
#         compute_sample_weights(valid_data, threshold),
#     )
#     test_data = sample_data(
#         test_data,
#         n_test_draw,
#         random_state,
#         compute_sample_weights(test_data, threshold),
#     )
#
#     return train_data, valid_data, test_data


def sample_games(data, n, random_state, balanced=False):
    if balanced:
        n //= 3
        draws = data[data["Result"] == DRAW_STR].sample(n=n, random_state=random_state)
        black_wins = data[data["Result"] == BLACK_WIN_STR].sample(n=n, random_state=random_state)
        white_wins = data[data["Result"] == WHITE_WIN_STR].sample(n=n, random_state=random_state)
        games = pd.concat([draws, white_wins, black_wins]).reset_index(drop=True)
    else:
        games = data.sample(n=n, random_state=random_state)
    return games


def get_not_balanced_train_valid(used_ids, data, random_state):
    train_data = data[~data["GameId"].isin(used_ids)].sample(n=66000, random_state=random_state)
    used_ids = pd.concat([used_ids, train_data["GameId"]])

    valid_data = data[~data["GameId"].isin(used_ids)].sample(n=9000, random_state=random_state)
    return train_data, valid_data


def get_balanced_train_valid(used_ids, data, random_state):
    train_data = sample_data(data[~data["GameId"].isin(used_ids)], random_state, n=66000)

    warnings.warn("Balanced Valid set will take values from the balanced train set.")
    valid_data = sample_data(train_data, random_state, n=9000)
    train_data = train_data[~train_data["GameId"].isin(valid_data["GameId"])]
    return train_data, valid_data


def explode_data(
    interim_df: pd.DataFrame,
    # balanced_train=False,
    # balance_valid=False,
):
    """
    Test set will be the same distribution as the original data.
    It will be split first. Other sets will be sampled from the remaining data.

    For valid and train sets there are 2 scenarios:
    1) balance_train is False && balance_valid is False:
        - train set will be sampled from the remaining data after test set is split.
        - valid set will be sampled from the remaining data after test and train sets are split.

    2) balance_train is True && balance_valid is True:
        - balanced train set will be sampled from the remaining data after test set is split.
        - valid set will be sampled from the balanced train set.
            Train set will be resampled according to the valid set size.

    """
    random_state = 42
    np.random.seed(random_state)

    data = interim_df.copy()

    data["GameId"] = data.index

    test_data = data.sample(n=10000, random_state=random_state)
    used_ids = test_data["GameId"]

    original_train_data, original_valid_data = get_not_balanced_train_valid(used_ids, data, random_state)
    used_ids = pd.concat([used_ids, original_train_data["GameId"], original_valid_data["GameId"]])

    balanced_train_data, balanced_valid_data = get_balanced_train_valid(used_ids, data, random_state)
    balanced_train_data = balanced_train_data.sample(frac=1).reset_index(drop=True)
    balanced_valid_data = balanced_valid_data.sample(frac=1).reset_index(drop=True)

    # save
    print("Saving test")
    test_data_path = os.path.join(INTERIM_STATIC_MOVE_SPLIT_ORIGINAL, "test.csv")
    transform_data(test_data, test_data_path)

    print("Saving original train")
    original_train_data_path = os.path.join(INTERIM_STATIC_MOVE_SPLIT_ORIGINAL, "train.csv")
    transform_data(original_train_data, original_train_data_path)

    print("Saving original valid")
    original_valid_data_path = os.path.join(INTERIM_STATIC_MOVE_SPLIT_ORIGINAL, "valid.csv")
    transform_data(original_valid_data, original_valid_data_path)

    print("Saving balanced train")
    balanced_train_data_path = os.path.join(INTERIM_STATIC_MOVE_SPLIT_BALANCED, "train.csv")
    transform_data(balanced_train_data, balanced_train_data_path)

    print("Saving balanced valid")
    balanced_valid_data_path = os.path.join(INTERIM_STATIC_MOVE_SPLIT_BALANCED, "valid.csv")
    transform_data(balanced_valid_data, balanced_valid_data_path)
