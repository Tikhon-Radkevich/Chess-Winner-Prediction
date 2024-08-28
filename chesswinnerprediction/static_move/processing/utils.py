import os
from ast import literal_eval

import pandas as pd
import numpy as np

from chesswinnerprediction.static_move.constants import STATIC_MOVE_COLUMNS


def assign_times(group: pd.DataFrame) -> pd.DataFrame:
    group = group.reset_index(drop=True)

    group["white_remaining_time"] = np.where(
        group.index % 2 == 0, group["times_in_second"], np.nan
    )
    group["black_remaining_time"] = np.where(
        group.index % 2 == 1, group["times_in_second"], np.nan
    )

    group.at[0, "black_remaining_time"] = group.at[0, "white_remaining_time"]
    return group[["white_remaining_time", "black_remaining_time"]]


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    move_columns = ["GameDurations", "evaluations_list", "times_in_second"]
    data[move_columns] = data[move_columns].map(literal_eval)

    data = data.explode(move_columns).reset_index(drop=True)

    data["i_move"] = data.groupby("GameId").cumcount() + 1
    data["is_checkmate_countdown"] = data["evaluations_list"].str.startswith("#")
    data["eval"] = (
        data["evaluations_list"].str.replace("#", "", regex=False).astype(np.float64)
    )
    data["GameDurations"] = data["GameDurations"].astype(np.float64)
    data["times_in_second"] = data["times_in_second"].astype(np.float64)

    times_df = (
        data.groupby("GameId")
        .apply(assign_times)
        .reset_index(names=["GameId", "i_move"])
    )
    times_df["i_move"] += 1  # Adjust i_move for merge

    data = pd.merge(data, times_df, on=["GameId", "i_move"], how="left")
    time_columns = ["white_remaining_time", "black_remaining_time"]
    data[time_columns] = data[time_columns].ffill()

    norm_columns = [
        "time_diff",
        "white_remaining_time",
        "black_remaining_time",
        "GameDurations",
    ]
    data["time_diff"] = data["white_remaining_time"] - data["black_remaining_time"]
    data[[f"{col}_norm" for col in norm_columns]] = data[norm_columns].div(
        data["BaseTime"], axis=0
    )

    return data[STATIC_MOVE_COLUMNS]


def get_test_train_valid_game_ids(
    data: pd.DataFrame, random_state: int, n_test: int, n_train: int, n_valid: int
):
    test_ids = data.sample(n=n_test, random_state=random_state)["GameId"]
    remaining_data = data[~data["GameId"].isin(test_ids)]

    train_ids = remaining_data.sample(n=n_train, random_state=random_state)["GameId"]
    remaining_data = remaining_data[~remaining_data["GameId"].isin(train_ids)]

    valid_ids = remaining_data.sample(n=n_valid, random_state=random_state)["GameId"]

    return test_ids, train_ids, valid_ids


def explode_and_save_data(
    interim_df: pd.DataFrame,
    path_to_save: str,
    n_test: int,
    n_train: int,
    n_valid: int,
    random_state: int,
):
    np.random.seed(random_state)

    interim_df["GameId"] = interim_df.index

    train_test_valid_ids = get_test_train_valid_game_ids(
        interim_df, random_state, n_test, n_train, n_valid
    )

    combined_ids = pd.concat(train_test_valid_ids)
    transformed_data = transform_data(
        interim_df[interim_df["GameId"].isin(combined_ids)].copy()
    )

    for ids, csv_name in zip(
        train_test_valid_ids, ["train.csv", "test.csv", "valid.csv"]
    ):
        file_path = os.path.join(path_to_save, csv_name)
        df = transformed_data[transformed_data["GameId"].isin(ids)]
        df.to_csv(file_path, index=False)
