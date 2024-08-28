import os

import pandas as pd
import numpy as np

from chesswinnerprediction.constants import INTERIM_STATIC_MOVE, PROCESSED_STATIC_MOVE


def process_df(df: pd.DataFrame, n, random_state, i_move_threshold):
    valid_con = (df["i_move"] == 1) & (
        df["black_remaining_time_norm"] == df["white_remaining_time_norm"]
    )
    df = df[df["GameId"].isin(df[valid_con]["GameId"])]

    df = df[df["i_move"] < i_move_threshold]
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df.drop(columns=["GameId"], inplace=True)

    if n is not None:
        df = df.sample(n=n, random_state=random_state)

    df[["black_remaining_time_norm", "white_remaining_time_norm"]] = df[
        ["black_remaining_time_norm", "white_remaining_time_norm"]
    ].clip(lower=0.0, upper=1.0)
    df["GameDurations_norm"] = df["GameDurations_norm"].clip(lower=0.0, upper=2.0)
    df["time_diff_norm"] = df["time_diff_norm"].clip(lower=-1.0, upper=1.0)
    return df


def main():
    random_state = 42
    dir_name = "lichess_db_standard_rated_2017-05"

    interim_dir_path = os.path.join(INTERIM_STATIC_MOVE, dir_name)
    processed_dir_path = os.path.join(PROCESSED_STATIC_MOVE, dir_name)
    if not os.path.exists(processed_dir_path):
        os.mkdir(processed_dir_path)

    train_valid_test = ("train.csv", "valid.csv", "test.csv")
    i_move_thresholds = (150, 150, 200)
    train_valid_test_samples = (60_000, 20_000, None)

    for csv_file, n, i_move_threshold in zip(
        train_valid_test, train_valid_test_samples, i_move_thresholds
    ):
        df = pd.read_csv(os.path.join(interim_dir_path, csv_file))
        df = process_df(df, n, random_state, i_move_threshold)
        df.to_csv(os.path.join(processed_dir_path, csv_file), index=False)


if __name__ == "__main__":
    main()
