import os

import pandas as pd
import numpy as np

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from chesswinnerprediction.constants import (
    WHITE_WIN_STR,
    BLACK_WIN_STR,
    DRAW_STR,
    RESULTS_STR_TO_INT,
    BASELINE_COLUMNS,
    MIN_GAME_DURATION
)


def process_elo(data: pd.DataFrame) -> pd.DataFrame:
    data["WhiteElo"] = data["WhiteElo"].astype(np.int16)
    data["BlackElo"] = data["BlackElo"].astype(np.int16)
    data["EloDiff"] = (data["WhiteElo"] - data["BlackElo"]).astype(np.int16)
    return data


def process_time_control(data: pd.DataFrame) -> pd.DataFrame:
    data[["BaseTime", "IncrementTime"]] = data["TimeControl"].str.split("+", expand=True)
    data["BaseTime"] = data["BaseTime"].astype(np.int16)
    data["IncrementTime"] = data["IncrementTime"].astype(np.int16)
    data["ZeroIncrementTime"] = (data["IncrementTime"] == 0).astype(np.int8)
    return data


def process_result(data: pd.DataFrame) -> pd.DataFrame:
    data["ResultEncoded"] = data["Result"].map(RESULTS_STR_TO_INT)
    data["WhiteWin"] = (data["Result"] == WHITE_WIN_STR).astype(np.int8)
    data["BlackWin"] = (data["Result"] == BLACK_WIN_STR).astype(np.int8)
    data["Draw"] = (data["Result"] == DRAW_STR).astype(np.int8)
    return data


def add_external_features(data: pd.DataFrame) -> pd.DataFrame:
    # data["EloDiffAbs"] = data["EloDiff"].abs().astype(np.int16)
    # data["MeanElo"] = ((data["WhiteElo"] + data["BlackElo"]) / 2).astype(np.float32)

    data["DrawEventProb"] = data.groupby("Event")["Draw"].transform("mean")
    data["WhiteWinEventProb"] = data.groupby("Event")["WhiteWin"].transform("mean")
    data["BlackWinEventProb"] = data.groupby("Event")["BlackWin"].transform("mean")
    return data


def drop_data(data: pd.DataFrame) -> pd.DataFrame:
    # data.drop(columns=["TimeControl"], inplace=True)
    # data.drop(columns=["Result"], inplace=True)
    # data.drop(columns=["Event"], inplace=True)

    data = data[data["GameDuration"] > MIN_GAME_DURATION]
    return data


def parse_times_list_to_seconds(time_list_str):
    def time_str_to_seconds(time_str):
        h, m, s = map(int, time_str.split(":"))
        return h * 3600 + m * 60 + s

    # time_list_str[2:-2] - get string without '"[' and ']"' symbols
    time_list = time_list_str[2:-2].split("', '")
    return [time_str_to_seconds(time_str) for time_str in time_list]


def calc_game_duration(time_and_increment):
    time_list, increment_time = time_and_increment.values
    increment = (len(time_list) - 2) * increment_time
    return sum(time_list[:2]) - sum(time_list[-2:]) + increment


def process_moves_time(data: pd.DataFrame) -> pd.DataFrame:
    data["times_in_second"] = data["times_list"].apply(parse_times_list_to_seconds)
    data.drop(columns=["times_list"], inplace=True)

    data["GameDuration"] = data[["times_in_second", "IncrementTime"]].apply(calc_game_duration, axis=1)
    return data


def process_data_df(data: pd.DataFrame) -> pd.DataFrame:
    df = data[BASELINE_COLUMNS]
    df = df[df["Result"] != "*"]

    df["Event"] = df["Event"].str.split(" http").str[0]
    # df = pd.get_dummies(df, columns=["Event"], dtype=np.int8, prefix="", prefix_sep="")

    # header data
    df = process_elo(df)
    df = process_time_control(df)
    df = process_result(df)

    # move data
    df = process_moves_time(df)

    # df = add_external_features(df)

    df = drop_data(df)

    return df


def process_file(file_path):
    data = pd.read_csv(file_path)
    processed_data = process_data_df(data)
    return processed_data


def process_and_concat_raw_data(dir_path, output_file):
    csv_files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith(".csv")]

    combined_data = pd.DataFrame()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in csv_files}

        for future in tqdm(as_completed(futures), total=len(csv_files)):
            processed_data = future.result()
            combined_data = pd.concat([combined_data, processed_data], ignore_index=True)

    print(f"Saving data to {output_file}")
    combined_data.to_csv(output_file, index=False)
