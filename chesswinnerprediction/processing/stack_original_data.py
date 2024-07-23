import os

import pandas as pd
import numpy as np

from tqdm import tqdm


BASELINE_COLUMNS = ["Event", "WhiteElo", "BlackElo", "TimeControl", "Result",]
TARGET_COLUMNS = ["ResultEncoded", "WhiteWin", "BlackWin", "Draw"]

WHITE_WIN = "1-0"
BLACK_WIN = "0-1"
DRAW = "1/2-1/2"

RESULTS = {
    WHITE_WIN: 1,
    BLACK_WIN: 0,
    DRAW: -1
}


def process_elo(data: pd.DataFrame) -> pd.DataFrame:
    data["WhiteElo"] = data["WhiteElo"].astype(np.int16)
    data["BlackElo"] = data["BlackElo"].astype(np.int16)
    data["MeanElo"] = ((data["WhiteElo"] + data["BlackElo"]) / 2).astype(np.float32)
    data["EloDiff"] = (data["WhiteElo"] - data["BlackElo"]).astype(np.int16)
    data["EloDiffAbs"] = data["EloDiff"].abs().astype(np.int16)
    return data


def process_time_control(data: pd.DataFrame) -> pd.DataFrame:
    data[["BaseTime", "IncrementTime"]] = data["TimeControl"].str.split("+", expand=True)
    data["BaseTime"] = data["BaseTime"].astype(np.int16)
    data["IncrementTime"] = data["IncrementTime"].astype(np.int16)
    data["ZeroIncrementTime"] = (data["IncrementTime"] == 0).astype(np.int8)
    return data


def process_result(data: pd.DataFrame) -> pd.DataFrame:
    data["ResultEncoded"] = data["Result"].map(RESULTS)
    data["WhiteWin"] = (data["Result"] == "1-0").astype(np.int8)
    data["BlackWin"] = (data["Result"] == "0-1").astype(np.int8)
    data["Draw"] = (data["Result"] == "1/2-1/2").astype(np.int8)
    return data


def add_external_features(data: pd.DataFrame) -> pd.DataFrame:
    data["DrawEventProb"] = data.groupby("Event")["Draw"].transform("mean")
    data["WhiteWinEventProb"] = data.groupby("Event")["WhiteWin"].transform("mean")
    data["BlackWinEventProb"] = data.groupby("Event")["BlackWin"].transform("mean")
    return data


def drop_columns(data: pd.DataFrame) -> pd.DataFrame:
    # data.drop(columns=["TimeControl"], inplace=True)
    # data.drop(columns=["Result"], inplace=True)
    # data.drop(columns=["Event"], inplace=True)
    return data


def process_data_df(data: pd.DataFrame) -> pd.DataFrame:
    df = data[BASELINE_COLUMNS]
    df = df[df["Result"] != "*"]

    df["Event"] = df["Event"].str.split(" http").str[0]
    df = pd.get_dummies(df, columns=["Event"], dtype=np.int8, prefix="", prefix_sep="")

    df = process_elo(df)
    df = process_time_control(df)
    df = process_result(df)

    df = add_external_features(df)

    df = drop_columns(df)

    return df


def concat_raw_data(dir_path, output_file):
    data_frames = []

    for file_name in tqdm(os.listdir(dir_path)):
        if file_name.endswith(".csv"):
            file_path = os.path.join(dir_path, file_name)
            data = pd.read_csv(file_path)
            data = process_data_df(data)
            data_frames.append(data)

    combined_data = pd.concat(data_frames, ignore_index=True)
    combined_data.to_csv(output_file, index=False)


def main():
    concat_raw_data('/home/tikhon/PycharmProjects/ChessWinnerPrediction/data/raw/lichess_db_standard_rated_2017-03',
                   '/home/tikhon/PycharmProjects/ChessWinnerPrediction/data/processed/lichess_db_standard_rated_2017-03.csv')


if __name__ == "__main__":
    main()
