import os
from ast import literal_eval

import pandas as pd
import numpy as np
import chess

from chesswinnerprediction.static_move.constants import (
    STATIC_MOVE_COLUMNS,
    PIECE_VALUES,
)


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
    move_columns = [
        "GameDurations",
        "evaluations_list",
        "times_in_second",
        "chess_moves_list",
    ]
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

    # interim_df["GameId"] = interim_df.index

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


def process_game(moves_pgn):
    board = chess.Board()
    num_moves = len(moves_pgn)

    white_scores = np.empty(num_moves, dtype=np.int16)
    black_scores = np.empty(num_moves, dtype=np.int16)
    n_pieces = np.empty(num_moves, dtype=np.int16)

    # Initialize material scores at the start (39 is the sum of all non-king pieces)
    # todo pawn can reach the end of the board and became another piece; but I manually set the start score to 39.
    # this thing leads to negative score values.
    white_score, black_score = 39, 39
    n_pieces_on_board = 32

    for i, move_san in enumerate(moves_pgn):
        move = board.parse_san(move_san)
        captured_piece = board.piece_at(
            move.to_square
        )  # Check the destination square before the move

        # If there's a captured piece, update values
        if captured_piece is not None:
            n_pieces_on_board -= 1

            piece_value = PIECE_VALUES[captured_piece.piece_type]
            if captured_piece.color == chess.WHITE:
                white_score -= piece_value
            else:
                black_score -= piece_value

        # Push the move to update the board state
        board.push(move)

        # Store the current values after the move
        n_pieces[i] = n_pieces_on_board
        white_scores[i] = white_score
        black_scores[i] = black_score

    return white_scores, black_scores, n_pieces


def process_all_games(df):
    results = []

    for game_id, group in df.groupby("GameId"):
        white_scores, black_scores, n_pieces = process_game(
            group["chess_moves_list"].tolist()
        )

        game_results = pd.DataFrame(
            {
                "GameId": game_id,
                "i_move": np.arange(1, len(white_scores) + 1),
                "w_score": white_scores,
                "b_score": black_scores,
                "n_pieces": n_pieces,
            }
        )

        results.append(game_results)

    games_df = pd.concat(results, ignore_index=True)
    # todo: fix process_game func, remove clip.
    games_df[["w_score", "b_score"]] = games_df[["w_score", "b_score"]].clip(lower=0)
    return games_df


def process_moves_pgn(df):
    pieces_data = process_all_games(df)
    df = pd.merge(df, pieces_data, on=["GameId", "i_move"], how="left")
    return df


def add_time_features(df: pd.DataFrame):
    epsilon = 1e-6

    df["white_time_per_move"] = (1 - df["white_remaining_time_norm"]) / df[
        "i_move"
    ] + epsilon
    df["black_time_per_move"] = (1 - df["black_remaining_time_norm"]) / df[
        "i_move"
    ] + epsilon

    df["white_increment_pct_in_time_per_move"] = (
        df["IncrementTime"] / df["BaseTime"]
    ) / df["white_time_per_move"]
    df["black_increment_pct_in_time_per_move"] = (
        df["IncrementTime"] / df["BaseTime"]
    ) / df["black_time_per_move"]

    df["white_increment_pct_in_time_per_move"] = df[
        "white_increment_pct_in_time_per_move"
    ].clip(lower=0.0, upper=1.0)
    df["black_increment_pct_in_time_per_move"] = df[
        "black_increment_pct_in_time_per_move"
    ].clip(lower=0.0, upper=1.0)

    df["white_time_will_end_on_move"] = (
        df["white_remaining_time_norm"] / df["white_time_per_move"]
    )
    df["black_time_will_end_on_move"] = (
        df["black_remaining_time_norm"] / df["black_time_per_move"]
    )

    df["white_time_will_end_on_move"] = df["white_time_will_end_on_move"].clip(
        lower=0, upper=200
    )
    df["black_time_will_end_on_move"] = df["black_time_will_end_on_move"].clip(
        lower=0, upper=200
    )

    return df


def process_df(df: pd.DataFrame, n=None, random_state=42) -> pd.DataFrame:
    valid_con = (
        (df["i_move"] == 1)
        & (df["black_remaining_time_norm"] == df["white_remaining_time_norm"])
        & (df["black_remaining_time_norm"] == 1.0)
    )
    df = df[df["GameId"].isin(df[valid_con]["GameId"])]

    df = process_moves_pgn(df)
    df.drop(columns=["chess_moves_list"], inplace=True)

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

    df = add_time_features(df)
    return df
