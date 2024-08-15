import os
import argparse

import pandas as pd
import numpy as np

from chesswinnerprediction.constants import (
    STATIC_MOVE_DATA_PATH,
    EXAMPLE_NAME,
    GAME_ID,
    INTERIM_FOLDER_PATH,
    STATIC_MOVE
)


def split_csv(file_path, train_size, valid_size, test_size, random_state):
    np.random.seed(random_state)

    if not os.path.exists(os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE)):
        os.makedirs(os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE))

    csv_file_name = os.path.basename(file_path)
    processed_file_dir, _ = csv_file_name.split(".")
    processed_dir_path = os.path.join(STATIC_MOVE_DATA_PATH, processed_file_dir)

    if not os.path.exists(processed_dir_path):
        os.makedirs(processed_dir_path)
    else:
        raise ValueError(
            f"Processed data already exists at: {processed_dir_path}.\nDelete directory to process again."
        )

    print(f"Reading data from {file_path}")
    df = pd.read_csv(file_path)

    print(
        f"Splitting data into: \nTrain: {train_size}\nValidation: {valid_size}\nTest: {test_size}"
    )

    games = df[GAME_ID].unique()
    valid_test_games = np.random.choice(games, int(len(games) * (valid_size + test_size)), replace=False)
    split_idx = int(len(valid_test_games) * valid_size / (valid_size + test_size))
    valid_games, test_games = np.split(valid_test_games, [split_idx])

    train_df = df[~df[GAME_ID].isin(valid_test_games)]
    valid_df = df[df[GAME_ID].isin(valid_games)]
    test_df = df[df[GAME_ID].isin(test_games)]

    print(f"Saving data to {processed_dir_path}")
    train_df.to_csv(os.path.join(str(processed_dir_path), "train.csv"), index=False)
    valid_df.to_csv(os.path.join(str(processed_dir_path), "valid.csv"), index=False)
    test_df.to_csv(os.path.join(str(processed_dir_path), "test.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a CSV file into train, validation, and test sets."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random state for reproducibility (Default 42)",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE, EXAMPLE_NAME + ".csv"),
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.8,
        help="Train split size (default: 0.8)",
    )
    parser.add_argument(
        "--valid_size", type=float, default=0.1, help="Valid split size (default: 0.1)"
    )
    parser.add_argument(
        "--test_size", type=float, default=0.1, help="Test split size(default: 0.1)"
    )

    args = parser.parse_args()

    try:
        if args.train_size + args.valid_size + args.test_size != 1.0:
            raise ValueError(
                "The sum of train_size, valid_size, and test_size must be 1."
            )

        split_csv(
            args.file_path,
            args.train_size,
            args.valid_size,
            args.test_size,
            args.random_state,
        )
    except ValueError as e:
        print(f"Error: {e}")
