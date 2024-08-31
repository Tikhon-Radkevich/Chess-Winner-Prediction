import os
import argparse

import pandas as pd

from chesswinnerprediction.static_move.processing.utils import explode_and_save_data
from chesswinnerprediction.constants import INTERIM_STATIC_MOVE, INTERIM_FOLDER_PATH


def main(file_path, n_test, n_train, n_valid):
    file_name = os.path.basename(file_path).removesuffix(".csv")
    interim_dir_path = os.path.join(INTERIM_STATIC_MOVE, str(file_name))

    if not os.path.exists(interim_dir_path):
        os.mkdir(interim_dir_path)

    interim_df = pd.read_csv(file_path)

    random_state = 42
    explode_and_save_data(
        interim_df, interim_dir_path, n_test, n_train, n_valid, random_state
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and explode chess PGN data.")
    parser.add_argument(
        "file_path", help=f"Path to the CSV file from {INTERIM_FOLDER_PATH} to process."
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100_000,
        help="Number of samples for the test dataset. Default: 100000",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=100_000,
        help="Number of samples for the train dataset. Default: 100000",
    )
    parser.add_argument(
        "--n_valid",
        type=int,
        default=10_000,
        help="Number of samples for the validation dataset. Default: 10000",
    )
    args = parser.parse_args()

    try:
        main(args.file_path, args.n_test, args.n_train, args.n_valid)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
