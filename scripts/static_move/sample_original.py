import os
import argparse

import pandas as pd

from chesswinnerprediction.static_move.processing.utils import process_df
from chesswinnerprediction.constants import INTERIM_STATIC_MOVE, PROCESSED_STATIC_MOVE
from chesswinnerprediction.static_move.constants import (
    TRAIN_VALID_TEST,
    I_MOVE_THRESHOLD,
    DEFAULT_N_SAMPLES,
)


def main(dir_name, train_valid_test_samples):
    random_state = 42

    interim_dir_path = str(os.path.join(INTERIM_STATIC_MOVE, dir_name))
    processed_dir_path = str(os.path.join(PROCESSED_STATIC_MOVE, dir_name))
    if not os.path.exists(processed_dir_path):
        os.mkdir(processed_dir_path)

    if train_valid_test_samples is None:
        train_valid_test_samples = DEFAULT_N_SAMPLES

    for csv_file, n, i_move_threshold in zip(
        TRAIN_VALID_TEST, train_valid_test_samples, I_MOVE_THRESHOLD
    ):
        print(f"Processing {csv_file}...")
        df = pd.read_csv(os.path.join(interim_dir_path, csv_file))
        df = process_df(df, n, random_state, i_move_threshold)
        df.to_csv(os.path.join(processed_dir_path, csv_file), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process chess PGN files into train/valid/test datasets."
    )
    parser.add_argument(
        "dir_name",
        help=f"Directory name from {INTERIM_STATIC_MOVE}, containing the chess data files.",
    )
    parser.add_argument(
        "--train_valid_test_samples",
        nargs=3,
        type=int,
        default=[60_000, 25_000, None],
        help="Number of samples for train, valid, and test datasets. Use 'None' to include all remaining data in set.",
    )
    args = parser.parse_args()

    try:
        main(args.dir_name, args.train_valid_test_samples)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
