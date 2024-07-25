import os
import argparse

from constants import (
    MIN_DATE,
    EXAMPLE_URL,
    EXTERNAL_FOLDER_PATH,
    RAW_FOLDER_PATH,
)
from chesswinnerprediction import download_pgn_zst_file, pgn_zst_to_dataframe


def main(url, split_size):
    date = url.split("_")[-1].replace(".pgn.zst", "")
    if date < MIN_DATE:
        raise ValueError(
            f"Data in {date} does not contain 'clk' and 'eval' data. You must use data from {MIN_DATE} or later!"
        )

    file_path = download_pgn_zst_file(url, EXTERNAL_FOLDER_PATH)
    file_name = os.path.basename(file_path)

    csv_files_dir_name = file_name.replace(".pgn.zst", "")
    csv_files_dir = os.path.join(RAW_FOLDER_PATH, str(csv_files_dir_name))
    if os.path.exists(csv_files_dir):
        raise ValueError(f"To process {file_name} again, delete the directory: {csv_files_dir}")

    os.makedirs(csv_files_dir)

    pgn_zst_to_dataframe(file_path, csv_files_dir, split_size=split_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process chess PGN files.")
    parser.add_argument(
        "url",
        nargs="?",
        default=EXAMPLE_URL,
        help=f"URL from https://database.lichess.org/ of the .pgn.zst file to download. Default: {EXAMPLE_URL}",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=125000,
        help="Size to split the PGN files into. Default: 125000",
    )
    args = parser.parse_args()

    try:
        main(args.url, args.split_size)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
