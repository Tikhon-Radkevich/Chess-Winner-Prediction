import os
import argparse

from constants import (
    MIN_DATE,
    EXAMPLE_URL,
    EXTERNAL_FOLDER_PATH,
    RAW_FOLDER_PATH,
)
from chesswinnerprediction import download_pgn_zst_file
from chesswinnerprediction import pgn_zst_to_dataframe


def main(url):
    date = url.split("_")[-1].replace(".pgn.zst", "")
    if date < MIN_DATE:
        raise ValueError(
            f"Data in {date} does not contain 'clk' and 'eval' data. You must dataloader data from {MIN_DATE} or later!"
        )

    file_path = download_pgn_zst_file(url, EXTERNAL_FOLDER_PATH)
    file_name = os.path.basename(file_path)

    csv_files_dir_name = file_name.replace(".pgn.zst", "")
    csv_files_dir = os.path.join(RAW_FOLDER_PATH, str(csv_files_dir_name))
    if os.path.exists(csv_files_dir):
        raise ValueError(f"To process {file_name} again, delete the directory: {csv_files_dir}")

    os.makedirs(csv_files_dir)

    pgn_zst_to_dataframe(file_path, csv_files_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process chess PGN files.")
    parser.add_argument(
        "url",
        nargs="?",
        default=EXAMPLE_URL,
        help=f"URL from https://database.lichess.org/ of the .pgn.zst file to dataloader. Default: {EXAMPLE_URL}",
    )
    args = parser.parse_args()

    try:
        main(args.url)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
