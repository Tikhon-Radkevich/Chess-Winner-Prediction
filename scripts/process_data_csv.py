import os
import argparse

from chesswinnerprediction.constants import RAW_FOLDER_PATH, PROCESSED_FOLDER_PATH, EXAMPLE_CSV_DIR
from chesswinnerprediction import process_and_concat_raw_data


def main(dir_name):
    if not os.path.exists(PROCESSED_FOLDER_PATH):
        os.makedirs(PROCESSED_FOLDER_PATH)

    file_name = os.path.basename(dir_name)
    file_path = os.path.join(PROCESSED_FOLDER_PATH, file_name + ".csv")
    process_and_concat_raw_data(dir_name, file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process csv files.")
    parser.add_argument(
        "dir_name",
        nargs="?",
        default=EXAMPLE_CSV_DIR,
        help=f"Directory name in {RAW_FOLDER_PATH} to process. Default: {EXAMPLE_CSV_DIR}",
    )
    args = parser.parse_args()

    try:
        main(args.dir_name)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
