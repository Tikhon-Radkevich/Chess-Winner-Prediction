import os
import argparse

from chesswinnerprediction.static_move.processing.utils import process_data

from chesswinnerprediction.constants import INTERIM_FOLDER_PATH, EXAMPLE_NAME, STATIC_MOVE


def main(file_path):
    data = process_data(file_path)
    data["i_move"] = data.groupby("GameId").cumcount() + 1

    file_name = os.path.basename(file_path)
    file_path = os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE, file_name)
    data.to_csv(file_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and process csv files.")
    parser.add_argument(
        "file_path",
        nargs="?",
        default=os.path.join(INTERIM_FOLDER_PATH, EXAMPLE_NAME + ".csv"),
        help=f"File name in {INTERIM_FOLDER_PATH} to process. Default: {EXAMPLE_NAME}",
    )
    args = parser.parse_args()

    try:
        main(args.file_path)
    except Exception as e:
        print(f"Error: {e}")
        exit(1)
