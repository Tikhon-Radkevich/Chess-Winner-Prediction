import os
import argparse

from chesswinnerprediction.static_move.processing.utils import process_data

from chesswinnerprediction.constants import INTERIM_FOLDER_PATH, EXAMPLE_NAME, STATIC_MOVE_DATA_PATH


def main(file_path):
    train, valid, test = process_data(file_path)

    # file_name = os.path.basename(file_path)
    data_dir = os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE_DATA_PATH)

    train.to_csv(os.path.join(str(data_dir), "train.csv"), index=False)
    valid.to_csv(os.path.join(str(data_dir), "valid.csv"), index=False)
    test.to_csv(os.path.join(str(data_dir), "test.csv"), index=False)


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
