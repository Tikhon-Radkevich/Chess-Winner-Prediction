import os

from chesswinnerprediction.config import MIN_DATE, EXAMPLE_URL, EXTERNAL_FOLDER_PATH, RAW_FOLDER_PATH
from download_pgn_zst import download_file
from pgn_zst_to_csv import pgn_zst_to_dataframe


def main():
    url = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-03.pgn.zst"

    date = url.split("_")[-1].replace(".pgn.zst", "")
    if date < MIN_DATE:
        print(f"Data in {date} does not contain 'clk' and 'eval' data. You must download data from {MIN_DATE} or later")
        return

    file_path = download_file(url, EXTERNAL_FOLDER_PATH)

    csv_file_name = file_path.split("/")[-1].replace(".pgn.zst", ".csv")
    csv_file_path = os.path.join(RAW_FOLDER_PATH, csv_file_name)
    if os.path.exists(csv_file_path):
        print(f"File already exists at: {csv_file_path}")
        return

    pgn_zst_to_dataframe(file_path, csv_file_path)


if __name__ == "__main__":
    main()
