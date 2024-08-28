import os

import pandas as pd

from chesswinnerprediction.static_move.processing.utils import explode_and_save_data
from chesswinnerprediction.constants import INTERIM_STATIC_MOVE


def main(file_path):
    file_name = os.path.basename(file_path).removesuffix(".csv")
    interim_dir_path = os.path.join(INTERIM_STATIC_MOVE, str(file_name))

    if not os.path.exists(interim_dir_path):
        os.mkdir(interim_dir_path)

    interim_df = pd.read_csv(file_path)

    random_state = 42
    n_test, n_train, n_valid = 100_00, 100_00, 10_00
    explode_and_save_data(
        interim_df, interim_dir_path, n_test, n_train, n_valid, random_state
    )


if __name__ == "__main__":
    f_path = "/home/tikhon/PycharmProjects/ChessWinnerPrediction/data/interim/lichess_db_standard_rated_2017-05.csv"
    main(f_path)
