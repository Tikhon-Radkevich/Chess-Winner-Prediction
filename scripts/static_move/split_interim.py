import os

import pandas as pd

from chesswinnerprediction.static_move.processing.utils import explode_data
from chesswinnerprediction.constants import (
    INTERIM_STATIC_MOVE_SPLIT_BALANCED,
    INTERIM_STATIC_MOVE_SPLIT_ORIGINAL,
    INTERIM_FOLDER_PATH
)


def shaffle_train_valid_test(train, valid, test):
    train = train.sample(frac=1).reset_index(drop=True)
    valid = valid.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    return train, valid, test


def save_train_valid_test(train, valid, test, data_dir):
    train.to_csv(os.path.join(data_dir, "train.csv"), index=False)
    valid.to_csv(os.path.join(data_dir, "valid.csv"), index=False)
    test.to_csv(os.path.join(data_dir, "test.csv"), index=False)


def main(file_path):
    interim_df = pd.read_csv(file_path)

    explode_data(interim_df)
    # train, valid, test = shaffle_train_valid_test(train, valid, test)
    # print("Saving original split")
    # save_train_valid_test(train, valid, test, INTERIM_STATIC_MOVE_SPLIT_ORIGINAL)
    #
    # train, valid, test = explode_data(interim_df, balanced_train=True, balance_valid=True)
    # train, valid, test = shaffle_train_valid_test(train, valid, test)
    # print("Saving balanced split")
    # save_train_valid_test(train, valid, test, INTERIM_STATIC_MOVE_SPLIT_BALANCED)


if __name__ == "__main__":
    f_path = "/data/interim/lichess_db_standard_rated_2017-05.csv"
    main(f_path)
