import os

import mlflow
import pandas as pd

from sklearn.preprocessing import StandardScaler

from config import MLRUNS_FOLDER_PATH
from chesswinnerprediction.constants import STATIC_MOVE_DATA_PATH
from chesswinnerprediction.static_move.constants import RANDOM_STATE


def get_x_and_y(data):
    x_data = data.drop(columns=["Result"])
    y_data = data["Result"]

    return x_data, y_data


def load_train_valid_test(
    # data_dir="lichess_db_standard_rated_2017-03",
    train_sample_size=0.03,
    valid_sample_size=0.05,
    test_sample_size=1,
    random_state=RANDOM_STATE,
):
    # data_path = os.path.join(STATIC_MOVE_DATA_PATH)

    train_df = pd.read_csv(os.path.join(STATIC_MOVE_DATA_PATH, "train.csv"))
    valid_df = pd.read_csv(os.path.join(STATIC_MOVE_DATA_PATH, "valid.csv"))
    test_df = pd.read_csv(os.path.join(STATIC_MOVE_DATA_PATH, "test.csv"))

    train_df.drop(columns=["Event"], inplace=True)
    valid_df.drop(columns=["Event"], inplace=True)
    test_df.drop(columns=["Event"], inplace=True)

    # std_scaler = StandardScaler()
    # train_data = transform_and_scale_df(train_df, std_scaler)
    # valid_data = transform_and_scale_df(valid_df, std_scaler, fit_scaler=False)
    # test_data = transform_and_scale_df(test_df, std_scaler, fit_scaler=False)

    X_train, y_train = get_x_and_y(train_df)
    X_valid, y_valid = get_x_and_y(valid_df)
    X_test, y_test = get_x_and_y(test_df)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri(MLRUNS_FOLDER_PATH)
    mlflow.set_experiment(experiment_name)
