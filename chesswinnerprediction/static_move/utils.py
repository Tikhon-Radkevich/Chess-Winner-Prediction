import os

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score

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


def calculate_metrics(model, x, y_true, n_bins, i_move_bins):
    balanced_accuracy_per_bin = {}
    recall_per_bin = {class_label: {} for class_label in ["1-0", "0-1", "1/2-1/2"]}
    precision_per_bin = {class_label: {} for class_label in ["1-0", "0-1", "1/2-1/2"]}

    for bin_id in range(n_bins):
        bin_mask = (i_move_bins == bin_id)
        X_test_filtered = x[bin_mask]
        y_test_filtered = y_true[bin_mask]

        if len(y_test_filtered) == 0:
            continue

        y_pred = model.predict(X_test_filtered)
        balanced_accuracy_per_bin[bin_id] = balanced_accuracy_score(y_test_filtered, y_pred)

        for class_label in recall_per_bin.keys():
            recall_per_bin[class_label][bin_id] = recall_score(
                y_test_filtered, y_pred, labels=[class_label], average="macro"
            )
            precision_per_bin[class_label][bin_id] = precision_score(
                y_test_filtered, y_pred, labels=[class_label], average="macro"
            )
    return balanced_accuracy_per_bin, recall_per_bin, precision_per_bin


def log_metrics_plot(metrics_per_bin, bin_centers, n_bins, title):
    plt.figure(figsize=(12, 6))

    for class_label in metrics_per_bin.keys():
        plt.plot(bin_centers, [metrics_per_bin[class_label].get(i, 0) for i in range(n_bins)], marker="o",
                 linestyle="-", label=f"{class_label}")

    plt.xlabel("i_move (Step Number)")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    # Save the plot as an image file
    image_path = f"{title.lower().replace(' ', '_')}_plot.png"
    plt.savefig(image_path)
    plt.close()

    mlflow.log_artifact(image_path)

    os.remove(image_path)


def log_x_distribution(n_bins, i_move_bins, y_true, bin_centers):
    class_distribution_per_bin = pd.DataFrame(0, index=range(n_bins), columns=["1-0", "0-1", "1/2-1/2"])

    for bin_id in range(n_bins):
        bin_mask = (i_move_bins == bin_id)
        class_counts = y_true[bin_mask].value_counts()

        for class_label in class_counts.index:
            class_distribution_per_bin.loc[bin_id, class_label] = class_counts[class_label]

    plt.figure(figsize=(12, 6))
    bar_width = bin_centers[1] - bin_centers[0]
    bottom_values = np.zeros(n_bins)

    for class_label in class_distribution_per_bin.columns:
        plt.bar(bin_centers, class_distribution_per_bin[class_label], width=bar_width, bottom=bottom_values,
                label=class_label)
        bottom_values += class_distribution_per_bin[class_label]

    title = "Class Distribution vs. i_move (Binned)"
    plt.xlabel("i_move (Step Number)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title="Class")
    plt.grid(True)

    image_path = f"{title.lower().replace(' ', '_')}_plot.png"
    plt.savefig(image_path)
    plt.close()

    mlflow.log_artifact(image_path)

    os.remove(image_path)


def log_prediction(model, x, y_true):
    n_bins = 24

    bin_edges = np.linspace(0, max(x["i_move"]), n_bins + 1)
    i_move_bins = pd.cut(x["i_move"], bins=bin_edges, labels=False, include_lowest=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    balanced_accuracy, recall, precision = calculate_metrics(model, x, y_true, n_bins, i_move_bins)

    recall["Accuracy"] = balanced_accuracy
    precision["Accuracy"] = balanced_accuracy

    log_metrics_plot(recall, bin_centers, n_bins, "Recall")
    log_metrics_plot(precision, bin_centers, n_bins, "Precision")
    log_x_distribution(n_bins, i_move_bins, y_true, bin_centers)
