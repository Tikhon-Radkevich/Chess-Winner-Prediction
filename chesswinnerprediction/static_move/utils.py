import os
import shutil

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import recall_score, precision_score, balanced_accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import MLRUNS_FOLDER_PATH
from chesswinnerprediction.constants import PROCESSED_STATIC_MOVE
from chesswinnerprediction.static_move.constants import RANDOM_STATE


def get_x_and_y(data):
    x_data = data.drop(columns=["Result"])
    y_data = data["Result"]

    return x_data, y_data


def get_i_move_to_result_bins(data, n_bins):
    bin_labels = [f"{i}_bin" for i in range(n_bins)]
    i_move_bin = pd.cut(data["i_move"], bins=n_bins, labels=bin_labels, include_lowest=True).astype(str)
    return i_move_bin

    # i_move_bin_to_result = i_move_bin + "_" + data["Result"]
    # return i_move_bin_to_result


def scale_df(data, columns, std_scaler, fit_scaler=True):
    if fit_scaler:
        data[columns] = std_scaler.fit_transform(data[columns])
        return data

    data[columns] = std_scaler.transform(data[columns])
    return data


def load_train_valid_test(
        data_dir="lichess_db_standard_rated_2017-05",
        random_state=RANDOM_STATE,
        drop_event=True,
):
    # data_path = os.path.join(STATIC_MOVE_DATA_PATH)
    n_bins = 3

    train_df = pd.read_csv(os.path.join(PROCESSED_STATIC_MOVE, data_dir, "train.csv"))
    valid_df = pd.read_csv(os.path.join(PROCESSED_STATIC_MOVE, data_dir, "valid.csv"))
    test_df = pd.read_csv(os.path.join(PROCESSED_STATIC_MOVE, data_dir, "test.csv"))

    if drop_event:
        train_df.drop(columns=["Event"], inplace=True)
        valid_df.drop(columns=["Event"], inplace=True)
        test_df.drop(columns=["Event"], inplace=True)

    i_move_bin_to_result = get_i_move_to_result_bins(train_df, n_bins)
    train_df["sample_weight"] = compute_sample_weight("balanced", i_move_bin_to_result)

    i_move_bin_to_result_valid = get_i_move_to_result_bins(valid_df, n_bins)
    valid_df["sample_weight"] = compute_sample_weight("balanced", i_move_bin_to_result_valid)

    # train_df.drop(columns=["GameId"], inplace=True)
    # valid_df.drop(columns=["GameId"], inplace=True)
    # test_df.drop(columns=["GameId"], inplace=True)

    # valid_df = add_i_move_to_result_bins(valid_df, n_bins)
    # test_df = add_i_move_to_result_bins(test_df, n_bins)

    # columns_to_scale = [
    #     "eval",
    #     "EloDiff",
    #     "MeanElo",
    #     "BaseTime",
    #     "IncrementTime",
    # ]
    # std_scaler = StandardScaler()
    # train_data = scale_df(train_df, std_scaler)
    # valid_data = scale_df(valid_df, std_scaler, fit_scaler=False)
    # test_data = scale_df(test_df, std_scaler, fit_scaler=False)

    X_train, y_train = get_x_and_y(train_df)
    X_valid, y_valid = get_x_and_y(valid_df)
    X_test, y_test = get_x_and_y(test_df)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def setup_mlflow(experiment_name):
    mlflow.set_tracking_uri(MLRUNS_FOLDER_PATH)
    mlflow.set_experiment(experiment_name)


def calculate_metrics(model, x, y_true, n_bins, i_move_bins):
    balanced_accuracy_per_bin = {"balanced_accuracy": {}}
    recall_per_bin = {class_label: {} for class_label in ["1-0", "0-1", "1/2-1/2"]}
    precision_per_bin = {class_label: {} for class_label in ["1-0", "0-1", "1/2-1/2"]}

    for bin_id in range(n_bins):
        bin_mask = (i_move_bins == bin_id)
        X_test_filtered = x[bin_mask]
        y_test_filtered = y_true[bin_mask]

        if len(y_test_filtered) == 0:
            continue

        y_pred = model.predict(X_test_filtered)
        balanced_accuracy_per_bin["balanced_accuracy"][bin_id] = balanced_accuracy_score(y_test_filtered, y_pred)

        for class_label in recall_per_bin.keys():
            recall_per_bin[class_label][bin_id] = recall_score(
                y_test_filtered, y_pred, labels=[class_label], average=None, zero_division=0
            )
            precision_per_bin[class_label][bin_id] = precision_score(
                y_test_filtered, y_pred, labels=[class_label], average=None, zero_division=0
            )
    return balanced_accuracy_per_bin, recall_per_bin, precision_per_bin


def save_metrics_plot(ax, metrics_per_bin, bin_centers, n_bins, title):
    # colors = plt.get_cmap("Dark2")
    for i, class_label in enumerate(metrics_per_bin.keys()):
        ax.plot(
            bin_centers,
            [metrics_per_bin[class_label].get(i, 0) for i in range(n_bins)],
            marker="o",
            linestyle="-",
            label=f"{class_label}",
            # color=colors(i),
        )
    ax.set_title(title)
    ax.set_xlabel("i_move (Step Number)")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(True)


def save_x_distribution(n_bins, i_move_bins, y_true, bin_centers, dir_path):
    class_distribution_per_bin = pd.DataFrame(0, index=range(n_bins), columns=["1-0", "0-1", "1/2-1/2"])

    for bin_id in range(n_bins):
        bin_mask = (i_move_bins == bin_id)
        class_counts = y_true[bin_mask].value_counts()

        for class_label in class_counts.index:
            class_distribution_per_bin.loc[bin_id, class_label] = class_counts[class_label]

    bar_width = bin_centers[1] - bin_centers[0]
    bottom_values = np.zeros(n_bins)

    fig = plt.figure(figsize=(12, 6))
    for class_label in class_distribution_per_bin.columns:
        plt.bar(bin_centers,
                class_distribution_per_bin[class_label],
                width=bar_width,
                bottom=bottom_values,
                label=class_label,
                edgecolor='black'
                )
        bottom_values += class_distribution_per_bin[class_label]

    title = "Class Distribution vs i_move (Binned)"
    plt.xlabel("i_move (Step Number)")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(title="Class")
    # plt.grid(True)

    image_path = os.path.join(dir_path, f"{title.lower().replace(' ', '_')}_plot.png")
    plt.savefig(image_path)
    plt.close()


def log_prediction(model, x, y_true, set_name):
    if not os.path.exists(set_name):
        os.mkdir(set_name)
    balanced_accuracy = balanced_accuracy_score(y_true, model.predict(x))
    mlflow.log_metric(f"{set_name} Balanced Accuracy", balanced_accuracy)

    n_bins = 8
    bin_edges = np.linspace(0, max(x["i_move"]), n_bins + 1)
    i_move_bins = pd.cut(x["i_move"], bins=bin_edges, labels=False, include_lowest=True)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    save_x_distribution(n_bins, i_move_bins, y_true, bin_centers, set_name)

    balanced_accuracy, recall, precision = calculate_metrics(model, x, y_true, n_bins, i_move_bins)

    fig, (ax_recall, ax_precision, ax_accuracy) = plt.subplots(3, 1, figsize=(8, 12))

    save_metrics_plot(ax_recall, recall, bin_centers, n_bins, "Recall")
    save_metrics_plot(ax_precision, precision, bin_centers, n_bins, "Precision")
    save_metrics_plot(ax_accuracy, balanced_accuracy, bin_centers, n_bins, "Balanced Accuracy")
    plt.tight_layout()

    image_path = os.path.join(set_name, "metrics_plot.png")
    plt.savefig(image_path)
    plt.close()

    mlflow.log_artifact(set_name)
    shutil.rmtree(set_name)
