import os

import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibrationDisplay, calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

from IPython.display import display, HTML

from config import PROCESSED_FOLDER_PATH, BASELINE_EXPERIMENT, MLRUNS_FOLDER_PATH
from chesswinnerprediction.constants import RESULTS_STR_TO_STR, DRAW_STR
from chesswinnerprediction.baseline.constants import (
    BASELINE_COLUMNS,
    columns_to_scale,
    BASELINE_RANDOM_STATE,
)


def load_train_valid_test(
    data_dir: str = "lichess_db_standard_rated_2017-05",
    train_sample_size: float = 0.2,
    valid_sample_size: float = 0.1,
    test_sample_size: float = 1.0,
    random_state: int = BASELINE_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Returns: X_train, y_train, X_valid, y_valid, X_test, y_test
    """

    data_path = os.path.join(PROCESSED_FOLDER_PATH, data_dir)

    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    valid_df = pd.read_csv(os.path.join(data_path, "valid.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    train_df = train_df.sample(frac=train_sample_size, random_state=random_state)
    valid_df = valid_df.sample(frac=valid_sample_size, random_state=random_state)
    test_df = test_df.sample(frac=test_sample_size, random_state=random_state)

    std_scaler = StandardScaler()
    train_data = transform_and_scale_df(train_df, std_scaler)
    valid_data = transform_and_scale_df(valid_df, std_scaler, fit_scaler=False)
    test_data = transform_and_scale_df(test_df, std_scaler, fit_scaler=False)

    X_train, y_train = get_x_and_y(train_data, predict_draws=True)
    X_valid, y_valid = get_x_and_y(valid_data, predict_draws=True)
    X_test, y_test = get_x_and_y(test_data, predict_draws=True)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def show_feature_importance(model, feature_importance, grid_y=False) -> None:
    labels = [RESULTS_STR_TO_STR[label] for label in model.classes_]
    num_classes = len(feature_importance)
    importance_dfs = []
    for i in range(num_classes):
        importance_df = pd.DataFrame(
            {"Feature": model.feature_names_in_, "Importance": feature_importance[i]}
        )
        importance_df.sort_values(by="Importance", ascending=False, inplace=True)
        importance_dfs.append(importance_df)

    if num_classes == 1:
        fig_size = (7, 5)
        labels = ["Feature Importance"]
    else:
        fig_size = (15, 5)
    _, axes = plt.subplots(ncols=num_classes, figsize=fig_size, sharey=True)
    if num_classes == 1:
        axes = [axes]

    for i in range(num_classes):
        sns.barplot(
            x="Importance",
            y="Feature",
            hue="Feature",
            data=importance_dfs[i],
            palette="viridis",
            ax=axes[i],
        )

        axes[i].set_xlabel("Importance", fontsize=12)
        axes[i].set_title(labels[i], fontsize=14)
        if grid_y:
            axes[i].grid(axis="y")
        axes[i].invert_yaxis()

    axes[0].set_ylabel("Feature", fontsize=12)

    plt.tight_layout()
    plt.show()


def print_report(
    model,
    x1,
    y1,
    x2=None,
    y2=None,
    report_title_1="Train Report",
    report_title_2="Validation Report",
) -> None:
    predict_1 = model.predict(x1)
    report_1 = metrics.classification_report(y1, predict_1, zero_division=np.nan)

    print("\n" + " " * 48 + "Classification Report")
    if x2 is None:
        print(report_1)
        return

    predict_2 = model.predict(x2)
    report_2 = metrics.classification_report(y2, predict_2, zero_division=np.nan)

    print(" " * 24, report_title_1, " " * 36, report_title_2)
    for part_1, part_2 in zip(report_1.split("\n\n"), report_2.split("\n\n")):
        for val_1, val_2 in zip(part_1.split("\n"), part_2.split("\n")):
            print(val_1, " " * 6 + val_2[12:])


def plot_confusion_matrix(model, predict, y_test) -> None:
    conf_matrix = metrics.confusion_matrix(
        y_test, predict, labels=model.classes_, normalize="true"
    )
    labels = [RESULTS_STR_TO_STR[label] for label in model.classes_]
    metrics.ConfusionMatrixDisplay(conf_matrix, display_labels=labels).plot(
        cmap="Blues"
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def plot_calibration_curve(model, prob_predict, y_test: pd.Series) -> None:
    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    n_classes = len(model.classes_)

    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 5), sharey=True)
    colors = plt.get_cmap("tab10")

    for i, class_label in enumerate(model.classes_):
        ax = axes[i]

        class_index = i
        CalibrationDisplay.from_predictions(
            y_test == class_label,
            prob_predict[:, class_index],
            n_bins=n_bins,
            name=f"{class_label}",
            ax=ax,
            strategy="uniform",
            color=colors(i)
        )

        bin_counts = np.zeros((n_bins, n_classes))
        bin_indices = np.searchsorted(bin_edges[1:-1], prob_predict[:, i])

        for bin_i in range(n_bins):
            for j in range(n_classes):
                bin_counts[bin_i, j] = np.sum(bin_indices[y_test == model.classes_[j]] == bin_i)

        ax2 = ax.twinx()
        bottom = np.zeros(n_bins)
        bar_width = bin_edges[1] - bin_edges[0]

        for j in range(n_classes):
            ax2.bar(
                bin_edges[:-1] + bar_width / 2,
                bin_counts[:, j],
                width=bar_width,
                bottom=bottom,
                alpha=0.1,
                edgecolor="black",
                color=colors(j),
                label=f'{model.classes_[j]}'
            )
            bottom += bin_counts[:, j]

        ax.legend(loc="upper right")
        ax.set_ylabel("")
        ax2.set_ylabel("")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_title(class_label)

        if i == 0:
            ax.set_ylabel("Fraction of Positives")
        elif i == n_classes - 1:
            ax2.set_ylabel("Number of Samples")

    plt.suptitle("Calibration Curves and Sample Distributions")
    plt.tight_layout()
    plt.show()


def compute_brier_score_loss(prob_predict, y_test) -> float:
    return metrics.brier_score_loss(y_test, prob_predict)


def estimate_baseline_model(
    model, feature_importance, x_train, y_train, x_test, y_test, **kwargs
) -> None:
    predict = model.predict(x_test)
    prob_predict = model.predict_proba(x_test)

    loss = metrics.log_loss(y_test, prob_predict)
    print(f"Log Loss on test data: {round(loss, 4)}")

    weighted_accuracy = metrics.balanced_accuracy_score(y_test, predict)
    print(f"Balanced Accuracy on test data: {round(weighted_accuracy * 100, 2)}%\n")

    print_report(model, x_train, y_train, x_test, y_test, **kwargs)

    plot_confusion_matrix(model, predict, y_test)

    plot_calibration_curve(model, prob_predict, y_test)

    if feature_importance is not None:
        show_feature_importance(model, feature_importance)


def get_class_weights(y, verbose=False):
    unique_y = y.unique()
    class_weights = compute_class_weight("balanced", classes=unique_y, y=y)
    class_weights = dict(zip(unique_y, class_weights))
    if verbose:
        print("Class weights:")
        for key, value in class_weights.items():
            print(f"\t{key}: {float(value)}")

    return class_weights


def get_x_and_y(data, predict_draws=False) -> tuple[pd.DataFrame, pd.Series]:
    if not predict_draws:
        data = data[data["Result"] != DRAW_STR]

    x_data = data.drop(columns=["Result"])
    y_data = data["Result"]

    return x_data, y_data


def transform_and_scale_df(
    df, scaler, fit_scaler=True, dummies_event=True
) -> pd.DataFrame:
    X = df[BASELINE_COLUMNS].copy()
    X["ZeroIncrementTime"] = X["ZeroIncrementTime"].astype(np.float64)
    if dummies_event:
        X = pd.get_dummies(
            X, columns=["Event"], dtype=np.bool_, prefix="", prefix_sep=""
        )
    if fit_scaler:
        X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    else:
        X[columns_to_scale] = scaler.transform(X[columns_to_scale])
    return X


def get_worst_params_df(cv_results) -> pd.DataFrame:
    df_results = pd.DataFrame(cv_results)
    sorted_df = df_results.sort_values(by="mean_test_score")
    param_cols = [col for col in sorted_df.columns if col.startswith("param_")]
    worst_params = (
        sorted_df[param_cols + ["mean_test_score"]].groupby(param_cols).mean()
    )
    return worst_params.sort_values(by="mean_test_score")


def setup_mlflow(
    tracking_uri=MLRUNS_FOLDER_PATH,
    experiment_name=BASELINE_EXPERIMENT,
    sklearn_autolog_disable=True,
    log_post_training_metrics=True,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(
        disable=sklearn_autolog_disable,
        log_post_training_metrics=log_post_training_metrics,
    )
