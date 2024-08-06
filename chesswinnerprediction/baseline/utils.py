import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    log_loss,
    balanced_accuracy_score,
)

from chesswinnerprediction.constants import RESULTS_STR_TO_STR, DRAW_STR
from chesswinnerprediction.baseline.constants import BASELINE_COLUMNS, columns_to_scale


def show_feature_importance(model, feature_importance, grid_y=False):
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
):
    predict_1 = model.predict(x1)
    report_1 = classification_report(y1, predict_1, zero_division=np.nan)

    print("\n" + " " * 48 + "Classification Report")
    if x2 is None:
        print(report_1)
        return

    predict_2 = model.predict(x2)
    report_2 = classification_report(y2, predict_2, zero_division=np.nan)

    print(" " * 24, report_title_1, " " * 36, report_title_2)
    for part_1, part_2 in zip(report_1.split("\n\n"), report_2.split("\n\n")):
        for val_1, val_2 in zip(part_1.split("\n"), part_2.split("\n")):
            print(val_1, " " * 6 + val_2[12:])


def estimate_baseline_model(
    model, feature_importance, x_train, y_train, x_test, y_test
):
    predict = model.predict(x_test)
    prob_predict = model.predict_proba(x_test)

    loss = log_loss(y_test, prob_predict)
    print(f"Log Loss on test data: {round(loss, 4)}")

    weighted_accuracy = balanced_accuracy_score(y_test, predict)
    print(f"Balanced Accuracy on test data: {round(weighted_accuracy*100, 2)}%\n")

    print_report(model, x_train, y_train, x_test, y_test)

    conf_matrix = confusion_matrix(y_test, predict, labels=model.classes_)
    labels = [RESULTS_STR_TO_STR[label] for label in model.classes_]
    ConfusionMatrixDisplay(conf_matrix, display_labels=labels).plot(cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

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


def get_x_and_y(data, predict_draws=False):
    if not predict_draws:
        data = data[data["Result"] != DRAW_STR]

    x_data = data.drop(columns=["Result"])
    y_data = data["Result"]

    return x_data, y_data


def transform_and_scale_df(df, scaler, fit_scaler=True):
    X = df[BASELINE_COLUMNS].copy()
    X = pd.get_dummies(X, columns=["Event"], dtype=np.int8, prefix="", prefix_sep="")
    if fit_scaler:
        X[columns_to_scale] = scaler.fit_transform(X[columns_to_scale])
    else:
        X[columns_to_scale] = scaler.transform(X[columns_to_scale])
    return X


def get_worst_params_df(cv_results):
    df_results = pd.DataFrame(cv_results)
    sorted_df = df_results.sort_values(by="mean_test_score")
    param_cols = [col for col in sorted_df.columns if col.startswith("param_")]
    worst_params = (
        sorted_df[param_cols + ["mean_test_score"]].groupby(param_cols).mean()
    )
    return worst_params.sort_values(by="mean_test_score")
