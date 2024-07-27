import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, log_loss

from chesswinnerprediction.constants import RESULTS_STR_TO_STR, DRAW_STR


def estimate_baseline_model(model, x_test, y_test, feature_importance):
    predict = model.predict(x_test)
    prob_predict = model.predict_proba(x_test)

    loss = log_loss(y_test, prob_predict)
    print(f"log loss on test data: {loss}")

    report = classification_report(y_test, predict)
    print("\n" + "\t"*6 + f"Classification Report \n{report}")

    conf_matrix = confusion_matrix(y_test, predict, labels=model.classes_)
    labels = [RESULTS_STR_TO_STR[label] for label in model.classes_]
    ConfusionMatrixDisplay(conf_matrix, display_labels=labels).plot(cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    importance_df = pd.DataFrame({"Feature": model.feature_names_in_, "Importance": feature_importance})
    importance_df.sort_values(by="Importance", ascending=False, inplace=True)
    plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()


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
