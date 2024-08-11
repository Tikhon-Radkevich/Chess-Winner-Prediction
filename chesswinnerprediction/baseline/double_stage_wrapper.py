import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import compute_sample_weight
from sklearn import metrics

from chesswinnerprediction.baseline.custom_knn import CustomKNNClassifier

from chesswinnerprediction.constants import DRAW_STR
from chesswinnerprediction.baseline.utils import estimate_baseline_model


class Model:
    win_to_draw_splitter_draw_symbol = False
    win_to_draw_splitter_win_symbol = True

    def __init__(self, win_to_draw_splitter, black_to_white_splitter, name):
        self.win_to_draw_splitter = win_to_draw_splitter
        self.black_to_white_splitter = black_to_white_splitter
        self.name = name
        self.classes_ = None

    def __repr__(self):
        return f"{self.win_to_draw_splitter}-{self.black_to_white_splitter}"

    def fit(self, x, y):
        # Predict whether the game is a win or a draw
        win_condition = y != DRAW_STR

        win_to_draw_splitter_sample_weight = compute_sample_weight("balanced", win_condition)
        self.win_to_draw_splitter.fit(x, win_condition, win_to_draw_splitter_sample_weight)

        # If the game is not a draw, predict which player wins
        black_to_white_sample_weight = compute_sample_weight("balanced", y[win_condition])
        self.black_to_white_splitter.fit(x[win_condition], y[win_condition], black_to_white_sample_weight)

        # Set class labels
        self.classes_ = [DRAW_STR, *self.black_to_white_splitter.classes_]

    def predict_proba(self, x):
        win_probs = self.win_to_draw_splitter.predict_proba(x)

        classes = self.win_to_draw_splitter.classes_.tolist()
        win_prob = win_probs[:, classes.index(self.win_to_draw_splitter_win_symbol)]
        draw_prob = win_probs[:, classes.index(self.win_to_draw_splitter_draw_symbol)]

        # P(black_win) = P(black_win | win) * P(win)
        # P(white_win) = P(white_win | win) * P(win)
        black_to_win_probs = self.black_to_white_splitter.predict_proba(x)
        black_to_win_probs *= win_prob[:, np.newaxis]

        # according to self.classes_ :
        probs = np.zeros((x.shape[0], 3))
        probs[:, 0] = draw_prob  # set draw prob
        probs[:, 1:] = black_to_win_probs  # set black and white probs

        return probs

    def score(self, x, y):
        predictions = self.predict(x)
        return metrics.balanced_accuracy_score(y, predictions)

    def predict(self, x):
        is_win = self.win_to_draw_splitter.predict(x)

        predictions = np.full(x.shape[0], DRAW_STR, dtype=object)
        if np.any(is_win):
            predictions[is_win] = self.black_to_white_splitter.predict(x[is_win])

        return predictions

    @staticmethod
    def _get_feature_importance(model):
        if isinstance(model, (LogisticRegression, CustomKNNClassifier)):
            return model.coef_
        # elif isinstance(model, DecisionTreeClassifier):
        #     return [model.feature_importances_]
        elif isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier)):
            return model.feature_importances_.reshape(1, -1)

        return None

    def estimate_model(self, x_train, y_train_data, x_valid, y_valid_data, **kwargs):

        feature_importance_1 = self._get_feature_importance(self.win_to_draw_splitter)
        feature_importance_2 = self._get_feature_importance(self.black_to_white_splitter)

        win_condition_train = y_train_data != DRAW_STR
        win_condition_valid = y_valid_data != DRAW_STR

        black_to_white_y_train = y_train_data[win_condition_train]
        black_to_white_y_valid = y_valid_data[win_condition_valid]
        black_to_white_x_train = x_train[win_condition_train]
        black_to_white_x_valid = x_valid[win_condition_valid]

        print(" " * 4, f"Win to Draw Splitter \n{self.win_to_draw_splitter}")
        estimate_baseline_model(
            self.win_to_draw_splitter,
            feature_importance_1,
            x_train,
            win_condition_train,
            x_valid,
            win_condition_valid,
            **kwargs
        )
        print(" " * 4, f"Black to White Splitter \n{self.black_to_white_splitter}")
        estimate_baseline_model(
            self.black_to_white_splitter,
            feature_importance_2,
            black_to_white_x_train,
            black_to_white_y_train,
            black_to_white_x_valid,
            black_to_white_y_valid,
            **kwargs
        )
        print(" " * 4, "Overall:")
        estimate_baseline_model(
            self,
            None,
            x_train,
            y_train_data,
            x_valid,
            y_valid_data,
            **kwargs
        )
