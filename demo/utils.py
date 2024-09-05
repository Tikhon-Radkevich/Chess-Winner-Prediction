import joblib
import pandas as pd
import chess
import matplotlib.pyplot as plt

from chesswinnerprediction.static_move.processing.utils import process_df
from chesswinnerprediction.constants import BASELINE_MODEL, STATIC_MOVE_MODEL, BASELINE_DEMO_DATA, STATIC_MOVE_DEMO_DATA


def static_move_prediction(static_move_model, game):
    X_data: pd.DataFrame = process_df(game)
    X_data.drop(columns=["Event", "Result"], inplace=True)
    predictions = static_move_model.predict_proba(X_data)
    return predictions


def get_baseline_prediction(baseline_model, data):
    X_data = data.drop(columns=["GameId"])
    predictions = baseline_model.predict_proba(X_data)
    return predictions


def process_game(static_move_model, game):
    board = chess.Board()
    game_dict = dict()
    st_predictions = static_move_prediction(static_move_model, game)
    for ((_, (i_move, move_san)), predict) in zip(game[["i_move", "chess_moves_list"]].iterrows(), st_predictions):
        move = board.parse_san(move_san)
        board.push(move)
        game_dict[i_move] = {
            "board": board.copy(),
            "static_move_prediction": predict,
        }
    return game_dict


def get_games():
    baseline_model = joblib.load(BASELINE_MODEL)
    static_move_model = joblib.load(STATIC_MOVE_MODEL)

    static_move_data_df = pd.read_csv(STATIC_MOVE_DEMO_DATA)
    baseline_data_df = pd.read_csv(BASELINE_DEMO_DATA)

    games = dict()
    game_ids = static_move_data_df["GameId"].unique()
    for game_id in game_ids:
        x_baseline = baseline_data_df[baseline_data_df["GameId"] == game_id]
        baseline_prediction = get_baseline_prediction(baseline_model, x_baseline)

        x_static_move = static_move_data_df[static_move_data_df["GameId"] == game_id]
        game_result = x_static_move["Result"].values[0]
        game_processed = process_game(static_move_model, x_static_move)

        games[game_id] = {
            "result": game_result,
            "baseline_prediction": baseline_prediction,
            "game": game_processed
        }
    return games, baseline_model.classes_, static_move_model.classes_


def get_game_prediction_plot(game, labels, i_move):
    plot_data = {
        labels[0]: [],
        labels[1]: [],
        labels[2]: [],
    }
    for move, data in game.items():
        plot_data[labels[0]].append(data["static_move_prediction"][0])
        plot_data[labels[1]].append(data["static_move_prediction"][1])
        plot_data[labels[2]].append(data["static_move_prediction"][2])

    df = pd.DataFrame(plot_data)
    df.index = range(1, len(df) + 1)

    # Plotting the data
    ax = df.plot(kind="line", figsize=(8, 3), title="Result Prediction Over Moves")

    # Adding the vertical line at x = i_move
    plt.axvline(x=i_move, color='red', linestyle='--', label=f"Move {i_move}")

    # Add labels and legend
    plt.xlabel("Move")
    plt.ylabel("Prediction Proba")
    plt.legend()
