import joblib
import pandas as pd
import chess
import plotly.graph_objects as go

from chesswinnerprediction.static_move.processing.utils import process_df
from chesswinnerprediction.constants import BASELINE_MODEL, STATIC_MOVE_MODEL, INTERIM_DEMO_DATA


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
    for ((_, (i_move, move_san)), predict) in (
            zip(game[["i_move", "chess_moves_list"]].iterrows(), st_predictions)
    ):
        move = board.parse_san(move_san)
        board.push(move)
        game_dict[i_move] = {
            "board": board.copy(),
            "static_move_prediction": predict,
        }
    return game_dict


def get_games(baseline_data_df, static_move_data_df):
    baseline_model = joblib.load(BASELINE_MODEL)
    static_move_model = joblib.load(STATIC_MOVE_MODEL)

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


def add_game_info(games, static_move_data_df):
    interim_data = pd.read_csv(INTERIM_DEMO_DATA)
    for game_id in games.keys():
        game_row = interim_data[interim_data["GameId"] == game_id]
        games[game_id]["TimeControl"] = game_row["TimeControl"].values[0]
        games[game_id]["WhiteElo"] = str(game_row["WhiteElo"].values[0])
        games[game_id]["BlackElo"] = str(game_row["BlackElo"].values[0])
        games[game_id]["White"] = game_row["White"].values[0]
        games[game_id]["Black"] = game_row["Black"].values[0]

        static_move_game = static_move_data_df[static_move_data_df["GameId"] == game_id]
        w_times = static_move_game["white_remaining_time_norm"]
        b_times = static_move_game["black_remaining_time_norm"]
        evals = static_move_game["eval"]
        base_time = static_move_game["BaseTime"].values[0]
        for i_move, w_time, b_time, ev in zip(static_move_game["i_move"], w_times, b_times, evals):
            games[game_id]["game"][i_move]["white_remaining_time"] = int(w_time*base_time)
            games[game_id]["game"][i_move]["black_remaining_time"] = int(b_time*base_time)
            games[game_id]["game"][i_move]["eval"] = ev

    return games


def generate_game_prediction_plot(game, labels, i_move):
    # Prepare plot data
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

    fig = go.Figure()

    for label in labels:
        fig.add_trace(go.Scatter(x=df.index, y=df[label], mode='lines', name=label))

    # Add vertical line at x = i_move
    fig.add_vline(
        x=i_move,
        line=dict(color="gray", dash="dash"),
        annotation_text=f"Move {i_move}",
        annotation_position="top right"
    )

    # Customize the layout
    fig.update_layout(
        title={
            'text': "Result Prediction Over Moves",
            'x': 0.5,  # Centers the title
            'xanchor': 'center'
        },
        xaxis_title="Move",
        yaxis_title="Prediction Proba",
        template="plotly_dark",
        legend_title="Labels",
        margin=dict(l=40, r=40, t=40, b=40),
        width=800, height=300
    )
    return fig

