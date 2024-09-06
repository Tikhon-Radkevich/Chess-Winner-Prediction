import streamlit as st
import pandas as pd
import chess.svg
import chess

from utils import generate_game_prediction_plot


def show_general_game_info(games, game_id):
    game_data = {
        "Game Result": [games[game_id]["result"]],
        "Time Control": [games[game_id]["TimeControl"]],
        "WhiteElo": [games[game_id]["WhiteElo"]],
        "BlackElo": [games[game_id]["BlackElo"]],
    }
    st.dataframe(pd.DataFrame(game_data), hide_index=True, use_container_width=True)


def show_baseline_prediction(games, game_id, baseline_labels):
    pregame_predictions = games[game_id]["baseline_prediction"][0]
    pregame_predict_data = {
        "Label": baseline_labels,
        "Prediction (%)": [f"{pred * 100:.2f}%" for pred in pregame_predictions]
    }
    pregame_predict_df = pd.DataFrame(pregame_predict_data).set_index('Label').T

    st.markdown("<h6 style='text-align: center;'>Baseline Prediction</h6>", unsafe_allow_html=True)
    st.dataframe(pregame_predict_df, hide_index=True, use_container_width=True)


def show_static_move_prediction(games, game_id, i_move, static_move_labels):
    static_move_prediction = games[game_id]["game"][i_move]["static_move_prediction"]
    game_predict_data = {
        "Label": static_move_labels,
        "Prediction (%)": [f"{pred * 100:.2f}%" for pred in static_move_prediction]
    }
    game_predict_df = pd.DataFrame(game_predict_data).set_index("Label").T
    st.markdown("<h6 style='text-align: center;'>Static Move Prediction</h6>", unsafe_allow_html=True)
    st.dataframe(game_predict_df, hide_index=True, use_container_width=True)


def show_page(games, game_id, baseline_labels, static_move_labels):
    show_general_game_info(games, game_id)

    board_column, description_column = st.columns((5, 4))
    with description_column:
        show_baseline_prediction(games, game_id, baseline_labels)

        n_moves = len(games[game_id]["game"].keys())
        i_move = st.slider("Pick Move", 1, n_moves, 1)

        show_static_move_prediction(games, game_id, i_move, static_move_labels)

    with board_column:
        board = games[game_id]["game"][i_move]["board"]
        board_svg = chess.svg.board(board)

        st.image(board_svg, width=350)

    fig = generate_game_prediction_plot(games[game_id]["game"], static_move_labels, i_move)
    st.plotly_chart(fig)


def show_select_box(games):
    option_to_game_id = {
        f"{i}) Result: {value['result']}": game_id for i, (game_id, value) in enumerate(games.items(), 1)
    }
    option = st.selectbox("Select The Game", options=option_to_game_id.keys())
    return option_to_game_id[option]
