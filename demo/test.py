import pandas as pd
import streamlit as st
import chess
import chess.svg

import matplotlib.pyplot as plt

from utils import get_games, get_game_prediction_plot


games, baseline_labels, static_move_labels = get_games()


def show_game_info(game_id):
    n_moves = len(games[game_id]["game"].keys())
    i_move = st.slider("Pick Move", 1, n_moves, 1)

    board = games[game_id]["game"][i_move]["board"]
    board_svg = chess.svg.board(board)

    board_column, description_column = st.columns((5, 4))
    with board_column:
        st.image(board_svg, width=350)

    with description_column:
        st.write(f"Game Result: {games[game_id]['result']}")

        pregame_predictions = games[game_id]["baseline_prediction"][0]
        pregame_predict_data = {
            "Label": baseline_labels,
            "Prediction (%)": [f"{pred * 100:.2f}%" for pred in pregame_predictions]
        }
        pregame_predict_df = pd.DataFrame(pregame_predict_data).set_index('Label').T  # Transpose the DataFrame
        st.write("Baseline Prediction")
        st.dataframe(pregame_predict_df, hide_index=True, use_container_width=True)

        # Get static move predictions and display them as columns
        static_move_prediction = games[game_id]["game"][i_move]["static_move_prediction"]
        game_predict_data = {
            "Label": static_move_labels,
            "Prediction (%)": [f"{pred * 100:.2f}%" for pred in static_move_prediction]
        }
        game_predict_df = pd.DataFrame(game_predict_data).set_index("Label").T  # Transpose the DataFrame
        st.write("Move Prediction")
        st.dataframe(game_predict_df, hide_index=True, use_container_width=True)

    get_game_prediction_plot(games[game_id]["game"], static_move_labels, i_move)
    st.pyplot(plt)


selected = st.selectbox("Select The Game", games.keys())
show_game_info(selected)





