import pandas as pd

from utils import get_games, add_game_info
from streamlit_utils import show_page, show_select_box
from chesswinnerprediction.constants import BASELINE_DEMO_DATA, STATIC_MOVE_DEMO_DATA


static_move_data_df = pd.read_csv(STATIC_MOVE_DEMO_DATA)
baseline_data_df = pd.read_csv(BASELINE_DEMO_DATA)

games, baseline_labels, static_move_labels = get_games(baseline_data_df, static_move_data_df)
games = add_game_info(games, static_move_data_df)


def main():
    game_id = show_select_box(games)
    show_page(games, game_id, baseline_labels, static_move_labels)


if __name__ == "__main__":
    main()








