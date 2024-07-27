import os

from config import ROOT_DIR


# Dataset constants
MIN_DATE = "2017-03"  # 2017 - March; size - 2.17 GB; games - 11,346,745
BASE_NAME = "lichess_db_standard_rated_"
EXAMPLE_NAME = BASE_NAME + MIN_DATE
EXAMPLE_URL = f"https://database.lichess.org/standard/{EXAMPLE_NAME}.pgn.zst"

# Folder paths
EXTERNAL_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "external")
RAW_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "processed")

EXAMPLE_CSV_DIR = os.path.join(RAW_FOLDER_PATH, EXAMPLE_NAME)

# Results constants
WHITE_WIN_STR: str = "1-0"
BLACK_WIN_STR: str = "0-1"
DRAW_STR: str = "1/2-1/2"

WHITE_WIN_INT = 1
BLACK_WIN_INT = 0
DRAW_INT = -1

RESULTS_STR_TO_INT = {
    WHITE_WIN_STR: WHITE_WIN_INT,
    BLACK_WIN_STR: BLACK_WIN_INT,
    DRAW_STR: DRAW_INT
}
RESULTS_INT_TO_STR = {
    WHITE_WIN_INT: WHITE_WIN_STR,
    BLACK_WIN_INT: BLACK_WIN_STR,
    DRAW_INT: DRAW_STR
}

RESULTS_STR_TO_STR = {
    WHITE_WIN_STR: "White Win",
    BLACK_WIN_STR: "Black Win",
    DRAW_STR: "Draw"
}

# Data processing constants
MIN_GAME_DURATION = 0  # seconds

# TODO add min moves value
# MIN_MOVES_IN_GAME = 4

BASELINE_COLUMNS = [
    "Event",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "Result",
    "times_list",
    "Termination",
    "ECO",
    "White",
    "Black"
]

# TARGET_COLUMNS = ["ResultEncoded", "WhiteWin", "BlackWin", "Draw"]
