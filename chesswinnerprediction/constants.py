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
INTERIM_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "interim")

EXAMPLE_CSV_DIR = os.path.join(RAW_FOLDER_PATH, EXAMPLE_NAME)

MODELS_DIR = os.path.join(ROOT_DIR, "models")
BASELINE_MODEL = os.path.join(MODELS_DIR, "baseline.pkl")
STATIC_MOVE_MODEL = os.path.join(MODELS_DIR, "static_move.pkl")

DEMO_DATA_DIR = os.path.join(PROCESSED_FOLDER_PATH, "demo")
BASELINE_DEMO_DATA = os.path.join(DEMO_DATA_DIR, "baseline.csv")
STATIC_MOVE_DEMO_DATA = os.path.join(DEMO_DATA_DIR, "static_move.csv")
INTERIM_DEMO_DATA = os.path.join(DEMO_DATA_DIR, "interim.csv")

# static_move constants
# todo: remove commented code
STATIC_MOVE = "static_move"
INTERIM_STATIC_MOVE = os.path.join(INTERIM_FOLDER_PATH, STATIC_MOVE)
# INTERIM_STATIC_MOVE_SPLIT_BALANCED = os.path.join(INTERIM_STATIC_MOVE_SPLIT, "balanced")
# INTERIM_STATIC_MOVE_SPLIT_ORIGINAL = os.path.join(INTERIM_STATIC_MOVE_SPLIT, "original")
# INTERIM_STATIC_MOVE_SPLIT_EXTRA = os.path.join(INTERIM_STATIC_MOVE_SPLIT, "extra")

PROCESSED_STATIC_MOVE = os.path.join(PROCESSED_FOLDER_PATH, STATIC_MOVE)
# PROCESSED_STATIC_MOVE_ORIGINAL = os.path.join(PROCESSED_FOLDER_PATH, STATIC_MOVE, "original")
# STATIC_MOVE_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "static_move")
GAME_ID = "GameId"

if not os.path.exists(INTERIM_STATIC_MOVE):
    os.makedirs(INTERIM_STATIC_MOVE)
# if not os.path.exists(INTERIM_STATIC_MOVE_SPLIT_BALANCED):
#     os.makedirs(INTERIM_STATIC_MOVE_SPLIT_BALANCED)
# if not os.path.exists(INTERIM_STATIC_MOVE_SPLIT_ORIGINAL):
#     os.makedirs(INTERIM_STATIC_MOVE_SPLIT_ORIGINAL)
# if not os.path.exists(INTERIM_STATIC_MOVE_SPLIT_EXTRA):
#     os.makedirs(INTERIM_STATIC_MOVE_SPLIT_EXTRA)
if not os.path.exists(PROCESSED_STATIC_MOVE):
    os.makedirs(PROCESSED_STATIC_MOVE)

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
    DRAW_STR: DRAW_INT,
}
RESULTS_INT_TO_STR = {
    WHITE_WIN_INT: WHITE_WIN_STR,
    BLACK_WIN_INT: BLACK_WIN_STR,
    DRAW_INT: DRAW_STR,
}

RESULTS_STR_TO_STR = {
    WHITE_WIN_STR: "White Win",
    BLACK_WIN_STR: "Black Win",
    DRAW_STR: "Draw",
    False: "Draw",
    True: "Win",
}

# Data processing constants
MIN_GAME_DURATION = 0  # seconds
MIN_MOVES_IN_GAME = 4

PROCESS_COLUMNS = [
    "Event",
    "WhiteElo",
    "BlackElo",
    "TimeControl",
    "Result",
    "times_list",
    "Termination",
    "evaluations_list",
    "chess_moves_list",
    "ECO",
    "White",
    "Black",
    "parse_success",
]

# TARGET_COLUMNS = ["ResultEncoded", "WhiteWin", "BlackWin", "Draw"]
