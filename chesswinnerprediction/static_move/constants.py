# Static Move Constants
RANDOM_STATE = 42

PIECE_VALUES = [0, 1, 3, 3, 5, 9, 0]

STATIC_MOVE_COLUMNS = [
    "GameId",
    "Event",
    "Result",
    "eval",
    "EloDiff",
    "MeanElo",
    "BaseTime",
    "IncrementTime",
    "time_diff_norm",
    "white_remaining_time_norm",
    "black_remaining_time_norm",
    "GameDurations_norm",
    "chess_moves_list",
    "is_checkmate_countdown",
    "i_move",
]

# Default values for static move scripts
TRAIN_VALID_TEST = ("train.csv", "valid.csv", "test.csv")
DEFAULT_N_SAMPLES = (60_000, 25_000, None)
