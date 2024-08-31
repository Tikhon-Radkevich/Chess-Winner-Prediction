RANDOM_STATE = 42

PIECE_VALUES = [0, 1, 3, 3, 5, 9, 0]

STATIC_MOVE_COLUMNS = [
    "GameId",
    "Event",
    # "WhiteElo",
    # "BlackElo",
    "Result",
    "eval",
    "EloDiff",
    "MeanElo",
    "BaseTime",
    "IncrementTime",
    # "ZeroIncrementTime",
    # "white_remaining_time",
    # "black_remaining_time",
    # "mean_base_time",
    "time_diff_norm",
    "white_remaining_time_norm",
    "black_remaining_time_norm",
    "GameDurations_norm",
    "chess_moves_list",
    "is_checkmate_countdown",
    "i_move",
]

TRAIN_VALID_TEST = ("train.csv", "valid.csv", "test.csv")
I_MOVE_THRESHOLD = (150, 150, 200)
DEFAULT_N_SAMPLES = (60_000, 25_000, None)
