BASELINE_RANDOM_STATE = 42

BASELINE_COLUMNS = [
    "Result",
    "EloDiff",
    "MeanElo",
    "WhiteElo",
    "BlackElo",
    "Event",
    "BaseTime",
    "IncrementTime",
    "ZeroIncrementTime"
]

columns_to_scale = ["WhiteElo", "BlackElo", "EloDiff", "BaseTime", "IncrementTime", "MeanElo"]
