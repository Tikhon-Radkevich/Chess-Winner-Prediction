import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "processed")
INTERIM_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "interim")
MLRUNS_FOLDER_PATH = os.path.join(ROOT_DIR, "mlruns")

BASELINE_EXPERIMENT = "baseline"

RANDOM_STATE = 42
