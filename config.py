import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

PROCESSED_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "processed")
MLRUNS_FOLDER_PATH = os.path.join(ROOT_DIR, "mlruns")

BASELINE_EXPERIMENT = "baseline"

# mlflow ui --backend-store-uri=/home/tikhon/PycharmProjects/ChessWinnerPrediction/mlruns --port=5000
