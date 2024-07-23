import os

from config import ROOT_DIR


BASE_NAME = "lichess_db_standard_rated_"
MIN_DATE = "2017-03"  # 2017 - March; size - 2.17 GB; games - 11,346,745
EXAMPLE_NAME = BASE_NAME + MIN_DATE
EXAMPLE_URL = f"https://database.lichess.org/standard/{EXAMPLE_NAME}.pgn.zst"

EXTERNAL_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "external")
RAW_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_FOLDER_PATH = os.path.join(ROOT_DIR, "data", "processed")

EXAMPLE_CSV_DIR = os.path.join(RAW_FOLDER_PATH, EXAMPLE_NAME)

