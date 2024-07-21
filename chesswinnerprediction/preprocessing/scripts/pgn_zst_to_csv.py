import os

from tqdm import tqdm
import zstandard as zstd
import pandas as pd


HEADER_TITLES = [
    "Event",
    "Site",
    "White",
    "Black",
    "Result",
    "WhiteElo",
    "BlackElo",
    "WhiteRatingDiff",
    "BlackRatingDiff",
    "ECO",
    "Opening",
    "TimeControl",
    "Termination",
]
MOVE_TITLES = ["chess_moves_list", "evaluations_list", "times_list"]
DATA_TITLES = HEADER_TITLES + MOVE_TITLES
HEADER_TITLES_SET = set(HEADER_TITLES)


def process_moves_and_evals_and_ckl(moves):
    moves = moves.replace(b"!", b"").replace(b"?", b"").decode("utf-8")
    split_moves = moves.split(". ")[1:]

    chess_moves, evaluations, times = [], [], []
    for move_data in split_moves:
        move_parts = move_data.split(" ")
        if len(move_parts) > 8:
            break

        chess_move = move_parts[0]
        evaluation, clk = None, None

        if len(move_parts) == 8:
            evaluation = move_parts[3][:-1]
            clk = move_parts[5][:-1]
        elif len(move_parts) == 6:
            if move_parts[2] == "%eval":
                evaluation = move_parts[3][:-1]
            else:
                clk = move_parts[3][:-1]

        chess_moves.append(chess_move)
        evaluations.append(evaluation)
        times.append(clk)

    return chess_moves, evaluations, times


def process_and_add_moves(moves, data):
    chess_moves, evaluations, times = process_moves_and_evals_and_ckl(moves)

    data["chess_moves_list"].append(chess_moves)
    data["evaluations_list"].append(evaluations)
    data["times_list"].append(times)


def process_and_add_headers(headers, data):
    header_set_names = HEADER_TITLES_SET.copy()

    def add_none_to_data():
        for header_name in header_set_names:
            data[header_name].append(None)

    def convert_to_string(x):
        return x.decode("utf-8")[1:-1].split(" ", 1)

    headers = headers.split(b"\n")
    for header in headers:
        name, value = convert_to_string(header)
        if name in HEADER_TITLES_SET:
            header_set_names.remove(name)
            data[name].append(value[1:-1])

    add_none_to_data()


def save_df_and_clear_data(df_dir_path, data, idx):
    file_path_with_idx = os.path.join(df_dir_path, f"data_{idx}.csv")
    print(f"\nSaving data to {file_path_with_idx}")
    pd.DataFrame(data).to_csv(file_path_with_idx, index=False)
    for values in data.values():
        values.clear()


def pgn_zst_to_dataframe(pgn_zst_path, df_dir_path, split_size=125000):
    MOVES_PATTERN = b"] [%clk "
    ZST_COMPRESSION_INDEX = (
        7.1  # info from https://database.lichess.org/#standard_games
    )
    estimated_total_size = os.path.getsize(pgn_zst_path) * ZST_COMPRESSION_INDEX
    print(f"Estimated total size: ~{estimated_total_size / (1024 ** 3):.1f}GB")

    data = {title: [] for title in DATA_TITLES}
    file_idx = 0

    with open(pgn_zst_path, "rb") as compressed_file:
        with zstd.ZstdDecompressor().stream_reader(compressed_file) as reader:
            with tqdm(unit="B", unit_scale=True, desc="Reading file") as pbar:
                buffer, part = b"", b""
                for chunk in iter(lambda: reader.read(size=4096), b""):
                    buffer += chunk
                    pbar.update(len(chunk))
                    while b"\n\n" in buffer:
                        header_part = part
                        part, buffer = buffer.split(b"\n\n", maxsplit=1)
                        if MOVES_PATTERN in part:
                            process_and_add_headers(header_part, data)
                            process_and_add_moves(part, data)

                        if len(data["chess_moves_list"]) >= split_size:
                            save_df_and_clear_data(df_dir_path, data, file_idx)
                            file_idx += 1

    save_df_and_clear_data(df_dir_path, data, file_idx)

