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
MOVE_TITLES = ["chess_moves_list", "evaluations_list", "times_list", "parse_success"]
DATA_TITLES = HEADER_TITLES + MOVE_TITLES
HEADER_TITLES_SET = set(HEADER_TITLES)


def process_moves_and_evals_and_ckl(moves):
    # TODO: process "?" and "!" in moves
    moves = moves.replace(b"!", b"").replace(b"?", b"").decode("utf-8")
    split_moves = moves.split(". ")[1:]

    parse_success = True
    chess_moves, evaluations, times = [], [], []
    for move_parts in map(str.split, split_moves):
        if len(move_parts) != 8:
            parse_success = False
            break

        chess_move = move_parts[0]
        evaluation = move_parts[3][:-1]
        clk = move_parts[5][:-1]

        chess_moves.append(chess_move)
        evaluations.append(evaluation)
        times.append(clk)

    return chess_moves, evaluations, times, parse_success


def process_and_add_moves(moves, data):
    chess_moves, evaluations, times, parse_success = process_moves_and_evals_and_ckl(moves)

    data["chess_moves_list"].append(chess_moves)
    data["evaluations_list"].append(evaluations)
    data["times_list"].append(times)
    data["parse_success"].append(parse_success)


def process_and_add_headers(headers, data):
    header_data = {}
    for header in headers.split(b"\n"):
        name, value = header.decode("utf-8")[1:-1].split(" ", 1)
        header_data[name] = value[1:-1]

    for name in HEADER_TITLES:
        data[name].append(header_data.get(name, None))


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

