import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from utils import process_file


def process_and_concat_raw_data(dir_path, output_file):
    csv_files = [os.path.join(dir_path, file_name) for file_name in os.listdir(dir_path) if file_name.endswith(".csv")]

    combined_data = pd.DataFrame()

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_file, file_path): file_path for file_path in csv_files}

        for future in tqdm(as_completed(futures), total=len(csv_files)):
            processed_data = future.result()
            combined_data = pd.concat([combined_data, processed_data], ignore_index=True)

    print(f"Saving data to {output_file}")
    combined_data.to_csv(output_file, index=False)
