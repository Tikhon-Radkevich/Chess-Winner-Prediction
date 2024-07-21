import os

import requests
from tqdm import tqdm


def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    filename = url.split("/")[-1]
    file_path = os.path.join(dest_folder, filename)

    if os.path.exists(file_path):
        print(f"File already exists at: {os.path.abspath(file_path)}")
        return file_path

    print(f"Starting downloading {url}")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024

        with open(file_path, "wb") as out_file:
            with tqdm(total=total_size, unit="iB", unit_scale=True) as pbar:
                for data in response.iter_content(block_size):
                    pbar.update(len(data))
                    out_file.write(data)

    print(f"File downloaded to: {os.path.abspath(file_path)}")
    return file_path
