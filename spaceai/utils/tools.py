import os
import zipfile
from typing import Optional

import requests  # type: ignore[import-untyped]
from tqdm import tqdm


def download_and_extract_zip(url: str, extract_to: str, cleanup: bool = False):
    filename = download_file(url)
    extract_zip(filename, extract_to, cleanup)


def download_file(url: str, to: Optional[str] = None):
    if to is None:
        local_filename = url.split("/")[-1]
    else:
        local_filename = to

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        with open(local_filename, "wb") as f, tqdm(
            desc=local_filename,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in r.iter_content(chunk_size=block_size):
                f.write(chunk)
                bar.update(len(chunk))
    return local_filename


def extract_zip(filename: str, extract_to: str, cleanup: bool = False):
    with zipfile.ZipFile(filename, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    if cleanup:
        os.remove(filename)
