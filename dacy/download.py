import os
import shutil
from pathlib import Path
import urllib.request

from tqdm import tqdm


DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".dacy")


dacy_small_000 = "https://sciencedata.dk//shared/d845d4fef9ea165ee7bd6dd954b95de2?download"
dacy_medium_000 = "https://sciencedata.dk//shared/c205edf59195583122d7213a3c26c077?download"
dacy_large_000 = "https://sciencedata.dk//shared/0da7cb975b245d9e6574458c7c89dfd9?download"


models_url = {
    "da_dacy_small_tft-0.0.0": dacy_small_000,
    "da_dacy_medium_tft-0.0.0": dacy_medium_000,
    "da_dacy_large_tft-0.0.0": dacy_large_000,
}


def models():
    return list(models_url.keys())


def where_is_my_dacy():
    return DEFAULT_CACHE_DIR


def extract_all(archives, extract_path):
    for filename in archives:
        shutil.unpack_archive(filename, extract_path)


def download_model(model: str, save_path: str = DEFAULT_CACHE_DIR):
    """
    model (str): use models() to see all available models

    Examples:
    download_model(model="da_dacy_medium_tft-0.0.0")
    """
    if model not in models_url:
        raise ValueError(
            "The model is not available in DaCy. Please use dacy_models() to see a list of all models"
        )
    url = models_url[model]
    path = os.path.join(save_path, model)
    dl_path = os.path.join(save_path, "tmp.zip")
    if os.path.exists(path):
        return True

    Path(save_path).mkdir(parents=True, exist_ok=True)
    download_url(url, dl_path)
    shutil.unpack_archive(dl_path, save_path)
    os.remove(dl_path)


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

