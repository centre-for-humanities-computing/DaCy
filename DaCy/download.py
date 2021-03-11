import os
import shutil
from pathlib import Path
import urllib.request

from tqdm import tqdm

DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".dacy")


dacy_medium_000 = "blank"
dacy_large_000 = "blank"


models_url = {
    "dacy_medium_tft-0.0.0": dacy_medium_000,
    "dacy_large_tft-0.0.0": dacy_large_000,
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
    """
    if model not in models_url:
        raise ValueError(
            "The model is not available in DaCy. Please use dacy_models() to see a list of all models"
        )
    url = models_url[model]
    path = os.path.join(save_path, model)
    dl_path = os.path.join(save_path, "tmp.tar.gz")
    if os.path.exists(path):
        return True
    download_url(url, dl_path)
    shutil.unpack_archive(dl_path, save_path)
    
    return True


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
