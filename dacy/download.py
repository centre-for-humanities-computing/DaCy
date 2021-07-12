"""
Functions for downloading DaCy models.
"""
import os
import shutil
from pathlib import Path
import urllib.request
from typing import Optional

from tqdm import tqdm


DEFAULT_CACHE_DIR = os.path.join(str(Path.home()), ".dacy")


dacy_small_000 = (
    "https://sciencedata.dk//shared/d845d4fef9ea165ee7bd6dd954b95de2?download"
)
dacy_medium_000 = (
    "https://sciencedata.dk//shared/c205edf59195583122d7213a3c26c077?download"
)
dacy_large_000 = (
    "https://sciencedata.dk//shared/0da7cb975b245d9e6574458c7c89dfd9?download"
)


models_url = {
    "da_dacy_small_tft-0.0.0": dacy_small_000,
    "da_dacy_medium_tft-0.0.0": dacy_medium_000,
    "da_dacy_large_tft-0.0.0": dacy_large_000,
}


def models() -> list:
    """
    Returns a list of valid DaCy models

    Returns:
        list: list of valid DaCy models
    """
    return list(models_url.keys())


def download_model(
    model: str,
    save_path: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Downloads a DaCy model to the specified save_path or to the default cache directory.

    Args:
        model (str): string indicating DaCy model, use dacy.models() to get a list of models
        save_path (str, optional): The path you want to save your model to. Defaults to None denoting the default cache directory. This can be found using using dacy.where_is_my_dacy().
        force (bool, optional): Should it download the model regardless of it already being present? Defaults to False.
        verbose (bool): Toggles the verbosity of the function. Defaults to True.

    Returns:
        True if the model is downloaded as intended

    Example:
        >>> download_model(model="da_dacy_medium_tft-0.0.0")
    """
    if model in {"small", "medium", "large"}:
        model = f"da_dacy_{model}_tft-0.0.0"
    if model not in models_url:
        raise ValueError(
            "The model is not available in DaCy. Please use dacy.models() to see a list of all models"
        )
    if save_path is None:
        save_path = DEFAULT_CACHE_DIR

    url = models_url[model]
    path = os.path.join(save_path, model)
    dl_path = os.path.join(save_path, "tmp.zip")
    if os.path.exists(path) and force is False:
        return True

    if verbose is True:
        print(f"\n[INFO] Downloading '{model}'")
    Path(save_path).mkdir(parents=True, exist_ok=True)
    download_url(url, dl_path)
    shutil.unpack_archive(dl_path, save_path)
    os.remove(dl_path)
    return True


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize=None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
