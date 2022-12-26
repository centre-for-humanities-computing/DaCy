"""Functions for downloading DaCy models."""
import os
import shutil
from pathlib import Path
from typing import Optional

from tqdm import tqdm
from wasabi import msg

DEFAULT_CACHE_DIR = os.getenv("DACY_CACHE_DIR", os.path.join(str(Path.home()), ".dacy"))

models_url = {
    "da_dacy_small_tft-0.0.0": "https://sciencedata.dk//shared/d845d4fef9ea165ee7bd6dd954b95de2?download",
    "da_dacy_medium_tft-0.0.0": "https://sciencedata.dk//shared/c205edf59195583122d7213a3c26c077?download",
    "da_dacy_large_tft-0.0.0": "https://sciencedata.dk//shared/0da7cb975b245d9e6574458c7c89dfd9?download",
    "da_dacy_small_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_small_trf/resolve/main/da_dacy_small_trf-any-py3-none-any.whl",
    "da_dacy_medium_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_medium_trf/resolve/main/da_dacy_medium_trf-any-py3-none-any.whl",
    "da_dacy_large_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_large_trf/resolve/main/da_dacy_large_trf-any-py3-none-any.whl",
}


def models() -> list:
    """Returns a list of valid DaCy models.

    Returns:
        list: list of valid DaCy models
    """
    return list(models_url.keys())


def download_model(
    model: str,
    save_path: Optional[str] = None,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """Downloads and install a specified DaCy pipeline.

    Args:
        model (str): string indicating DaCy model, use dacy.models() to get a list of models
        save_path (str, optional): The path you want to save your model to. Is only used for DaCy models of v0.0.0 as later models are installed as modules to allow for better versioning. Defaults to None denoting the default cache directory. This can be found using using dacy.where_is_my_dacy().
        force (bool, optional): Should it download the model regardless of it already being present? Defaults to False.
        verbose (bool): Toggles the verbosity of the function. Defaults to True.

    Returns:
        a string of the model location

    Example:
        >>> download_model(model="da_dacy_medium_tft-0.0.0")
    """
    if model in {"small", "medium", "large"}:
        model = f"da_dacy_{model}_trf-0.1.0"

    if model not in models_url:
        raise ValueError(
            "The model is not available in DaCy. Please use dacy.models() to see a list of all models",
        )

    if int(model.split("-")[-1].split(".")[1]) < 1:  # model v. 0.0.0
        if save_path is None:
            save_path = DEFAULT_CACHE_DIR

        url = models_url[model]
        path = os.path.join(save_path, model)
        dl_path = os.path.join(save_path, "tmp.zip")
        if os.path.exists(path) and force is False:
            return path

        if verbose is True:
            msg.info(f"\nDownloading '{model}'")
        Path(save_path).mkdir(parents=True, exist_ok=True)
        download_url(url, dl_path)
        shutil.unpack_archive(dl_path, save_path)
        os.remove(dl_path)
        if verbose is True:
            msg.info(
                rf"\Model successfully downloaded, you can now load it using dacy.load({model})",
            )
        return path
    else:
        from spacy.util import get_installed_models

        mdl = model.split("-")[0]
        if mdl not in get_installed_models():
            install(models_url[model])
        return mdl


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize=None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: str) -> None:
    import urllib.request

    with DownloadProgressBar(
        unit="B",
        unit_scale=True,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def install(package):
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"],
    )
