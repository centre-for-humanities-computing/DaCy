"""Functions for downloading DaCy models."""
import os
from distutils.version import StrictVersion
from importlib.metadata import version
from pathlib import Path

from spacy.util import get_installed_models
from tqdm import tqdm

versions = ["1.1.2", "1.0.0", "1.3.3", "1.0.12", "1.0.2"]
versions.sort(key=StrictVersion)
DACY_DEFAULT_PATH = Path.home() / ".dacy"

DEFAULT_CACHE_DIR = os.getenv(
    "DACY_CACHE_DIR",
    DACY_DEFAULT_PATH,
)

models_url = {
    "da_dacy_small_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_small_trf/resolve/a3da03433d42538fca37847e3c73503d8e029088/da_dacy_small_trf-any-py3-none-any.whl",
    "da_dacy_medium_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_medium_trf/resolve/61a54ab9e9ab437f5c603c023d4238ecc5bb8eb5/da_dacy_medium_trf-any-py3-none-any.whl",
    "da_dacy_large_trf-0.1.0": "https://huggingface.co/chcaa/da_dacy_large_trf/resolve/5cfbb2bccf8e9509126e32fa3c537cc3c062aec2/da_dacy_large_trf-any-py3-none-any.whl",
    "small": None,
    "medium": None,
    "large": None,
    "da_dacy_small_ner_fine_grained-0.1.0": "https://huggingface.co/chcaa/da_dacy_small_ner_fine_grained/resolve/43fedc5a1b1c1d193f461d13225f217f2ced507d/da_dacy_small_ner_fine_grained-any-py3-none-any.whl",
    "da_dacy_medium_ner_fine_grained-0.1.0": "https://huggingface.co/chcaa/da_dacy_medium_ner_fine_grained/resolve/4bfc4397b720acdb6428d64f18e90bfd439c80fc/da_dacy_medium_ner_fine_grained-any-py3-none-any.whl",
    "da_dacy_large_ner_fine_grained-0.1.0": "https://huggingface.co/chcaa/da_dacy_large_ner_fine_grained/resolve/08f973a1ff57120268bf30d3b7e7c4656ed25a58/da_dacy_large_ner_fine_grained-any-py3-none-any.whl",
}


def get_latest_version(model: str) -> str:
    """Returns the latest version of a DaCy model.

    Args:
        model: string indicating the model

    Returns:
        str: latest version of the model
    """
    if model in {"small", "medium", "large"}:
        model = f"da_dacy_{model}_trf"
    versions = [mdl.split("-")[-1] for mdl in models_url if "ner_fine_grained" in mdl]
    versions = sorted(
        versions,
        key=lambda s: [int(u) for u in s.split(".")],
        reverse=True,
    )
    return versions[0]


def models() -> list[str]:
    """Returns a list of valid DaCy models.

    Returns:
        list: list of valid DaCy models
    """
    return list(models_url.keys())


class DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize=None) -> None:  # noqa
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


def install(package):  # noqa
    import subprocess
    import sys

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", package, "--no-deps"],
    )


def download_model(
    model: str,
    force: bool = False,
) -> str:
    """Downloads and install a specified DaCy pipeline.

    Args:
        model: string indicating DaCy model, use dacy.models() to get a list of
            models.
        force: Should it download the model regardless of it already
            being present? Defaults to False.

    Returns:
        a string of the model location

    Example:
        >>> download_model(model="da_dacy_medium_trf-0.1.0")
    """
    if model in {"small", "medium", "large"}:
        latest_version = get_latest_version(model)
        model = f"da_dacy_{model}_trf-{latest_version}"
    mdl_version = model.split("-")[-1]

    if model not in models_url:
        raise ValueError(
            f"The model '{model}' is not available in DaCy. Please use dacy.models() to see a"
            + " list of all models",
        )

    mdl = model.split("-")[0]
    if mdl in get_installed_models() and not force:
        if version(mdl) == mdl_version:
            return mdl
    else:
        install(models_url[model])
    return mdl
