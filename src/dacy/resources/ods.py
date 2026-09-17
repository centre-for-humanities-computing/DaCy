"""Helper function for loading the ODS full-form list."""

import csv
import zipfile
from os import PathLike
from pathlib import Path

import pandas as pd

from ..download import DEFAULT_CACHE_DIR, download_url
from .constants import RESOURCES


def load_ods_fullforms(
    save_path: str | PathLike | None = None,
    redownload: bool = False,
) -> pd.DataFrame:
    """Loads the ODS full-form list, preserving original spelling and capitalization.

    Args:
        save_path: Directory in which to cache the ODS download. Defaults to
            the resources subfolder of dacy.where_is_my_dacy().
        redownload: Download again even if the file is cached. Defaults to False.

    Returns:
        pd.DataFrame: Columns form, headword, homograph, pos and id, all strings.
            Missing homograph numbers are empty strings. Multiple entries for
            the same form are retained.

    Example:
        >>> from dacy.resources import load_ods_fullforms
        >>> entries = load_ods_fullforms()
        >>> wordforms = set(entries["form"])
    """
    if save_path is None:
        save_path = Path(DEFAULT_CACHE_DIR) / "resources"
    save_path = Path(save_path) / "ods"
    dl_path = save_path / "ods-fullform.zip"

    if redownload or not dl_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        download_url(RESOURCES["ods"], str(dl_path))

    with zipfile.ZipFile(dl_path) as archive:
        filename = next(
            name
            for name in archive.namelist()
            if name.startswith("ods_fullforms_") and name.endswith(".csv")
        )
        with archive.open(filename) as source:
            return pd.read_csv(
                source,
                sep="\t",
                names=["form", "headword", "homograph", "pos", "id"],
                dtype=str,
                keep_default_na=False,
                quoting=csv.QUOTE_NONE,
                encoding="utf-8",
            )
