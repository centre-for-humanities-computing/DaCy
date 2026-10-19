"""Helper functions for loading Danish dictionaries."""

import csv
import zipfile
from pathlib import Path

import pandas as pd

from ..download import DEFAULT_CACHE_DIR, download_url
from .constants import RESOURCES


def load_ods_fullforms(redownload: bool = False) -> pd.DataFrame:
    """Loads the full-form list from ODS, a historical Danish dictionary.

    `Ordbog over det danske Sprog (ODS)
    <https://korpus.dsl.dk/resources/details/ods-fullforms.html>`_ covers Danish
    from approximately 1700 to 1950. The August 2020 full-form list contains
    about 1.3 million entries, including inflected forms. Original spelling
    and capitalization are preserved.

    `DSL's terms of use <https://korpus.dsl.dk/resources/licences/dsl-open.html>`_
    allow reuse, modification and redistribution, including commercial use,
    but prohibit publishing a dictionary or a product competing with DSL's
    products. Attribution to DSL is requested. Downloading accepts these terms.

    Args:
        redownload: Download again even if the file is cached. Defaults to False.

    Returns:
        A DataFrame with five string columns.

        - form: The word form, including inflected forms.
        - headword: The dictionary headword associated with the form.
        - homograph: The number distinguishing dictionary entries with the same
          headword spelling; an empty string when absent.
        - pos: The part of speech, using ODS abbreviations.
        - id: The identifier of the dictionary entry in ODS.

        Multiple entries for the same form are retained.

    Example:
        Look up the inflected form "Hesten" ("the horse") and its headword
        "Hest" ("horse"):

        >>> from dacy.resources import load_ods_fullforms
        >>> entries = load_ods_fullforms()
        >>> print(entries.loc[entries["form"] == "Hesten"].to_string(index=False))
          form headword homograph pos       id
        Hesten     Hest           sb. 60137181
        >>> wordforms = set(entries["form"])
    """
    save_path = Path(DEFAULT_CACHE_DIR) / "resources" / "ods"
    dl_path = save_path / "ods-fullform.zip"

    if redownload or not dl_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)
        download_url(RESOURCES["ods"], dl_path)

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
