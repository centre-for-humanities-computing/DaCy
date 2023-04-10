"""This includes the DaNE dataset wrapped and read in as a SpaCy corpus."""

import shutil
import subprocess
import sys
from os import PathLike
from pathlib import Path
from typing import List, Optional, Union

from spacy.training import Corpus

from ..download import DEFAULT_CACHE_DIR, download_url
from .constants import DATASETS


def dane(  # noqa
    save_path: Optional[PathLike] = None,
    splits: List[str] = ["train", "dev", "test"],  # noqa
    redownload: bool = False,
    n_sents: int = 1,
    open_unverified_connection: bool = False,
    **kwargs,  # noqa
) -> Union[List[Corpus], Corpus]:
    """Reads the DaNE dataset as a spacy Corpus.

    Args:
        save_path (str, optional): Path to the DaNE dataset If it does not contain the
            dataset it is downloaded to the folder. Defaults to None corresponding to
            dacy.where_is_my_dacy() in the datasets subfolder.
        splits (List[str], optional): Which splits of the dataset should be returned.
            Possible options include "train", "dev", "test", "all". Defaults to
            ["train", "dev", "test"].
        redownload (bool, optional): Should the dataset be redownloaded. Defaults to
            False.
        n_sents (int, optional): Number of sentences per document. Only applied if the
            dataset is downloaded. Defaults to 1.
        open_unverified_connection (bool, optional): Should you download from an
            unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of
            whether it already exists. Defaults to False.

    Returns:
        Union[List[Corpus], Corpus]: Returns a SpaCy corpus or a list thereof.

    Example:
        >>> from dacy.datasets import dane
        >>> train, dev, test = dane(splits=["train", "dev", "test"])
    """
    if open_unverified_connection:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    if save_path is None:
        save_path_ = Path(DEFAULT_CACHE_DIR) / "datasets"
    else:
        save_path_ = Path(save_path)
    save_path = save_path_ / "dane"

    if redownload is True or (not save_path.exists()):
        save_path.mkdir(parents=True, exist_ok=True)
        dl_path = save_path / "dane.zip"
        download_url(DATASETS["dane"], str(dl_path))
        shutil.unpack_archive(dl_path, save_path)
        dl_path.unlink()

    wpaths = [
        "dane_train.conllu",
        "dane_dev.conllu",
        "dane_test.conllu",
        "dane.conllu",
    ]

    for _wpath in wpaths:
        wpath = save_path / _wpath
        cpath = save_path / (wpath.stem + f"_{n_sents}")

        if cpath.with_suffix(".spacy").is_file():
            continue
        cpath = cpath.with_suffix(".conllu")
        shutil.copyfile(wpath, cpath)
        # convert to spacy
        subprocess.run(
            [
                sys.executable,
                "-m",
                "spacy",
                "convert",
                cpath,
                save_path,  # type: ignore
                "--converter",
                "conllu",
                "--merge-subtokens",
                "-n",
                str(n_sents),
            ],
            check=True,
        )
        cpath.unlink()
    if isinstance(splits, str):  # type: ignore
        splits = [splits]  # type: ignore
    corpora = []
    paths = {
        "all": f"dane_{n_sents}.spacy",
        "test": f"dane_test_{n_sents}.spacy",
        "dev": f"dane_dev_{n_sents}.spacy",
        "train": f"dane_train_{n_sents}.spacy",
    }

    for split in splits:
        corpora.append(Corpus(save_path / paths[split]))  # type: ignore
    if len(corpora) == 1:
        return corpora[0]
    return corpora
