"""This includes the DaNE dataset wrapped and read in as a SpaCy corpus."""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

from spacy.training import Corpus

from ..download import DEFAULT_CACHE_DIR, download_url
from .constants import DATASETS


def dane(  # noqa
    save_path: Optional[str] = None,
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
        save_path_ = os.path.join(DEFAULT_CACHE_DIR, "datasets")  # noqa
    else:
        save_path_ = save_path
    save_path = os.path.join(save_path_, "dane")  # noqa

    if (
        (not os.path.isdir(save_path))  # noqa
        or ("dane" not in os.listdir(save_path_))
        or (redownload is True)
    ):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        dl_path = os.path.join(save_path, "dane.zip")  # noqa
        download_url(DATASETS["dane"], dl_path)
        shutil.unpack_archive(dl_path, save_path)
        os.remove(dl_path)  # noqa

    wpaths = [
        "dane_train.conllu",
        "dane_dev.conllu",
        "dane_test.conllu",
        "dane.conllu",
    ]

    for wpath in wpaths:
        wpath = os.path.join(save_path, wpath)  # noqa
        cpath = wpath[:-7] + f"_{n_sents}"
        if os.path.isfile(cpath + ".spacy"):  # noqa
            continue
        cpath += ".conllu"
        shutil.copyfile(wpath, cpath)
        # convert to spacy
        os.system(
            f"python -m spacy convert {cpath} {save_path} --converter conllu "
            + f"--merge-subtokens -n {n_sents}",
        )
        os.remove(cpath)  # noqa

    if isinstance(splits, str):
        splits = [splits]
    corpora = []
    paths = {
        "all": f"dane_{n_sents}.spacy",
        "test": f"dane_test_{n_sents}.spacy",
        "dev": f"dane_dev_{n_sents}.spacy",
        "train": f"dane_train_{n_sents}.spacy",
    }

    for split in splits:
        corpora.append(Corpus(os.path.join(save_path, paths[split])))  # noqa
    if len(corpora) == 1:
        return corpora[0]
    return corpora
