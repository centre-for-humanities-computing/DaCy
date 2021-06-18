"""
This includes Danish datasets wrapped and read in in as a spacy corpus. This should allow for easier augmentation and model testing.
"""

import os
from pathlib import Path
import shutil
from typing import Optional, Union, Tuple

from danlp.datasets import DDT
from spacy.training import Corpus

from ..download import DEFAULT_CACHE_DIR, download_url
from .constants import DATASETS


def dane(
    save_path: Optional[str] = None,
    predefined_splits: bool = True,
    redownload: bool = False,
    n_sents: int = 1,
) -> Union[Tuple[Corpus, Corpus, Corpus], Corpus]:
    """
    reads the DaNE dataset as a spacy Corpus.

    Args:
        save_path (str, optional): The path which contain the dane dataset If it does not contain the dataset it
            is downloaded to the folder. Defaults to None corresponding to dacy.where_is_my_dacy() in the datasets subfolder.
        predefined_splits (bool, optional): If True returns the predifined splits in a tuple (train, dev, test)
            otherwise return one dataset. Defaults to True.
        redownload (bool, optional): Should the dataset be redownloaded. Defaults to False.
        n_sents (int, optional): Number of sentences per document. Only applied in datasets is downloaded. Defaults to 1.

    Returns:
        Union[Tuple[Corpus, Corpus, Corpus], Corpus]: Returns a spacy corpus or a tuple thereof if predefined splits is True.

    Example:
        >>> import dacy
        >>> train, dev, test = dacy.datasets.dane(predefined_splits=True)
    """
    if save_path is None:
        save_path = os.path.join(DEFAULT_CACHE_DIR, "datasets", "dane")

    if (
        (not os.path.isdir(save_path))
        or ("dane" not in os.listdir(save_path))
        or (redownload is True)
    ):
        Path(save_path).mkdir(parents=True, exist_ok=True)

        dl_path = os.path.join(save_path, "dane.zip")
        download_url(DATASETS["dane"], dl_path)
        shutil.unpack_archive(dl_path, save_path)
        os.remove(dl_path)

    wpaths = [
        "dane_train.conllu",
        "dane_dev.conllu",
        "dane_test.conllu",
        "dane.conllu",
    ]

    for wpath in wpaths:
        wpath = os.path.join(save_path, wpath)
        cpath = wpath[:-7] + f"_{n_sents}"
        if os.path.isfile(cpath + ".spacy"):
            continue
        cpath += ".conllu"
        shutil.copyfile(wpath, cpath)
        # convert to spacy
        os.system(
            f"python -m spacy convert {cpath} {save_path} --converter conllu --merge-subtokens -n {n_sents}"
        )
        os.remove(cpath)

    if predefined_splits is False:
        return Corpus(os.path.join(save_path, f"dane_{n_sents}.spacy"))
    else:
        train = Corpus(os.path.join(save_path, f"dane_train_{n_sents}.spacy"))
        dev = Corpus(os.path.join(save_path, f"dane_dev_{n_sents}.spacy"))
        test = Corpus(os.path.join(save_path, f"dane_test_{n_sents}.spacy"))
        return train, dev, test
