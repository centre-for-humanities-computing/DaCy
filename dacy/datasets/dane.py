"""
This includes Danish datasets wrapped and read in in as a spacy corpus. This should allow for easier augmentation and model testing.
"""

import os
from pathlib import Path
from typing import Optional, Union, Tuple
from numpy.lib.npyio import save

import spacy
from danlp.datasets import DDT
from danlp.models import load_bert_ner_model
from spacy import displacy
from spacy.cli.convert import conllu_to_docs
from spacy.scorer import Scorer
from spacy.tokens import Doc, Span
from spacy.training import Corpus

from ..download import DEFAULT_CACHE_DIR


def dane(
    save_path: Optional[str] = None,
    predefined_splits: bool = True,
    redownload: bool = False,
    n_sents: int = 1,
) -> Union[Tuple[Corpus, Corpus, Corpus], Corpus]:
    """
    reads in DaNE as a spacy Corpus.

    Args:
        save_path (str, optional): The path which contain the dane dataset If it does not contain the dataset it
            is downloaded to the folder. Defaults to None corresponding to dacy.where_is_my_dacy() in the datasets subfolder.
        predefined_splits (bool, optional): If True returns the predifined splits in a tuple (train, dev, test)
            otherwise return one dataset. Defaults to True.
        redownload (bool, optional): Should the dataset be redownloaded. Defaults to False.
        n_sents (int, optional): Number of sentences per document. Only applied in datasets is downloaded. Defaults to 1.

    Returns:
        Union[Tuple[Corpus, Corpus, Corpus], Corpus]: Returns a spacy corpus or a tuple thereof if predefined splits is True.
    """
    if save_path is None:
        save_path = os.path.join(DEFAULT_CACHE_DIR, "datasets")

    if ("dane" not in os.listdir(save_path)) or (redownload is True):
        save_path = os.path.join(save_path, "dane")
        Path(save_path).mkdir(parents=True, exist_ok=True)

        ddt = DDT()
        train, dev, test = ddt.load_as_conllu(predefined_splits=True)
        all = ddt.load_as_conllu(predefined_splits=False)

        wpaths = [
            "dane_train.conllu",
            "dane_dev.conllu",
            "dane_test.conllu",
            "dane.conllu",
        ]

        for dat, wpath in zip([train, dev, test, all], wpaths):
            with open(wpath, "w") as f:
                test.write(f)

            # convert to spacy
            os.system(
                f"python -m spacy convert {wpath} {save_path} --converter conllu --merge-subtokens -n {n_sents}"
            )
            os.remove(wpath)

    if predefined_splits is False:
        return Corpus("corpus/dane/dane.spacy")
    else:
        train = Corpus("corpus/dane/dane_test.spacy")
        dev = Corpus("corpus/dane/dane_test.spacy")
        test = Corpus("corpus/dane/dane_test.spacy")
        return train, dev, test