

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
