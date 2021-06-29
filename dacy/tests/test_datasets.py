
import dacy

from spacy.training import Corpus
from spacy.lang.da import Danish
from spacy.training import Example

from dacy.datasets import dane
test = dane(splits=["test"])

def test_dane():
    train, dev, test = dane()
    all_ = dacy.datasets.dane(splits=["all"])
    for d in [train, dev, test, all_]:
        assert isinstance(d, Corpus)

    nlp = Danish()
    examples = list(test(nlp)) # check if it read as intended
    assert isinstance(examples[0], Example)
