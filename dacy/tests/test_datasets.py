
import dacy

from spacy.training import Corpus

def test_dane():
    train, dev, test = dacy.datasets.dane(predefined_splits=True)
    all = dacy.datasets.dane(predefined_splits=False)
    for d in [train, dev, test, all]:
        assert isinstance(d, Corpus)


