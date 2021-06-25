from dacy.augmenters import (
    create_char_swap_augmenter,
    create_remove_spacing_augmenter,
    create_char_random_augmenter,
    create_char_replace_augmenter,
)
from spacy.lang.da import Danish


def test_create_char_swap_augmenter():
    aug = create_char_swap_augmenter(doc_level=1, char_level=1)
    nlp = Danish()
    doc = nlp("qw")
    doc = aug(doc)
    assert doc[0] == "wq"


def test_create_remove_spacing_augmenter():
    aug = create_remove_spacing_augmenter(doc_level=1, spacing_level=1)
    nlp = Danish()
    doc = nlp("en sætning.")
    doc = aug(doc)
    assert doc.text == "ensætning."


def test_create_char_random_augmenter():
    aug = create_char_random_augmenter(doc_level=1, char_level=1)
    nlp = Danish()
    doc = nlp("en sætning.")
    doc = aug(doc)
    assert doc.text != "en sætning."


def test_create_char_replace_augmenter():
    aug = create_char_replace_augmenter(
        doc_level=1, char_level=1, replacement={"q": ["a", "b"]}
    )
    nlp = Danish()
    doc = nlp("q w")
    doc = aug(doc)
    assert doc[0] in ["a", "b"]
    assert doc[1] == "w"
