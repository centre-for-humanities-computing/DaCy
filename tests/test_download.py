import pytest

import dacy
from dacy.load import load


def test_load():
    nlp = load("da_dacy_small_ner_fine_grained-0.1.0")
    nlp("Dette er en test tekst")


def test_load_coref_model_requires_extra():
    try:
        import spacy_experimental  # type: ignore # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("spacy-experimental is installed; coref guard is not triggered")

    with pytest.raises(ImportError, match="dacy\\[coref\\]"):
        dacy.download_model("small")


def test_models():
    print(dacy.models())


def test_where_is_my_dacy():
    print(dacy.where_is_my_dacy())
