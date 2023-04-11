import dacy
import pytest
from dacy.load import load


@pytest.mark.parametrize(
    "model",
    [
        ("da_dacy_small_trf-0.1.0"),
        ("small"),
    ],
)
def test_load(model: str):
    nlp = load(model)
    nlp("Dette er en test tekst")


def test_models():
    print(dacy.models())


def test_where_is_my_dacy():
    print(dacy.where_is_my_dacy())
