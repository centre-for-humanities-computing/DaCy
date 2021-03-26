from spacy.tokens import Doc

from dacy.readability import LIX_getter
import dacy

def test_LIX():
    Doc.set_extension("LIX", getter=LIX_getter)

    nlp = dacy.load("da_dacy_medium_tft-0.0.0")
    doc = nlp("Dette er en test tekst")
    doc._.LIX