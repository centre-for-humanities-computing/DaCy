"""
Functionality for extracting readability measures from text.
"""

from spacy.tokens import Doc


def LIX_getter(doc: Doc) -> float:
    """
    Extract the LIX score from a doc

    Args:
        doc (Doc): A SpaCy document

    Returns:
        float: the LIX score for the document

    Example:
        >>> from spacy.tokens import Doc
        >>> Doc.set_extension("LIX", getter=dacy.readability.LIX_getter)
        >>> doc = nlp("Dette er en simpel tekst")
        >>> doc._.LIX  # extrac the LIX score from your document
    """
    O = len(doc)
    P = len(list(doc.sents))
    L = len([t for t in doc if len(t) > 6])

    LIX = O / P + L * 100 / O
    return LIX
