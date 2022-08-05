from typing import Callable, Iterable, List

from spacy.tokens import Doc, Span
from spacy.training import Example


def no_misc_getter(doc, attr):
    spans = getattr(doc, attr)
    for span in spans:
        if span.label_ == "MISC":
            continue
        yield span


def add_iob(doc: Doc, iob: List[str]) -> Doc:
    """Add iob tags to Doc.

    Args:
        doc (Doc): A SpaCy doc
        iob (List[str]): a list of tokens on the IOB format

    Returns:
        Doc: A doc with the spans to the new IOB
    """
    ent = []
    for i, label in enumerate(iob):

        # turn OOB labels into spans
        if label == "O":
            continue
        iob_, ent_type = label.split("-")
        if (i - 1 >= 0 and iob_ == "I" and iob[i - 1] == "O") or (
            i == 0 and iob_ == "I"
        ):
            iob_ = "B"
        if iob_ == "B":
            start = i
        if i + 1 >= len(iob) or iob[i + 1].split("-")[0] != "I":
            ent.append(Span(doc, start, i + 1, label=ent_type))
    doc.set_ents(ent)
    return doc


def apply_on_multiple_examples(func: Callable) -> Callable:
    def inner(examples: Iterable[Example], **kwargs) -> Iterable[Example]:
        return [func(e, **kwargs) for e in examples]

    return inner
