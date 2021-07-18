import dacy
from spacy.training import Example
from typing import List, Callable, Iterator


def test_tutorial():
    def doc_to_example(doc):
        return Example(doc, doc)

    nlp = dacy.load("da_dacy_small_tft-0.0.0")
    doc = nlp(
        "Peter Schmeichel mener også, at det danske landshold anno 2021 tilhører verdenstoppen og kan vinde den kommende kamp mod England."
    )
    example = doc_to_example(doc)

    from spacy.training.augment import create_lower_casing_augmenter
    from dacy.augmenters import (
        create_keyboard_augmenter,
        create_pers_augmenter,
        create_spacing_augmenter,
    )
    from dacy.datasets import danish_names

    lower_aug = create_lower_casing_augmenter(level=1)
    keyboard_05 = create_keyboard_augmenter(
        doc_level=1, char_level=0.05, keyboard="QWERTY_DA"
    )
    keyboard_15 = create_keyboard_augmenter(
        doc_level=1, char_level=0.15, keyboard="QWERTY_DA"
    )
    space_aug = create_spacing_augmenter(doc_level=1, spacing_level=0.4)

    for aug in [lower_aug, keyboard_05, keyboard_15, space_aug]:
        aug_example = next(aug(nlp, example))  # augment the example
        doc = aug_example.y  # extract the reference doc
        print(doc)

    for aug in [lower_aug, keyboard_05, keyboard_15, space_aug]:
        aug_example = next(aug(nlp, example))  # augment the example
        doc = aug_example.y  # extract the reference doc
        print(doc)

    print(danish_names().keys())
    print(danish_names()["first_name"][0:5])
    print(danish_names()["last_name"][0:5])

    def augment_texts(texts: List[str], augmenter: Callable) -> Iterator[Example]:
        """Takes a list of strings and yields augmented examples"""
        docs = nlp.pipe(texts)
        for doc in docs:
            ex = Example(doc, doc)
            aug = augmenter(nlp, ex)
            yield next(aug).y

    texts = [
        "Hans Christian Andersen var en dansk digter og forfatter",
        "1, 2, 3, Schmeichel er en mur",
        "Peter Schmeichel mener også, at det danske landshold anno 2021 tilhører verdenstoppen og kan vinde den kommende kamp mod England.",
    ]

    # Create a dictionary to use for name replacement
    dk_name_dict = danish_names()

    # force_pattern augments PER entities to fit the format and length of `patterns`. Patterns allows you to specificy arbitrary
    # combinations of "fn" (first names), "ln" (last names), "abb" (abbreviated to first character) and "abbpunct" (abbreviated
    # to first character + ".") separeated by ",". If keep_name=True, the augmenter will not change names, but if force_pattern_size
    # is True it will make them fit the length and potentially abbreviate names.
    pers_aug = create_pers_augmenter(
        dk_name_dict, force_pattern_size=True, keep_name=False, patterns=["fn,ln"]
    )
    augmented_docs = augment_texts(texts, pers_aug)
    for d in augmented_docs:
        print(d)

    # Here's an example with keep_name=True and force_pattern_size=False which simply abbreviates first names
    abb_aug = create_pers_augmenter(
        dk_name_dict, force_pattern_size=False, keep_name=True, patterns=["abbpunct"]
    )
    augmented_docs = augment_texts(texts, abb_aug)
    for d in augmented_docs:
        print(d)

    # patterns can also take a list of patterns to replace from (which can be weighted using the
    # patterns_prob argument. The pattern to use is sampled for each entity.
    # This setting is especially useful for finetuning models.
    multiple_pats = create_pers_augmenter(
        dk_name_dict,
        force_pattern_size=True,
        keep_name=False,
        patterns=["fn,ln", "abbpunct,ln", "fn,ln,ln,ln"],
    )
    augmented_docs = augment_texts(texts, multiple_pats)
    for d in augmented_docs:
        print(d)

    docs = nlp.pipe(texts)
    augmented_docs = augment_texts(texts, multiple_pats)

    # Check that the added/removed PER entities are still tagged as entities
    for doc, aug_doc in zip(docs, augmented_docs):
        print(doc.ents, "\t\t", aug_doc.ents)