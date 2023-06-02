import random
from typing import Any, Dict, List

import augmenty
import catalogue
import numpy as np
import spacy
from datasets import load_dataset
from spacy.tokens import Doc
from spacy.training import Example

from .augmentations import get_gender_bias_augmenters, get_robustness_augmenters

datasets = catalogue.create("dacy", "datasets")


@datasets.register("dane")
def dane() -> Dict[str, List[Example]]:
    from dacy.datasets import dane as _dane

    train, dev, test = _dane(splits=["train", "dev", "test"])  # type: ignore
    nlp_da = spacy.blank("da")

    datasets = {}
    for nam, split in zip(["train", "dev", "test"], [train, dev, test]):  # type: ignore
        examples = list(split(nlp_da))
        datasets[nam] = examples

    return datasets


def augment_dataset(
    dataset: str,
    augmenters: dict,
    n_rep: int = 20,
    split: str = "test",
) -> List[Example]:
    # ensure seed
    random.seed(42)
    np.random.seed(42)

    nlp_da = spacy.blank("da")
    _dataset = datasets.get(dataset)
    ds_split = _dataset()[split]
    docs = [example.reference for example in ds_split]

    if not Doc.has_extension("meta"):
        Doc.set_extension("meta", default={}, force=True)

    # augment
    aug_docs = []
    for aug_name, aug in augmenters.items():
        for i in range(n_rep):
            _aug_docs = list(augmenty.docs(docs, augmenter=aug, nlp=nlp_da))
            for doc in _aug_docs:
                doc._.meta["augmenter"] = aug_name
                doc._.meta["n_rep"] = i
            aug_docs.extend(_aug_docs)

    # convert to examples
    examples = [Example(doc, doc) for doc in aug_docs]
    return examples


@datasets.register("gender_bias_dane")
def dane_gender_bias() -> Dict[str, List[Example]]:
    return {"test": augment_dataset("dane", augmenters=get_gender_bias_augmenters())}


@datasets.register("robustness_dane")
def dane_robustness() -> Dict[str, List[Example]]:
    return {"test": augment_dataset("dane", augmenters=get_robustness_augmenters())}


@datasets.register("dansk")
def dansk(**kwargs: Any) -> Dict[str, List[Example]]:
    splits = ["train", "dev", "test"]

    if not Doc.has_extension("meta"):
        Doc.set_extension("meta", default={}, force=True)

    nlp = spacy.blank("da")

    def convert_to_doc(example: Dict) -> Doc:
        doc = Doc(nlp.vocab).from_json(example)
        # set metadata
        for k in ["dagw_source", "dagw_domain", "dagw_source_full"]:
            doc._.meta[k] = example[k]
        return doc

    dataset = {}
    for split in splits:
        ds = load_dataset("chcaa/DANSK", split=split, **kwargs)
        docs = [convert_to_doc(example) for example in ds]  # type: ignore
        examples = [Example(doc, doc) for doc in docs]
        dataset[split] = examples

    return dataset
