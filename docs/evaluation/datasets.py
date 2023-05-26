from typing import Dict, List

import catalogue
import spacy
from datasets import load_dataset
from spacy.tokens import Doc
from spacy.training import Example

datasets = catalogue.create("dacy", "datasets")


@datasets.register("dane")
def dane() -> Dict[str, List[Example]]:
    from dacy.datasets import dane as _dane

    train, dev, test = _dane(splits=["train", "dev", "test"])  # type: ignore
    nlp_da = spacy.blank("da")

    datasets = {}
    for nam, split in zip(["train", "dev", "test"], [train, dev, test]):
        examples = list(split(nlp_da))
        datasets[nam] = examples

    return datasets


@datasets.register("dansk")
def dansk(**kwargs) -> Dict[str, List[Example]]:
    splits = ["train", "dev", "test"]

    if not Doc.has_extension("meta"):
        Doc.set_extension("meta", default={}, force=True)

    nlp = spacy.blank("da")

    def convert_to_doc(example):
        doc = Doc(nlp.vocab).from_json(example)
        # set metadata
        for k in ["dagw_source", "dagw_domain", "dagw_source_full"]:
            doc._.meta[k] = example[k]
        return doc

    dataset = {}
    for split in splits:
        ds = load_dataset("chcaa/DANSK", split=split, **kwargs)
        docs = [convert_to_doc(example) for example in ds]
        examples = [Example(doc, doc) for doc in docs]
        dataset[split] = examples

    return dataset
