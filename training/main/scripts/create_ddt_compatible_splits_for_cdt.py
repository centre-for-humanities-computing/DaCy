"""
A script for creating splits for the copenhagen dependency treebank which align with DDT.
"""

import json
from pathlib import Path

from conllu import parse


def load_cdt(
    assets_path=Path(__file__).parent.parent / "assets",
):
    """
    Load the copenhagen dependency treebank / DaCoref dataset
    """
    cdt_path = assets_path / "dacoref" / "CDT_coref.conllu"
    with cdt_path.open(encoding="utf-8") as f:
        text = f.read()

    sentences = parse(
        text,
        fields=[
            "id",
            "form",
            "lemma",
            "upos",
            "xpos",
            "feats",
            "head",
            "deprel",
            "deps",
            "misc",
            "coref_id",
            "coref_rel",
            "doc_id",
            "qid",
        ],
    )

    split_ids = {"train": [], "dev": [], "test": []}

    for split in split_ids:
        split_path = assets_path / "dacoref" / f"CDT_{split}_ids.json"
        with split_path.open(encoding="utf-8") as f:
            split_ids[split] = json.load(f)

    return sentences, split_ids


sents, split_ids = load_cdt()


doc_ids = {}

_doc_id = "nan"
for sent in sents:
    if "newdoc id" in sent.metadata:
        _doc_id = sent.metadata["newdoc id"]
    sent_id = sent.metadata["sent_id"]
    split = sent_id.split("-")[0]
    if _doc_id not in doc_ids:
        doc_ids[_doc_id] = []
    doc_ids[_doc_id].append(split)

test_docs = []
for s in split_ids["train"]:
    if s not in doc_ids:
        print("not in doc_ids")
        break

    if "test" in doc_ids[s]:
        test_docs.append(s)

new_split = {"test": test_docs}
dev_no_test = [s for s in split_ids["dev"] if s not in test_docs]
new_split["dev"] = dev_no_test
train_no_test = [s for s in split_ids["train"] if s not in test_docs]
test_no_test = [s for s in split_ids["test"] if s not in test_docs]
new_split["train"] = train_no_test + test_no_test

with open("assets/CDT_ddt_compatible_splits.json", "w") as f:
    json.dump(new_split, f, indent=2)
