"""
Merge CDT documents with the DDT sentences not covered by CDT (per split).

Reads corpus/cdt_ddt/data.spacy (output of combine.py), which holds both
CDT documents (with a doc_id) and single-sentence DDT/DaNE docs without a
CDT counterpart (no doc_id). The remaining DDT docs are assigned to
train/dev/test from the prefix of their sent_id (dev2/test2 are merged
into dev/test). They are then appended to the corresponding CDT split
from corpus/cdt/{split}.spacy.

Adapted from code (Kenneth) in training_0.2.0/main/scripts/merge_ddt_sents_with_cdt.py

Outputs:
    corpus/cdt_ddt/{train,dev,test}.spacy

Run via `spacy project run merge_cdt_ddt`. Paths are resolved relative
to the project root, assumed to be the parent of this script's folder.
"""
from pathlib import Path

import spacy
from spacy.tokens import DocBin, Doc

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"

nlp = spacy.blank("da")

# set doc extensions
Doc.set_extension("doc_id", default=None)
Doc.set_extension("sent_id", default=None)

# load data.spacy
cdt_ddt_file = CORPUS_DIR / "cdt_ddt" / "data.spacy"
db = DocBin().from_disk(cdt_ddt_file)

# get docs from DocBin
docs = list(db.get_docs(nlp.vocab))

# get docs with and without doc_ids
# docs with doc_ids are already used in training data
# docs without doc_ids are not
cdt_data = [doc for doc in docs if doc._.doc_id is not None]
remaining_data = [doc for doc in docs if doc._.doc_id is None]

# verify data sizes
print(f"cdt: {len(cdt_data)}, remaining: {len(remaining_data)}, total: {len(docs)}")

remaining_splits = {"train": [], "dev": [], "test": []}

# there's apparently both dev2 and test2 in the data
# we merge those
split_mapping = {"dev2": "dev", "test2": "test"}

# split remaining data
for doc in remaining_data:
    split = doc._.sent_id.split("-")[0]
    split = split_mapping.get(split, split)    
    remaining_splits[split].append(doc)

# verify split sizes
print(f"train: {len(remaining_splits['train'])}, dev: {len(remaining_splits['dev'])}, test: {len(remaining_splits['test'])}")

# load split cdt data alone
cdt_files = [CORPUS_DIR / "cdt" / f"{split}.spacy" for split in ["train", "dev", "test"]]

# combine cdt and newly split remaining data
for split, file in zip(["train", "dev", "test"], cdt_files):
    db = DocBin().from_disk(file)
    docs = list(db.get_docs(nlp.vocab))
    docs = docs + remaining_splits[split]
    combined_db = DocBin(store_user_data=True)
    for doc in docs:
        combined_db.add(doc)
    combined_db.to_disk(CORPUS_DIR / "cdt_ddt" / f"{split}.spacy")