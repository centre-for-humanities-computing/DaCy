"""
Derived from: https://github.com/explosion/projects/blob/v3/tutorials/nel_emerson/scripts/create_kb.py
"""
import json
import ssl
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

import spacy
import typer
from spacy.kb import InMemoryLookupKB
from spacy.language import Language
from spacy.tokens import DocBin
from wikidata.client import Client

ssl._create_default_https_context = ssl._create_unverified_context

project_path = Path(__file__).parent.parent


def main(
    vector_model_path: str,
    save_path_kb: Path = project_path / "assets/daned/knowledge_base.kb",
):
    """Step 1: create the Knowledge Base in spaCy and write it to file"""

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.load(vector_model_path, exclude="tagger, lemmatizer")
    # get vector size from the model
    s = nlp("text")
    vector_size = s.vector.shape[0]
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=vector_size)

    qid2desc = _load_qid_to_description()
    qid2probs = _load_qid_to_probs()
    ents = list(_load_ents_from_data(nlp))

    qid2ent = defaultdict(lambda: defaultdict(int))
    for ent in ents:
        qid2ent[ent.kb_id_][ent.text] += 1

    qid2alias = {
        qid: defaultdict(int) for qid in qid2desc.keys()
    }  # frequency to be estimates from the data
    for qid, names in qid2probs.items():
        for type, name in names:
            occurances = qid2ent[qid].get(name, 0)
            qid2alias[qid][name] += (
                occurances + 1
            )  # add 1 for each entry (will normalize the distribution)

    for qid, desc in qid2desc.items():
        if not desc.strip():
            print(f"Fetching description for {qid}")
            desc = _fetch_wikidata_description(qid)
        if not desc.strip():
            print(f"Could not fetch description for {qid}")
            desc = "No description."
            # or skip this entity?
            # but it might be a reasonable entity to have
            # even if it has no description
            # or create a random vector or vector of zeros
            # would not be well-positioned in the vector space
            # but could be learned, but so would the current
            # vector

        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        # Set arbitrary value for frequency
        qid_freq = sum(qid2alias[qid].values())
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=qid_freq)

    alias2qid = defaultdict(lambda: defaultdict(int))
    for qid, names in qid2alias.items():
        for name, freq in names.items():
            alias2qid[name][qid] += freq

    for name, qids in alias2qid.items():
        total = sum(qids.values())
        probs = [freq / total for qid, freq in qids.items()]
        kb.add_alias(alias=name, entities=qids.keys(), probabilities=probs)

    kb.to_disk(save_path_kb)


def _fetch_wikidata_description(qid: str):
    """
    Fetch the description of a Wikidata item.
    """
    client = Client()
    item = client.get(qid, load=True)  # type: ignore
    return item.description.get("da", item.description.get("en", ""))  # type: ignore


def _load_ents_from_data(nlp: Language, splits: List[str] = ["dev", "train"]):
    """
    Load in the data from the training set.
    """
    # load docbin
    cdt = project_path / "corpus/cdt"
    for split in splits:
        docbin_path = cdt / f"{split}.spacy"
        doc_bin = DocBin().from_disk(docbin_path)
        docs = list(doc_bin.get_docs(nlp.vocab))
        for doc in docs:
            for ent in doc.ents:
                yield ent


def _load_qid_to_description():
    """
    Location of "training/v0.2.0/assets/daned/desc.json"
    """
    desc_path = project_path / "assets/daned/desc.json"

    with open(desc_path, "r", encoding="utf8") as f:
        qid2description = json.load(f)
        return qid2description


def _load_qid_to_probs():
    """
    Location of "training/v0.2.0/assets/daned/probs.json"
    """
    probs_path = project_path / "assets/daned/props.json"

    with open(probs_path, "r", encoding="utf8") as f:
        qid2probs = json.load(f)
        return qid2probs


# def _load_qid_to_label(qids: Iterable[str]):
#     """
#     Location of "training/v0.2.0/assets/daned/labels.json" if it doesn't exist, create it.
#     """
#     label_path = project_path / "assets/daned/labels.json"
#     names = dict()

#     if label_path.exists():
#         with open(label_path, "r", encoding="utf8") as f:
#             qid2label = json.load(f)
#         return qid2label

#     # get names of QIDs from wikidata
#     client = Client()
#     for qid in qids:
#         entity = client.get(qid, load=True)  # type: ignore
#         # get danish label
#         label = entity.label.get("da", entity.label.get("en", None))  # type: ignore
#         if label:
#             names[qid] = label
#         else:
#             raise ValueError(f"Could not find label for {qid}")
#     with open(label_path, "w", encoding="utf8") as f:
#         json.dump(names, f)
#     return names

if __name__ == "__main__":
    typer.run(main)
