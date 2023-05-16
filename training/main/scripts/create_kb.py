"""
Initially inspired by: https://github.com/explosion/projects/blob/v3/tutorials/nel_emerson/scripts/create_kb.py
"""
import json
import ssl
from collections import defaultdict
from pathlib import Path
from typing import List

import numpy as np
import spacy
import typer
from spacy.kb import InMemoryLookupKB
from spacy.language import Language
from spacy.tokens import DocBin
from wasabi import msg
from wikidata.client import Client

ssl._create_default_https_context = ssl._create_unverified_context

project_path = Path(__file__).parent.parent


def main(
    trf_name: str,
    save_path_kb: Path = project_path / "assets/knowledge_bases/knowledge_base.kb",
    langs_to_fetch: List[str] = ["da", "en"],
):
    """Step 1: create the Knowledge Base in spaCy and write it to file"""
    spacy.require_gpu()  # type: ignore

    # First: create a simpel model from a model with an NER component
    # To ensure we get the correct entities for this demo, add a simple entity_ruler as well.
    nlp = spacy.blank("da")  # empty Danish pipeline
    # create the config with the name of your model
    # values omitted will get default values
    config = {
        "model": {
            "@architectures": "spacy-transformers.TransformerModel.v3",
            "name": trf_name,  # XXX customize this bit
        },
    }
    nlp.add_pipe("transformer", config=config)
    nlp.initialize()  # XXX don't forget this step!
    # nlp = spacy.load(vector_model_path, exclude="tagger, lemmatizer")
    # get vector size from the model
    doc = nlp("text")
    # vector_size = doc.vector.shape[0] # for non-transformer models
    vector_size = doc._.trf_data.tensors[0].shape[-1]
    kb = InMemoryLookupKB(vocab=nlp.vocab, entity_vector_length=vector_size)

    qid2desc = _load_qid_to_description()
    # qid2probs = _load_qid_to_probs()
    msg.info("Loading entities from data")
    ents = list(_load_ents_from_data(nlp))
    msg.good(f"Loaded {len(ents)} entities from data")
    qid2ent = defaultdict(lambda: defaultdict(int))
    for ent in ents:
        qid2ent[ent.kb_id_][ent.text] += 1

    msg.info("Starting to fetch aliases")
    qid2alias = {
        qid: defaultdict(lambda: 0) for qid in qid2desc
    }  # frequency to be estimates from the data but start with 1 to normalize the distribution
    for qid in qid2alias:
        # get occurance from the data
        for name, occurances in qid2ent[qid].items():
            qid2alias[qid][name] += occurances + 1
        # get aliases from wikidata
        aliases = _fetch_wikidata_aliases(qid, langs=langs_to_fetch)
        for alias in aliases:
            qid2alias[qid][alias] += 1
    msg.good("Finished fetching aliases")

    msg.info("Starting to creating embeddings for KB")
    for qid, desc in qid2desc.items():
        fetch_desc = _fetch_wikidata_description(
            qid,
        )  # just always fetch the description
        if fetch_desc.strip():  # if it is not empty
            desc = fetch_desc  # newer is probably better
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

        desc_doc = nlp(desc)  # not very efficient, but fast enough
        # desc_enc = desc_doc.vector # for non-transformer models
        # take the mean of the transformer vectors
        batch_tensor = desc_doc._.trf_data.tensors[0]
        desc_enc = np.mean(batch_tensor[0], axis=0)

        # Set arbitrary value for frequency
        qid_freq = sum(qid2alias[qid].values())
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=qid_freq)
    msg.good("Finished creating embeddings for KB")

    msg.info("Starting to add aliases to KB")
    alias2qid = defaultdict(lambda: defaultdict(int))
    for qid, names in qid2alias.items():
        for name, freq in names.items():
            alias2qid[name][qid] += freq

    for name, qids in alias2qid.items():
        total = sum(qids.values())
        probs = [freq / total for qid, freq in qids.items()]
        kb.add_alias(alias=name, entities=qids.keys(), probabilities=probs)
    msg.good("Finished adding aliases to KB")

    msg.info("Starting to add entities to KB")
    kb.to_disk(save_path_kb)
    msg.good(f"Finished adding entities to KB and saved to {save_path_kb}")


def _fetch_wikidata_description(qid: str):
    """
    Fetch the description of a Wikidata item.
    """
    client = Client()
    try:
        item = client.get(qid, load=True)  # type: ignore
    except Exception:
        print(f"Could not load {qid}")
        return ""
    return item.description.get("da", item.description.get("en", ""))  # type: ignore


def _fetch_wikidata_aliases(qid, langs: List[str] = ["da", "en"]):
    """
    Fetch the aliases of a Wikidata item.
    """
    client = Client()
    try:
        item = client.get(qid, load=True)  # type: ignore
    except Exception:
        print(f"Could not load {qid}")
        return []
    aliases_dict = item.data.get("aliases", {})  # type: ignore
    aliases = []
    for lang in langs:
        aliases += [a["value"] for a in aliases_dict.get(lang, [])]  # type: ignore
    return aliases


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

    with open(desc_path, encoding="utf8") as f:
        qid2description = json.load(f)
        return qid2description


def _load_qid_to_probs():
    """
    Location of "training/v0.2.0/assets/daned/probs.json"
    """
    probs_path = project_path / "assets/daned/props.json"

    with open(probs_path, encoding="utf8") as f:
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
