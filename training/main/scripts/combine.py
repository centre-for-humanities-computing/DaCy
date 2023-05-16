"""
TODO:
- Save data to HF datasets
- save as docbin using # docbin.to_disk(path, store_user_data=True)
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import spacy
from conllu import parse
from spacy.tokens import Doc, DocBin, Span, Token
from spacy.training.corpus import Corpus

file_path = Path(__file__)
assets_path = file_path.parent.parent / "assets"
corpus_path = file_path.parent.parent / "corpus"

Doc.set_extension("domain", default=None)
Doc.set_extension("sent_id", default=None)
Doc.set_extension("sent_ids", default=None)
Doc.set_extension("conllu", default=None)
Doc.set_extension("doc_id", default=None)
Token.set_extension("qid", default=None)


def load_cdt(custom_split_ids: bool = True):
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

    if not custom_split_ids:  # use the ones specified by the authors
        split_ids = {"train": [], "dev": [], "test": []}

        for split in split_ids:
            split_path = assets_path / "dacoref" / f"CDT_{split}_ids.json"
            with split_path.open(encoding="utf-8") as f:
                split_ids[split] = json.load(f)

    else:  # use the one we created that respects the DDT splits
        with open(assets_path / "CDT_ddt_compatible_splits.json") as f:
            split_ids = json.load(f)

    return sentences, split_ids


def _add_sent_id(docs, split, dataset):
    path = assets_path / dataset / f"{split}.conllu"
    with path.open(encoding="utf-8") as f:
        text = f.read()
    sentences = parse(text)
    for sent, doc in zip(sentences, docs):
        assert doc.text.strip() == sent.metadata["text"].strip()
        doc._.sent_id = sent.metadata["sent_id"]
        doc._.conllu = sent


def load_da_ddt():
    """
    Loads the UD Danish Dependency Treebank
    """
    nlp = spacy.blank("da")
    ddt_path = corpus_path / "da_ddt"

    # With a single file
    ddt = {}
    for split in ["train", "dev", "test"]:
        path = ddt_path / f"{split}.spacy"
        corpus = Corpus(path, shuffle=False)
        examples = list(corpus(nlp))
        docs = [e.reference for e in examples]
        ddt[split] = docs
        _add_sent_id(docs, split, dataset="da_ddt")

    return ddt


def load_dane():
    """
    Loads the UD Danish Dependency Treebank
    """
    nlp = spacy.blank("da")
    dane_path = corpus_path / "dane"

    # With a single file
    dane = {}
    for split in ["train", "dev", "test"]:
        path = dane_path / f"{split}.spacy"
        corpus = Corpus(path, shuffle=False)
        examples = list(corpus(nlp))
        docs = [e.reference for e in examples]
        dane[split] = docs
        _add_sent_id(docs, split, dataset="dane")
    return dane


def add_dane_to_ddt(ddt, dane):
    """
    Add the dane data to the ddt data
    """
    for split in ["train", "dev", "test"]:
        assert len(ddt[split]) == len(dane[split])
        for doc, dane_doc in zip(ddt[split], dane[split]):
            assert doc.text.strip() == dane_doc.text.strip()
            assert doc._.sent_id == dane_doc._.sent_id
            # convert dane ents to ddt ents
            ents = [Span(doc, e.start, e.end, label=e.label_) for e in dane_doc.ents]
            doc.ents = ents
    return ddt


def combine_docs(cdt_sentences, ddt_dane):
    sent_id_to_doc_instance = {}
    for _split, docs in ddt_dane.items():
        for doc in docs:
            assert doc._.sent_id not in sent_id_to_doc_instance
            sent_id_to_doc_instance[doc._.sent_id] = doc

    # combine documents
    doc_to_be_created: Dict[str, List[str]] = {}
    sent_id_to_sent = {}
    for sent in cdt_sentences:
        sent_id = sent.metadata["sent_id"]
        doc_id = sent[0]["doc_id"]
        if doc_id not in doc_to_be_created:
            doc_to_be_created[doc_id] = []
        doc_to_be_created[doc_id].append(sent_id)
        sent_id_to_sent[sent_id] = sent

    # create docs
    domain_mapping = {
        "mz": "magazine",
        "bn": "broadcast",
        "nw": "newswire",
    }
    docs = []
    for doc_id, sent_ids in doc_to_be_created.items():
        _docs = [sent_id_to_doc_instance.pop(sent_id) for sent_id in sent_ids]
        doc = Doc.from_docs(_docs)
        doc._.doc_id = doc_id
        doc._.sent_ids = sent_ids
        doc._.domain = domain_mapping[doc_id.split("/")[0]]
        doc._.conllu = [sent_id_to_sent[sent_id] for sent_id in sent_ids]
        docs.append(doc)

    # add the remaining docs
    for sent_id in list(sent_id_to_doc_instance.keys()):
        doc = sent_id_to_doc_instance.pop(sent_id)
        docs.append(doc)
    return docs


def add_coreference(cdt_sentences, docs):
    doc_id_to_doc_instance = {doc._.doc_id: doc for doc in docs}
    doc_id_to_cdt_sent = defaultdict(list)
    for sent in cdt_sentences:
        doc_id = sent[0]["doc_id"]
        doc_id_to_cdt_sent[doc_id].append(sent)

    for doc_id, sents in doc_id_to_cdt_sent.items():
        clustermap = defaultdict(list)
        doc = doc_id_to_doc_instance[doc_id]
        tokens = [t for sent in sents for t in sent]
        assert len(doc) == len(tokens)
        for token, s_token in zip(tokens, doc):  # type: ignore
            coref_rel = token["coref_rel"]
            if coref_rel == "-":
                continue
            clusters = sorted(coref_rel.split("|"), reverse=True)
            for mention in clusters:
                full_mention = mention.startswith("(") and mention.endswith(")")
                start_mention = mention.startswith("(")
                end_mention = mention.endswith(")")
                if full_mention:
                    cid = mention[1:-1]
                    clustermap[cid].insert(0, (s_token.i, s_token.i + 1))
                elif start_mention:
                    cid = mention[1:]
                    clustermap[cid].append(s_token.i)
                elif end_mention:
                    cid = mention[:-1]
                    start = clustermap[cid].pop()
                    clustermap[cid].insert(0, (start, s_token.i + 1))
        for i, (_key, vals) in enumerate(clustermap.items()):
            spans = [doc[start:end] for start, end in vals]
            skey = f"coref_clusters_{i}"
            doc.spans[skey] = spans

        # parse and get heads
        for i, (_key, val) in enumerate(clustermap.items()):
            heads = [doc[start:end].root.i for start, end in val]
            heads = list(set(heads))
            if len(heads) == 1:
                continue
            spans = [doc[hh : hh + 1] for hh in heads]
            skey = f"coref_head_clusters_{i}"
            doc.spans[skey] = spans
    return docs


def add_qid(docs):
    # add QID to each token
    for doc in docs:
        if doc._.doc_id is None:
            continue
        sents = doc._.conllu
        tokens = [t for sent in sents for t in sent]
        qid_spans = {}  # construct qid spans to check if any of them are not entities
        qid = None
        start = None
        for t, s_t in zip(tokens, doc):
            end_of_span = qid is not None and (t["qid"] == "-" or t["qid"] != qid)
            if end_of_span:
                qid_spans[(start, s_t.i)] = qid
                start = None
                qid = None
            if t["qid"] != "-":
                qid = t["qid"]
                assert qid.startswith("Q")
                s_t._.qid = qid
                if start is None:
                    start = s_t.i
        if start is not None:
            qid_spans[(start, s_t.i)] = qid  # type: ignore

        ents_spans = {(ent.start, ent.end) for ent in doc.ents}
        for qid_span in qid_spans:
            if qid_span[1] - qid_span[0] == 1:
                continue  # ignore single token spans
            if qid_span not in ents_spans:
                print(
                    f"{doc[qid_span[0]:qid_span[1]]} with QID {qid_spans[qid_span]} is not in entities",
                )
            # great no problems here!!

        # map QID to each entity
        new_ents = []
        for ent in doc.ents:
            start, end = ent.start, ent.end
            qids = [t["qid"] for t in tokens[start:end]]
            unique_qid = len(set(qids)) == 1
            if unique_qid:
                qid = qids[0]
                new_ent = Span(doc, start, end, label=ent.label_, kb_id=qid)
            else:
                print(f"Ent {ent} has multiple QIDs: {qids}")
                new_ent = Span(doc, start, end, label=ent.label_)
            new_ents.append(new_ent)
        doc.ents = new_ents
    return docs


cdt_sentences, cdf_split_ids = load_cdt()
doc_id_to_split_mapping = {
    id_: split for split, ids in cdf_split_ids.items() for id_ in ids
}

ddt = load_da_ddt()
dane = load_dane()
ddt_dane = add_dane_to_ddt(ddt, dane)

# add doc_id
sent_id_to_doc_id = {}
for sent in cdt_sentences:
    sent_id_to_doc_id[sent.metadata["sent_id"]] = sent[0]["doc_id"]
for split in ["train", "dev", "test"]:
    for doc in ddt_dane[split]:
        if doc._.sent_id in sent_id_to_doc_id:
            doc._.doc_id = sent_id_to_doc_id[doc._.sent_id]

# check that splits are the same  -- they are not
# for split in ["train", "dev", "test"]:
#     for doc in ddt_dane[split]:
#         assert doc_id_to_split_mapping[doc._.doc_id] == split

# any doc id in cdt that is not in ddt_dane
doc_ids = {doc._.doc_id for split, docs in ddt_dane.items() for doc in docs}
doc_ids_cdt = {sent[0]["doc_id"] for sent in cdt_sentences}
assert len(doc_ids_cdt - doc_ids) == 0

docs = combine_docs(cdt_sentences, ddt_dane)
docs = add_coreference(cdt_sentences, docs)
docs = add_qid(docs)

doc_bin = DocBin(store_user_data=True)
for doc in docs:
    doc_bin.add(doc)

save_path = corpus_path / "cdt_ddt" / "data.spacy"
save_path.parent.mkdir(parents=True, exist_ok=True)
doc_bin.to_disk(save_path)

# do it again for the cdt only but do it in splits:
for split in ["train", "dev", "test"]:
    doc_bin = DocBin(store_user_data=True)
    for doc in docs:
        if doc._.doc_id is None:
            continue
        _split = doc_id_to_split_mapping[doc._.doc_id]
        if _split != split:
            continue
        doc_bin.add(doc)

    save_path = corpus_path / "cdt" / f"{split}.spacy"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    doc_bin.to_disk(save_path)
