"""
"""

import asyncio
from pathlib import Path

# for running it in jupyter
import nest_asyncio
import spacy
from spacy.tokens import DocBin
from wikidata.client import Client

nest_asyncio.apply()


def get_ents():
    nlp = spacy.blank("da")

    path_to_cdt = Path(
        "/Users/au561649/Github/DaCy/training/main/corpus/cdt_ddt/data.spacy"
    )

    doc_bin = DocBin(store_user_data=True).from_disk(path_to_cdt)
    docs = list(doc_bin.get_docs(nlp.vocab))

    ents = [e for doc in docs for e in doc.ents]
    return docs, ents


async def _check_if_qid_in_wikidata(qid):
    client = Client()
    try:
        # item = client.get(qid, load=True)  # type: ignore
        item = await asyncio.to_thread(client.get, qid, load=True)
    except Exception:
        print(f"{id}: Could not load {qid}")
        return False
    return True


async def check_if_qids_in_wikidata(qids):
    tasks = [asyncio.create_task(_check_if_qid_in_wikidata(qid)) for qid in qids]
    results = await asyncio.gather(*tasks)
    return results


async def _is_given_name(qid):
    name_qids = [
        "Q202444",  # given name
        "Q101352",  # family name
        "Q11879590",  # male given name
        "Q12308941",  # female given name
        "Q3409032",  # unisex given name
        "Q49614",  # nickname
    ]
    if qid in name_qids:
        return qid
    client = Client()
    try:
        item = await asyncio.to_thread(client.get, qid, load=True)
    except Exception:
        print(f"Could not load {qid}")
        return False
    for claim in item.attributes["claims"]["P31"]:  # P31 == instance of # type: ignore
        _qid = claim["mainsnak"]["datavalue"]["value"]["id"]
        if _qid in name_qids:
            return _qid


async def is_given_name(qids):
    tasks = [asyncio.create_task(_is_given_name(qid)) for i, qid in enumerate(qids)]
    results = await asyncio.gather(*tasks)
    return results


def main():
    docs, ents = get_ents()
    qids = [ent.kb_id_ for ent in ents if ent.kb_id_ != "-"]
    qids = list(set(qids))

    results = asyncio.run(check_if_qids_in_wikidata(qids))

    qids_not_in_wikidata = [qids[i] for i, result in enumerate(results) if not result]
    print(f"Number of qids not in wikidata: {len(qids_not_in_wikidata)}")  # 0
    qids_in_wikidata = [qids[i] for i, result in enumerate(results) if result]
    print(f"Number of qids in wikidata: {len(qids_in_wikidata)}")  # 1199

    person = [e for e in ents if e.label_ in ["PER", "PERSON"]]
    person_w_qid = [e for e in person if e.kb_id_ != "" and e.kb_id_ != "-"]

    qids = [ent.kb_id_ for ent in person_w_qid]
    qids = list(set(qids))

    results = asyncio.run(is_given_name(qids))
    qid_to_given_name = {qids[i]: result for i, result in enumerate(results) if result}


if __name__ == "__main__":
    main()
