import spacy
from spacy.tokens import DocBin, Doc

# init nlp
nlp = Doc(vocab=spacy.blank("da").vocab)

# set doc extensions
Doc.set_extension("doc_id", default=None)
Doc.set_extension("sent_id", default=None)

# load data.spacy
cdt_ddt_file = "/work/training_practice/ddt_dane_cdt_project/corpus/cdt_ddt/data.spacy"
db = DocBin().from_disk(cdt_ddt_file)

# get docs from DocBin
docs = list(db.get_docs(nlp.vocab))

# get docs with and without doc_ids
# docs with doc_ids are already used in training data
# docs withoud doc_ids are not
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
cdt_files = [
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt/train.spacy",
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt/dev.spacy",
    "/work/training_practice/ddt_dane_cdt_project/corpus/cdt/test.spacy"
    ]

# combine cdt and newly split remaining data
for split, file in zip(["train", "dev", "test"], cdt_files):
    db = DocBin().from_disk(file)
    docs = list(db.get_docs(nlp.vocab))
    docs = docs + remaining_splits[split]
    combined_db = DocBin(store_user_data=True)
    for doc in docs:
        combined_db.add(doc)
    combined_db.to_disk(f"/work/training_practice/ddt_dane_cdt_project/corpus/cdt_ddt/{split}.spacy")