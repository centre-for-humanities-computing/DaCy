import spacy, sys
from spacy.tokens import DocBin


def upscale_dansk_train(path_dansk_train, path_ontonotes):
    nlp = spacy.blank("da")

    db_in_dansk = DocBin().from_disk(path_dansk_train)
    dansk_docs = list(db_in_dansk.get_docs(nlp.vocab))

    db_in_ontonotes = DocBin().from_disk(path_ontonotes)
    ontonotes_docs = list(db_in_ontonotes.get_docs(nlp.vocab))

    dansk_docs *= round(len(ontonotes_docs) / len(dansk_docs))

    db = DocBin()
    for doc in dansk_docs:
        db.add(doc)
    db.to_disk(path_dansk_train)


if __name__ == "__main__":
    path_dansk_train = "data/dansk_train.spacy"
    path_ontonotes = "data/ontonotes.spacy"

    if not bool(sys.argv[1]):
        print("Ontonotes is not included in the pipeline (see config), skipping")

    if bool(sys.argv[1]):
        print(f"Upscaling {path_dansk_train} ...")
        upscale_dansk_train(path_dansk_train, path_ontonotes)
        print(f"{path_dansk_train} has been upscaled successfully")
