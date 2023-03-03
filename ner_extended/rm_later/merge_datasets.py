import spacy, sys
from spacy.tokens import DocBin


def merge_datasets(no_dev_test, nlp_vocab):

    datasets_docs = {"train": [], "dev": [], "test": []}
    partitions = ["train", "dev", "test"]

    if not no_dev_test:
        print(
            "no-dev-test set to 0: Merging DANSK and DaNE in train.spacy, dev.spacy, test.spacy"
        )
        for p in partitions:
            dansk_docs = list(
                DocBin().from_disk(f"corpus/dansk_{p}.spacy").get_docs(nlp_vocab)
            )
            datasets_docs[f"{p}"].extend(dansk_docs)

            dane_docs = list(
                DocBin().from_disk(f"corpus/dane_{p}.spacy").get_docs(nlp_vocab)
            )
            datasets_docs[f"{p}"].extend(dane_docs)

            db = DocBin()
            for doc in datasets_docs[f"{p}"]:
                db.add(doc)
            db.to_disk(f"corpus/{p}.spacy")
            print(f"corpus/{p}.spacy saved successfully")

    else:
        print(
            "no-dev-test set to 1: Merging DANSK and DaNE in train.spacy - excluding dev and test sets"
        )
        for p in partitions:
            dansk_docs = list(
                DocBin().from_disk(f"corpus/dansk_{p}.spacy").get_docs(nlp_vocab)
            )
            datasets_docs["train"].extend(dansk_docs)

            dane_docs = list(
                DocBin().from_disk(f"corpus/dane_{p}.spacy").get_docs(nlp_vocab)
            )
            datasets_docs["train"].extend(dane_docs)

            db = DocBin()
            for doc in datasets_docs[f"{p}"]:
                db.add(doc)
            db.to_disk(f"corpus/{p}.spacy")
            print(f"corpus/{p}.spacy saved successfully")


if __name__ == "__main__":
    no_dev_test = bool(int(sys.argv[1]))
    nlp_vocab = spacy.blank("da").vocab

    merge_datasets(no_dev_test, nlp_vocab)
