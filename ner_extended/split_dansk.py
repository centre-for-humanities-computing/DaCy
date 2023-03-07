import spacy
from spacy.tokens import DocBin
import random


def partitioning():
    # Load DANSK
    nlp = spacy.blank("da")
    dansk_docs = list(DocBin().from_disk("assets/dansk.spacy").get_docs(nlp.vocab))

    # Shuffle DANSK
    random.seed(0)
    random.shuffle(dansk_docs)

    # Execute split
    print("Splitting commencing ...")
    ten_percent = len(dansk_docs) // 100 * 10
    partitions = {
        "train": dansk_docs[ten_percent * 2 :],
        "dev": dansk_docs[:ten_percent],
        "test": dansk_docs[ten_percent : ten_percent * 2],
    }

    # Save split files and print tag counts to terminal
    for partition in partitions:
        db = DocBin()
        for doc in partitions[partition]:
            db.add(doc)
        db.to_disk(f"corpus/{partition}.spacy")


if __name__ == "__main__":
    partitioning()
