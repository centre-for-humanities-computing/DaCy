import spacy
from spacy.tokens import DocBin
import random


def tag_counts(docs):
    # Define list of entity labels included in the docs
    unique_ent_labels = []
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ not in unique_ent_labels:
                unique_ent_labels.append(ent.label_)
    list(set(unique_ent_labels))
    # Define a dictionary with a count of entities in the list of docs
    count_of_ents = {}
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ in count_of_ents:
                count_of_ents[f"{ent.label_}"] += 1

            else:
                count_of_ents[f"{ent.label_}"] = 1
    return count_of_ents


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
        print(
            f"corpus/{partition}.spacy saved successfully.\nThis new serialized DocBin contains the following number of entity tags: {tag_counts(partitions[partition])}",
        )


if __name__ == "__main__":
    partitioning()
