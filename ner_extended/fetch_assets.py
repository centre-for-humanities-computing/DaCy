import spacy
from spacy.tokens import DocBin, Doc
from datasets import load_dataset


def fetch_dansk():
    # DANSK has yet to publically released. Write the authors for a request of early read access.

    # Download the datasetdict from the HuggingFace Hub
    try:
        datasets = load_dataset("chcaa/DANSK")
    except FileNotFoundError:
        print(
            "\nERROR: DANSK has yet to publically released. Write the authors for a request of early read access. Shutting down.\n"
        )
        raise

    nlp = spacy.blank("da")
    partitions = ["train", "dev", "test"]
    for p in partitions:
        db = DocBin()
        for doc in [
            Doc(nlp.vocab).from_json(dataset_row) for dataset_row in datasets[f"{p}"]
        ]:
            db.add(doc)
        db.to_disk(f"assets/{p}.spacy")
        db.to_disk(f"corpus/{p}.spacy")


if __name__ == "__main__":
    fetch_dansk()
