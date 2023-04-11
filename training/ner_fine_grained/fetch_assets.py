import spacy
from datasets import load_dataset
from spacy.tokens import Doc, DocBin


def fetch_dansk():
    # Download the datasetdict from the HuggingFace Hub
    try:
        datasets = load_dataset("chcaa/DANSK", cache_dir="assets")
    except FileNotFoundError:
        raise FileNotFoundError(
            "DANSK is not available. Check that HuggingFace is up and running, and that the dataset has been publically released.",
        )

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
