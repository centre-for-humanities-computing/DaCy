import spacy
from spacy.tokens import DocBin, Doc
from datasets import load_dataset


def fetch_dansk():
    # Download the datasetdict from the HuggingFace Hub
    try:
        datasets = load_dataset("chcaa/DANSK", cache_dir="assets")
    except FileNotFoundError:
        raise FileNotFoundError(
            "DANSK is not available. It might be due to either HuggingFace being down on the dataset not yet being publically released.",
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
