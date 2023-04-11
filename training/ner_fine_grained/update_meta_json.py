import json
import sys

"""This is a utility script for updating the spacy meta.json.

The script may be used by following the packaging call in the project.yml.
Sample call:
python -m spacy project run package --
"""


def main(version, size, meta_json_path, no_partitioning):
    print(f"Updating {meta_json_path} with relevant information from the config ...")
    with open(meta_json_path) as f:
        meta = json.load(f)

    mdl_used = {
        "small": {
            "name": "jonfd/electra-small-nordic",
            "author": "Jón Daðason",
            "url": "https://huggingface.co/jonfd/electra-small-nordic",
            "license": "CC BY 4.0",
        },
        "medium": {
            "name": "vesteinn/DanskBERT",
            "author": "Vésteinn Snæbjarnarson",
            "url": "https://huggingface.co/vesteinn/DanskBERT",
            "license": "agpl-3.0",
        },
        "large": {
            "name": "chcaa/dfm-encoder-large-v1",
            "author": "CHCAA",
            "url": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
            "license": "CC BY 4.0",
        },
        "test": {"test": "test"},
    }
    model = mdl_used[size]

    meta["datasets"] = "chcaa/DANSK"
    meta["version"] = version
    if no_partitioning:
        meta["name"] += "_no_partitioning"
    meta["email"] = "Kenneth.enevoldsen@cas.au.dk"
    meta["author"] = "Centre for Humanities Computing Aarhus"
    meta["url"] = "https://chcaa.io/#/"
    meta["license"] = "apache-2.0"
    meta["sources"] = [
        {
            "name": "DANSK - Danish Annotations for NLP Specific TasKs",
            "url": "https://huggingface.co/datasets/chcaa/DANSK",
            "license": "Creative Commons Attribution Share Alike 4.0 International",
            "author": "chcaa",
        },
        model,
    ]
    # fmt: off
    meta[
        "description"
    ] = f"""
<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://centre-for-humanities-computing.github.io/DaCy/_static/icon.png" width="175" height="175" align="right" /></a>

# DaCy_{size}_ner_fine_grained

DaCy is a Danish language processing framework with state-of-the-art pipelines as well as functionality for analyzing Danish pipelines.
At the time of publishing this model, also included in DaCy encorporates the only models for fine-grained NER using DANSK dataset - a dataset containing 18 annotation types in the same format as Ontonotes.
Moreover, DaCy's largest pipeline has achieved State-of-the-Art performance on Named entity recognition, part-of-speech tagging and dependency parsing for Danish on the DaNE dataset.
Check out the [DaCy repository](https://github.com/centre-for-humanities-computing/DaCy) for material on how to use DaCy and reproduce the results.
DaCy also contains guides on usage of the package as well as behavioural test for biases and robustness of Danish NLP pipelines.

For information about the use of this model as well as guides to its use, please refer to [DaCys documentation](https://centre-for-humanities-computing.github.io/DaCy/using_dacy.html).
    """
    # fmt: on
    with open(f"template_meta_{size}.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    version = str(sys.argv[1])
    size = str(sys.argv[2])
    meta_json_path = str(sys.argv[3])
    no_partitioning = bool(sys.argv[4])
    main(version, size, meta_json_path, no_partitioning)
