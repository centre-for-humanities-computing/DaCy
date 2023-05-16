import json
from pathlib import Path
from typing import Optional

import typer

EMAIL = "Kenneth.enevoldsen@cas.au.dk"
AUTHOR = "Kenneth Enevoldsen"
URL = "https://chcaa.io/#/"
LICENSE = "Apache-2.0"
WANB = "https://wandb.ai/kenevoldsen/dacy-v0.2.0"

DESCRIPTION = """
<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://centre-for-humanities-computing.github.io/DaCy/_static/icon.png" width="175" height="175" align="right" /></a>

# DaCy {size}

DaCy is a Danish language processing framework with state-of-the-art pipelines as well as functionality for analysing Danish pipelines.
DaCy's largest pipeline has achieved State-of-the-Art performance on parts-of-speech tagging and dependency
parsing for Danish on the Danish Dependency treebank as well as competitive performance on named entity recognition, named entity disambiguation and coreference resolution.
To read more check out the [DaCy repository](https://github.com/centre-for-humanities-computing/DaCy) for material on how to use DaCy and reproduce the results.
DaCy also contains guides on usage of the package as well as behavioural test for biases and robustness of Danish NLP pipelines.
"""

MODELS = {
    "small": {
        "name": "jonfd/electra-small-nordic",
        "author": "Jón Friðrik Daðason",
        "url": "https://huggingface.co/jonfd/electra-small-nordic",
        "license": "CC BY 4.0",
    },
    "medium": {
        "name": "vesteinn/DanskBERT",
        "author": "Vésteinn Snæbjarnarson",
        "url": "https://huggingface.co/vesteinn/DanskBERT",
        "license": "MIT",
    },
    "large": {
        "name": "chcaa/dfm-encoder-large-v1",
        "author": "The Danish Foundation Models team",
        "url": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
        "license": "CC BY 4.0",
    },
}

DATASETS = [
    {
        "name": "UD Danish DDT v2.11",
        "url": "https://github.com/UniversalDependencies/UD_Danish-DDT",
        "license": "CC BY-SA 4.0",
        "author": "Johannsen, Anders; Mart\u00ednez Alonso, H\u00e9ctor; Plank, Barbara",
    },
    {
        "name": "DaNE",
        "url": "https://huggingface.co/datasets/dane",
        "license": "CC BY-SA 4.0",
        "author": "Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard, Anders S\u00f8gaard",
    },
    {
        "name": "DaCoref",
        "url": "https://huggingface.co/datasets/alexandrainst/dacoref",
        "license": "CC BY-SA 4.0",
        "author": "Buch-Kromann, Matthias",
    },
    {
        "name": "DaNED",
        "url": "https://danlp-alexandra.readthedocs.io/en/stable/docs/datasets.html#daned",
        "license": "CC BY-SA 4.0",
        "author": "Barrett, M. J., Lam, H., Wu, M., Lacroix, O., Plank, B., & Søgaard, A.",
    },
]


def main(
    meta_json: str,
    out_json: str,
    size: str,
    metrics_json: Optional[str] = None,
    overwrite: bool = False,
):
    """
    This is a utility script for updating the spacy meta.json.

    Args:
        meta_json: Path to the meta.json file.
        out_json: Path to the output meta.json file.
        size: The size of the model.
        metrics_json: Path to the metrics.json file.
        overwrite: Whether to overwrite the output file if it already exists.
    """
    out_path = Path(out_json)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Use --overwrite to overwrite.",
        )

    with open(meta_json) as f:
        meta = json.load(f)

    model = MODELS[size]

    if metrics_json is not None:
        with open(metrics_json) as f:
            metrics = json.load(f)
        meta["performance"] = metrics

    # fmt: off
    meta["email"] = EMAIL
    meta["author"] = AUTHOR
    meta["url"] = URL
    meta["license"] = LICENSE
    meta["sources"] = [*DATASETS, model]
    meta["description"] = DESCRIPTION.format(size=size)
    meta["notes"] = f"\n\n### Training\nThis model was trained using [spaCy](https://spacy.io) and logged to [Weights & Biases]({WANB}). You can find all the training logs [here]({WANB})."
    # fmt: on

    with open(out_path, "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    typer.run(main)
