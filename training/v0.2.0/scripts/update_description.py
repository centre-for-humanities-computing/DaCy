from pathlib import Path
import typer
import json


EMAIL =  "Kenneth.enevoldsen@cas.au.dk"
AUTHOR = "Kenneth Enevoldsen"
URL = "https://chcaa.io/#/"
LICENSE = "Apache-2.0 License"
WANB = "https://wandb.ai/kenevoldsen/dacy-v0.2.0"

DESCRIPTION = """
<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://centre-for-humanities-computing.github.io/DaCy/_static/icon.png" width="175" height="175" align="right" /></a>

# DaCy {size}

DaCy is a Danish language processing framework with state-of-the-art pipelines as well as functionality for analysing Danish pipelines.
DaCy's largest pipeline has achieved State-of-the-Art performance on parts-of-speech tagging and dependency 
parsing for Danish on the DaNE dataset. To read more check out the [DaCy repository](https://github.com/centre-for-humanities-computing/DaCy) for material on how to use DaCy and reproduce the results. 
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
        "license": "agpl-3.0",
    },
    "large": {
        "name": "chcaa/dfm-encoder-large-v1",
        "author": "The Danish Foundation Models team",
        "url": "https://huggingface.co/chcaa/dfm-encoder-large-v1",
        "license": "CC BY 4.0",
    },
}

DATASETS =  [
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
        }
        ]



def main(meta_json: str, out_json: str, size: str, overwrite: bool=False, decimals=3):
    """
    This is a utility script for updating the spacy meta.json.

    Args:
        meta_json: Path to the meta.json file.
        size: The size of the model.
        decimals: The number of decimals to round the score to.
    """
    out_path = Path(out_json)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to overwrite.")

    with open(meta_json) as f:
        meta = json.load(f)

    model = MODELS[size]

    # fmt: off
    meta["email"] = EMAIL
    meta["author"] = AUTHOR
    meta["url"] = URL
    meta["license"] = LICENSE
    meta["sources"] = DATASETS + [model]
    meta["description"] = DESCRIPTION.format(size=size)
    meta["notes"] += f"\n\n### Training\nThis model was trained using [spaCy](https://spacy.io) and logged to [Weights & Biases]({WANB}). You can find all the training logs [here]({WANB})."
    # fmt: on

    with open(out_path, "w") as f:
        json.dump(meta, f)

if __name__ == "__main__":
    typer(main)
