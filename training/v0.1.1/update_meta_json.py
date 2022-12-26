"""This is a utility script for updating the spacy meta.json.

Sample call
python --meta meta.json --augment metrics/dane_augmented_best_dacy_small_trf-0.1.0.json --
"""

import json


def create_description():
    from augment import augmenters

    describtion = """
<details>

<summary> Description of Augmenters </summary>

    """
    describtion
    for aug, nam, k, desc in augmenters:
        describtion += f"\n\n**{nam}:**\n{desc}"
    describtion += "\n </details> \n <br /> \n"
    return describtion


def main(meta_json, meta_augment_json, size, decimals=3):
    with open(meta_json) as f:
        meta = json.load(f)
    with open(meta_augment_json) as f:
        meta_augment = json.load(f)

    meta["email"] = "Kenneth.enevoldsen@cas.au.dk"
    meta["author"] = "Centre for Humanities Computing Aarhus"
    meta["url"] = "https://chcaa.io/#/"
    meta["license"] = "Apache-2.0 License"

    mdl_used = {
        "small": {
            "name": "Maltehb/-l-ctra-danish-electra-small-cased",
            "author": "Malte Højmark-Bertelsen",
            "url": "https://huggingface.co/Maltehb/-l-ctra-danish-electra-small-cased",
            "license": "CC BY 4.0",
        },
        "medium": {
            "name": "Maltehb/danish-bert-botxo",
            "author": "BotXO.ai",
            "url": "https://huggingface.co/Maltehb/danish-bert-botxo",
            "license": "CC BY 4.0",
        },
        "large": {
            "name": "xlm-roberta-large",
            "author": "Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, Veselin Stoyanov",
            "url": "https://huggingface.co/xlm-roberta-large",
            "license": "CC BY 4.0",
        },
    }
    model = mdl_used[size]

    meta["sources"] = [
        {
            "name": "UD Danish DDT v2.5",
            "url": "https://github.com/UniversalDependencies/UD_Danish-DDT",
            "license": "CC BY-SA 4.0",
            "author": "Johannsen, Anders; Mart\u00ednez Alonso, H\u00e9ctor; Plank, Barbara",
        },
        {
            "name": "DaNE",
            "url": "https://github.com/alexandrainst/danlp/blob/master/docs/datasets.md#danish-dependency-treebank-dane",
            "license": "CC BY-SA 4.0",
            "author": "Rasmus Hvingelby, Amalie B. Pauli, Maria Barrett, Christina Rosted, Lasse M. Lidegaard, Anders S\u00f8gaard",
        },
        model,
    ]

    meta["requirements"] = ["spacy-transformers>=1.0.3,<1.1.0"]

    meta[
        "description"
    ] = f"""
<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://centre-for-humanities-computing.github.io/DaCy/_static/icon.png" width="175" height="175" align="right" /></a>

# DaCy {size} transformer

DaCy is a Danish language processing framework with state-of-the-art pipelines as well as functionality for analysing Danish pipelines.
DaCy's largest pipeline has achieved State-of-the-Art performance on Named entity recognition, part-of-speech tagging and dependency 
parsing for Danish on the DaNE dataset. Check out the [DaCy repository](https://github.com/centre-for-humanities-computing/DaCy) for material on how to use DaCy and reproduce the results. 
DaCy also contains guides on usage of the package as well as behavioural test for biases and robustness of Danish NLP pipelines.
    """

    meta[
        "notes"
    ] = """
## Bias and Robustness

Besides the validation done by SpaCy on the DaNE testset, DaCy also provides a series of augmentations to the DaNE test set to see how well the models deal with these types of augmentations.
The can be seen as behavioural probes akinn to the NLP checklist.

### Deterministic Augmentations
Deterministic augmentations are augmentation which always yield the same result.

| Augmentation | Part-of-speech tagging (Accuracy) | Morphological tagging (Accuracy) | Dependency Parsing (UAS) | Dependency Parsing (LAS) | Sentence segmentation (F1) | Lemmatization (Accuracy) | Named entity recognition (F1) |
| --- | --- | --- | --- |  --- | --- | --- |  --- |
"""

    for aug, metrics in meta_augment.items():
        if metrics["k"] == 1:
            pos = f'{round(metrics["mean"]["pos_acc"], decimals)}'
            morph = f'{round(metrics["mean"]["morph_acc"], decimals)}'
            dep_uas = f'{round(metrics["mean"]["dep_uas"], decimals)}'
            dep_las = f'{round(metrics["mean"]["dep_las"], decimals)}'
            sent_f = f'{round(metrics["mean"]["sents_f"], decimals)}'
            lemma = f'{round(metrics["mean"]["lemma_acc"], decimals)}'
            ents_f = f'{round(metrics["mean"]["ents_f"], decimals)}'
            meta[
                "notes"
            ] += f"| {aug} | {pos} | {morph} | {dep_uas} |  {dep_las} | {sent_f} | {lemma} | {ents_f} |\n"

    meta[
        "notes"
    ] += """


### Stochastic Augmentations
Stochastic augmentations are augmentation which are repeated mulitple times to estimate the effect of the augmentation.

| Augmentation | Part-of-speech tagging (Accuracy) | Morphological tagging (Accuracy) | Dependency Parsing (UAS) | Dependency Parsing (LAS) | Sentence segmentation (F1) | Lemmatization (Accuracy) | Named entity recognition (F1) |
| --- | --- | --- | --- |  --- | --- | --- |  --- |
"""

    for aug, metrics in meta_augment.items():
        if metrics["k"] > 1:
            pos = f'{round(metrics["mean"]["pos_acc"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            morph = f'{round(metrics["mean"]["morph_acc"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            dep_uas = f'{round(metrics["mean"]["dep_uas"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            dep_las = f'{round(metrics["mean"]["dep_las"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            sent_f = f'{round(metrics["mean"]["sents_f"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            lemma = f'{round(metrics["mean"]["lemma_acc"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            ents_f = f'{round(metrics["mean"]["ents_f"], decimals)} ({round(metrics["std"]["pos_acc"], decimals)})'
            meta[
                "notes"
            ] += f"| {aug} | {pos} | {morph} | {dep_uas} |  {dep_las} | {sent_f} | {lemma} | {ents_f} |\n"

    meta["notes"] += create_description()

    meta[
        "notes"
    ] += "\n\n### Hardware\nThis was run an trained on a Quadro RTX 8000 GPU."

    with open(f"template_meta_{size}.json", "w") as f:
        json.dump(meta, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta",
        type=str,
        help="the meta file you wish to update",
        required=True,
    )
    parser.add_argument(
        "--augment",
        type=str,
        help="the json file of the augmented resutls",
        required=True,
    )
    parser.add_argument("--size", type=str, help="the model size", required=True)

    args = parser.parse_args()
    main(args.meta, args.augment, args.size)
