"""
Convenient wrapper functions for wrapping DaNLP and Huggingface models in a SpaCy text processing pipeline.
"""

from .classification_transformer import install_classification_extensions
import os
from spacy.language import Language


def add_huggingface_model(
    nlp: Language,
    download_name: str,
    doc_extension: str,
    model_name: str,
    category: str,
    labels: list,
    force_extension: bool = False,
) -> Language:
    """
    adds a Huggingface sequence classification model to the spacy pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
        download_name (str): the name of the model you wish to download
        doc_extension (str): The extension to the doc which you wish the save the transformer data under.
            This includes output tensor, wordpieces and more.
        model_name (str): What you want your model to be called in the nlp pipeline
        category (str): The category of the output. This is the label which is used to extract from the model.
            E.g. "sentiment" would allow you to extract the sentiment from doc._.sentiment
        labels (list): The labels of the model
        verbose (bool): Toggles the verbosity of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included

    Example:
        >>> add_huggingface_model(nlp, download_name="pin/senda", doc_extension="senda_trf_data", model_name="senda", \
category="polarity", labels=["negative", "neutral", "positive"])
    """

    config = {
        "doc_extension_attribute": doc_extension,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": download_name,
            "num_labels": len(labels),
        },
    }

    install_classification_extensions(
        category=category,
        labels=labels,
        doc_extension=doc_extension,
        force=force_extension,
    )

    transformer = nlp.add_pipe(
        "classification_transformer", name=model_name, config=config
    )
    transformer.model.initialize()
    return nlp
