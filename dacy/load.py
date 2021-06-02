import os

import spacy
from spacy.language import Language

from .download import download_model, DEFAULT_CACHE_DIR


def load(model: str, path: str = DEFAULT_CACHE_DIR) -> Language:
    """
    load a dacy model as a SpaCy text processing pipeline. If the model is not downloaded the it will download the model.

    Args:
        model (str): the model you wish to load. To see available model see dacy.models()
        path (str, optional): The path to the downloaded model. Defaults to DEFAULT_CACHE_DIR
        which can be optained using dacy.where_is_my_dacy().

    Returns:
        Language: a SpaCy text-preprocessing pipeline
    """
    download_model(model, path)
    path = os.path.join(path, model)
    return spacy.load(path)