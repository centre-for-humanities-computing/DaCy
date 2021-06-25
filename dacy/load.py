"""
This includes a series of functionality for loading and locating DaCy models.
"""
import os
from typing import Optional

import spacy
from spacy.language import Language

from .download import download_model, DEFAULT_CACHE_DIR, models_url

def load(model: str, path: Optional[str] = None) -> Language:
    """
    load a dacy model as a SpaCy text processing pipeline. If the model is not downloaded the it will download the model.

    Args:
        model (str): the model you wish to load. To see available model see dacy.models()
        path (str, optional): The path to the downloaded model. Defaults to None which corresponds to the path optained using dacy.where_is_my_dacy().

    Returns:
        Language: a SpaCy text-preprocessing pipeline

    Example:
        >>> import dacy
        >>> dacy.load("da_dacy_medium_tft-0.0.0")
    """
    if path is None:
        path = DEFAULT_CACHE_DIR

    download_model(model, path)
    path = os.path.join(path, model)
    return spacy.load(path)


def where_is_my_dacy() -> str:
    """Returns a path to where DaCy models are located

    Returns:
        str: path to the location of DaCy models

    Example:
        >>> import dacy
        >>> dacy.where_is_my_dacy()
    """
    return DEFAULT_CACHE_DIR

def models() -> list:
    """
    Returns a list of valid DaCy models

    Returns:
        list: list of valid DaCy models

    Example:
        >>> import dacy
        >>> dacy.models()
    """
    return list(models_url.keys())