"""
Functionality for loading and locating DaCy models.
"""
import os
from typing import Optional

from spacy.language import Language

from .download import download_model, DEFAULT_CACHE_DIR, models_url


def load(model: str, path: Optional[str] = None) -> Language:
    """
    Load a DaCy model as a SpaCy text processing pipeline. If the model is not downloaded it will also download the model.

    Args:
        model (str): the model you wish to load. To see available model see dacy.models()
        path (str, optional): The path to the downloaded model. Defaults to None which corresponds to the path optained using dacy.where_is_my_dacy().

    Returns:
        Language: a SpaCy text-preprocessing pipeline

    Example:
        >>> import dacy
        >>> dacy.load("da_dacy_medium_tft-0.0.0")
        >>> # or equivalently
        >>> dacy.load("medium")
    """
    import spacy

    if path is None:
        path = DEFAULT_CACHE_DIR

    if model.lower() in {"small", "medium", "large"}:
        model = f"da_dacy_{model}_tft-0.0.0"
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
