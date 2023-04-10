"""Functionality for loading and locating DaCy models."""
from pathlib import Path
from typing import Any, Union

import spacy
from spacy.language import Language
from wasabi import msg

from .download import DEFAULT_CACHE_DIR, download_model, models_url


def load(
    model: str,
    force: bool = False,
    **kwargs: Any,
) -> Language:
    """Load a DaCy model as a SpaCy text processing pipeline. If the model is
    not downloaded it will also download the model.

    Args:
        model: the model you wish to load. To see available model see
            dacy.models()
        force: Should the model be redownloaded even if
            already downloaded? Default to False.
        kwargs: additional arguments passed to spacy.load()

    Returns:
        A SpaCy text-preprocessing pipeline

    Example:
        >>> import dacy
        >>> dacy.load("da_dacy_medium_trf-0.1.0")
        >>> # or equivalently
        >>> dacy.load("medium")
    """

    path = download_model(model, force=force)
    return spacy.load(path, **kwargs)


def where_is_my_dacy(verbose: bool = True) -> Union[str, Path]:
    """Returns a path to where DaCy models are located. The default the model
    location can be configured with the environmental variable
    `DACY_CACHE_DIR`.

    Args:
        verbose (bool, optional): Toggles the verbosity of the function. Defaults to
            True.

    Returns:
        str: path to the location of DaCy models

    Example:
        >>> import dacy
        >>> dacy.where_is_my_dacy()
    """
    if verbose is True:
        msg.info(
            "DaCy pipelines above and including version 0.1.0 are installed as a "
            + "python module and are thus located in your python environment under "
            + "the respective names. To get a list of installed models use "
            + "spacy.util.get_installed_models()",
        )
    return DEFAULT_CACHE_DIR


def models() -> list:
    """Returns a list of valid DaCy models.

    Returns:
        list: list of valid DaCy models

    Example:
        >>> import dacy
        >>> dacy.models()
    """
    return list(models_url.keys())
