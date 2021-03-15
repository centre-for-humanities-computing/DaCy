import os

import spacy

from .download import download_model, DEFAULT_CACHE_DIR


def load(model, path=DEFAULT_CACHE_DIR):
    """
    model (str): use models() to see all available models
    """
    download_model(model, path)
    path = os.path.join(path, model)
    return spacy.load(path)


def load(model, path=DEFAULT_CACHE_DIR):
    """
    model (str): use models() to see all available models
    """
    download_model(model, path)
    path = os.path.join(path, model)
    return spacy.load(path)