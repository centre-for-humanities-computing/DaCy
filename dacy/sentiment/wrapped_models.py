"""
this script include functions for reading in the wrapped version of DaNLP's BertTone model

This is not meant as a replace of DaNLP, but simply as a convenient wrapper around preexisting architecture.
"""
import os

from danlp.download import download_model as danlp_download
from danlp.download import _unzip_process_func
from danlp.download import DEFAULT_CACHE_DIR as DANLP_DIR

from dacy.subclasses import ClassificationTransformer, install_classification_extensions


def add_danlp_model(
    nlp,
    download_name: str,
    subpath: str,
    doc_extention: str,
    model_name: str,
    category: str,
    labels: list,
    verbose: bool,
):
    """
    adds a the DaNLP bert model to the pipeline
    """
    path_sub = danlp_download(
        download_name, DANLP_DIR, process_func=_unzip_process_func, verbose=verbose
    )
    path_sub = os.path.join(path_sub, subpath)

    config = {
        "doc_extention_attribute": doc_extention,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": path_sub,
            "num_labels": len(labels),
        },
    }

    install_classification_extensions(
        category=category, labels=labels, doc_extention=doc_extention
    )

    transformer = nlp.add_pipe(
        "classification_transformer", name=model_name, config=config
    )
    transformer.model.initialize()
    return nlp


def add_berttone_subjectivity(nlp, verbose: bool = True):
    """
    adds a the DaNLP BertTone for polarity classification to the spacy language pipeline
    """
    return add_danlp_model(
        nlp,
        download_name="bert.subjective",
        subpath="bert.sub.v0.0.1",
        doc_extention="berttone_subj_trf_data",
        model_name="berttone_subj",
        category="subjectivity",
        labels=["objective", "subjective"],
        verbose=verbose,
    )


def add_berttone_polarity(nlp, verbose: bool = True):
    """
    adds a the DaNLP BertTone for polarity classification to the spacy language pipeline
    """
    return add_danlp_model(
        nlp,
        download_name="bert.polarity",
        subpath="bert.pol.v0.0.1",
        doc_extention="berttone_pol_trf_data",
        model_name="berttone_pol",
        category="polarity",
        labels=["positive", "neutral", "negative"],
        verbose=verbose,
    )


def add_bertemotion_laden(nlp, verbose: bool = True):
    """
    adds to the spacy language pipeline a the DaNLP BertEmotion for classifying whether a text is
    emotionally laden or not
    """
    return add_danlp_model(
        nlp,
        download_name="bert.noemotion",
        subpath="bert.noemotion",
        doc_extention="bertemotion_laden_trf_data",
        model_name="bertemotion_laden",
        category="laden",
        labels=["Emotional", "No emotion"],
        verbose=verbose,
    )


def add_bertemotion_emo(nlp, verbose: bool = True):
    """
    adds a the DaNLP BertEmotion for emotion classification to the spacy language pipeline
    """
    labels = [
        "Glæde/Sindsro",
        "Tillid/Accept",
        "Forventning/Interrese",
        "Overasket/Målløs",
        "Vrede/Irritation",
        "Foragt/Modvilje",
        "Sorg/trist",
        "Frygt/Bekymret",
    ]
    return add_danlp_model(
        nlp,
        download_name="bert.emotion",
        subpath="bert.emotion",
        doc_extention="bertemotion_emo_trf_data",
        model_name="bertemotion_emo",
        category="emotion",
        labels=labels,
        verbose=verbose,
    )


def add_huggingface_model(
    nlp,
    download_name: str,
    doc_extention: str,
    model_name: str,
    category: str,
    labels: list,
    verbose: bool = True,
):
    """
    adds a Huggingface sequence classification model to the pipeline

    Example:
    add_huggingface_model(nlp, download_name="pin/senda", doc_extention="senda_trf_data", model_name="senda",
                          category="polarity", labels=["negative", "neutral", "positive"])
    """

    config = {
        "doc_extention_attribute": doc_extention,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": download_name,
            "num_labels": len(labels),
        },
    }

    install_classification_extensions(
        category=category, labels=labels, doc_extention=doc_extention
    )

    transformer = nlp.add_pipe(
        "classification_transformer", name=model_name, config=config
    )
    transformer.model.initialize()
    return nlp

def add_senda(nlp, verbose:bool=True):
    return add_huggingface_model(nlp, download_name="pin/senda", doc_extention="senda_trf_data", model_name="senda",
                          category="polarity", labels=["negative", "neutral", "positve"], verbose=verbose)