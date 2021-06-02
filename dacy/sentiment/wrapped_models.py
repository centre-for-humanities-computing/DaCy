"""
this script include functions for reading in the wrapped version of DaNLP's BertTone model

This is not meant as a replace of DaNLP, but simply as a convenient wrapper around preexisting architecture.
"""
import os
from spacy.language import Language

from danlp.download import download_model as danlp_download
from danlp.download import _unzip_process_func
from danlp.download import DEFAULT_CACHE_DIR as DANLP_DIR

from dacy.subclasses import (
    ClassificationTransformer,
    install_classification_extensions,
    add_huggingface_model,
)



def add_danlp_model(
    nlp: Language,
    download_name: str,
    subpath: str,
    doc_extension: str,
    model_name: str,
    category: str,
    labels: list,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:
    """Adds a DaNLP transformer model to the NLP pipeline. The transfomer model should be a model for sequence classification.

    Args:
        nlp (Language): A spacy text-processing pipeline
        download_name (str): the name of the model you wish to download
        subpath (str): the path under which DaNLP stores the model
        doc_extension (str): The extension to the doc which you wish the save the transformer data under. This includes output tensor, wordpieces and more.
        model_name (str): What you want your model to be called in the nlp pipeline
        category (str): The category of the output. This is the label which is used to extract from the model. E.g. "sentiment" would allow you to extract the sentiment from doc._.sentiment
        labels (list): The labels of the model
        verbose (bool): Toggles the verbosity of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    if open_unverified_connection:
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    path_sub = danlp_download(
        download_name, DANLP_DIR, process_func=_unzip_process_func, verbose=verbose
    )
    path_sub = os.path.join(path_sub, subpath)

    config = {
        "doc_extension_attribute": doc_extension,
        "model": {
            "@architectures": "dacy.ClassificationTransformerModel.v1",
            "name": path_sub,
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


def add_berttone_subjectivity(
    nlp: Language,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:  
    """Adds the daNLP bertTone model for detecting whether a statement is subjective to the pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
        verbose (bool, optional):  toggles the verbosity (whether it prints or not) of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_danlp_model(
        nlp,
        download_name="bert.subjective",
        subpath="bert.sub.v0.0.1",
        doc_extension="berttone_subj_trf_data",
        model_name="berttone_subj",
        category="subjectivity",
        labels=["objective", "subjective"],
        verbose=verbose,
        open_unverified_connection=open_unverified_connection,
        force_extension=force_extension,
    )


def add_berttone_polarity(
    nlp: Language,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:
    """Adds the daNLP bertTone model for classification of polarity to the pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
        verbose (bool, optional):  toggles the verbosity (whether it prints or not) of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_danlp_model(
        nlp,
        download_name="bert.polarity",
        subpath="bert.pol.v0.0.1",
        doc_extension="berttone_pol_trf_data",
        model_name="berttone_pol",
        category="polarity",
        labels=["positive", "neutral", "negative"],
        verbose=verbose,
        open_unverified_connection=open_unverified_connection,
        force_extension=force_extension,
    )


def add_bertemotion_laden(
    nlp: Language,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:
    """
    Adds the daNLP bertEmoiton model for classifying whether a text is
    emotionally laden or not

    Args:
        nlp (Language): A spacy text-processing pipeline
        verbose (bool, optional):  toggles the verbosity (whether it prints or not) of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_danlp_model(
        nlp,
        download_name="bert.noemotion",
        subpath="bert.noemotion",
        doc_extension="bertemotion_laden_trf_data",
        model_name="bertemotion_laden",
        category="laden",
        labels=["Emotional", "No emotion"],
        verbose=verbose,
        open_unverified_connection=open_unverified_connection,
        force_extension=force_extension,
    )


def add_bertemotion_emo(
    nlp: Language,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:
    """
    Adds the daNLP bertEmoiton model for emotion classification to the spacy language pipeline

    Args:
        nlp (Language): A spacy text-processing pipeline
        verbose (bool, optional):  toggles the verbosity (whether it prints or not) of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
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
        doc_extension="bertemotion_emo_trf_data",
        model_name="bertemotion_emo",
        category="emotion",
        labels=labels,
        verbose=verbose,
        open_unverified_connection=open_unverified_connection,
        force_extension=force_extension,
    )


def add_senda(nlp: Language, verbose: bool = True, force_extension: bool = False) -> Language:
    """
    Adds the senda tranformer model for classification of polarity to the spacy language pipeline

    Args:
        nlp (Language): A spacy text-processing pipeline
        verbose (bool, optional):  toggles the verbosity (whether it prints or not) of the download. Defaults to True.
        open_unverified_connection (bool, optional): Should you download from an unverified connection. Defaults to False.
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_huggingface_model(
        nlp,
        download_name="pin/senda",
        doc_extension="senda_trf_data",
        model_name="senda",
        category="polarity",
        labels=["negative", "neutral", "positive"],
        verbose=verbose,
        force_extension=force_extension,
    )
