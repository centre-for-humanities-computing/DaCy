"""
Functions for reading in the wrapped version of sentiment models inclduing DaNLP's sentiment model and Extra Bladet's senda. 
This is not meant as a replacement of existing frameworks, but simply as a convenient wrapper around preexisting architecture.
"""

from spacy.language import Language

from dacy.subclasses import (
    add_danlp_model,
    add_huggingface_model,
)


def add_berttone_subjectivity(
    nlp: Language,
    verbose: bool = True,
    open_unverified_connection: bool = False,
    force_extension: bool = False,
) -> Language:
    """Adds the DaNLP BertTone model for detecting whether a statement is subjective to the pipeline.

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
    """Adds the DaNLP BertTone model for classification of polarity to the pipeline.

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
    Adds the DaNLP BertEmotion model for classifying whether a text is
    emotionally laden or not.

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
    Adds the DaNLP BertEmotion model for emotion classification to the spacy language pipeline.

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


def add_senda(nlp: Language, force_extension: bool = False) -> Language:
    """
    Adds the senda tranformer model for classification of polarity to the spacy language pipeline

    Args:
        nlp (Language): A spacy text-processing pipeline
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
        force_extension=force_extension,
    )
