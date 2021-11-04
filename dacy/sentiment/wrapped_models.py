"""
Functions for reading in the wrapped version of sentiment models inclduing DaNLP's sentiment model and Extra Bladet's senda. 
This is not meant as a replacement of existing frameworks, but simply as a convenient wrapper around preexisting architecture.
"""

from spacy.language import Language

from dacy.subclasses import add_huggingface_model


def add_berttone_subjectivity(
    nlp: Language,
    force_extension: bool = False,
) -> Language:
    """Adds the DaNLP BertTone model for detecting whether a statement is subjective to the pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_huggingface_model(nlp, "DaNLP/da-bert-tone-subjective-objective", 
                                 "berttone_subj_trf_data",
                                 "berttone_subj",
                                  "subjectivity", 
                                  labels=["objective", "subjective"], force_extension=force_extension)


def add_berttone_polarity(
    nlp: Language,
    force_extension: bool = False,
) -> Language:
    """Adds the DaNLP BertTone model for classification of polarity to the pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_huggingface_model(nlp, "DaNLP/da-bert-tone-sentiment-polarity", 
                                 "berttone_pol_trf_data",
                                 "berttone_pol",
                                  "polarity", 
                                  labels=["positive", "neutral", "negative"], force_extension=force_extension)


def add_bertemotion_laden(
    nlp: Language,
    force_extension: bool = False,
) -> Language:
    """
    Adds the DaNLP BertEmotion model for classifying whether a text is
    emotionally laden or not.

    Args:
        nlp (Language): A spacy text-processing pipeline
        force_extension (bool, optional): Set the extension to the doc regardless of whether it already exists. Defaults to False.

    Returns:
        Language: your text processing pipeline with the transformer model included
    """
    return add_huggingface_model(nlp, "DaNLP/da-bert-emotion-binary", 
                                 "bertemotion_laden_trf_data",
                                 "bertemotion_laden",
                                  "laden", 
                                  labels=["Emotional", "No emotion"], force_extension=force_extension)


def add_bertemotion_emo(
    nlp: Language,
    force_extension: bool = False,
) -> Language:
    """
    Adds the DaNLP BertEmotion model for emotion classification to the spacy language pipeline.

    Args:
        nlp (Language): A spacy text-processing pipeline
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
    return add_huggingface_model(nlp, "DaNLP/da-bert-emotion-classification", 
                                 "bertemotion_emo_trf_data",
                                 "bertemotion_emo",
                                  "emotion", 
                                  labels=labels, force_extension=force_extension)


def add_senda(nlp: Language, force_extension: bool = False) -> Language:
    """
    Adds the senda tranformer model for classification of polarity to the spacy language pipeline

    Args:
        nlp (Language): A spacy text-processing pipeline
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
