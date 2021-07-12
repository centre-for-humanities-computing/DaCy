"""
Getters extension for extracting sentiment.
"""

from spacy.tokens import Doc
from .vaderSentiment_da import SentimentIntensityAnalyzer


def da_vader_getter(doc: Doc, lemmatization: bool = True) -> dict:
    """A getter function for extracting polarity using the Danish implementation of Vader

    Args:
        doc (Doc): a SpaCy document
        lemmatization (bool, optional): Should it use lemmatization of the document? Defaults to True.

    Returns:
        dict: a dictionary containing positive (pos), negative (neg), neutral (neu) polarity as well as a compound (compound)
    """

    analyser = SentimentIntensityAnalyzer()
    if lemmatization:
        polarity = analyser.polarity_scores(doc.text, tokenlist=[t.lemma_ for t in doc])
    else:
        polarity = analyser.polarity_scores(doc.text, tokenlist=[t.text for t in doc])
    return polarity
