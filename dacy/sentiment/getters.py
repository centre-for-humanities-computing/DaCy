from .vaderSentiment_da import SentimentIntensityAnalyzer


def da_vader_getter(doc, lemmatization=True):
    """
    extract polarity using a Danish implementation of Vader.

    Example:
    Doc.set_extension("vader_da", getter=da_vader_getter)
    # fetch result
    doc._.vader_polarity
    """
    analyser = SentimentIntensityAnalyzer()
    if lemmatization:
        polarity = analyser.polarity_scores(doc.text, tokenlist=[t.lemma_ for t in doc])
    else:
        polarity = analyser.polarity_scores(doc.text, tokenlist=[t.text for t in doc])
    return polarity
