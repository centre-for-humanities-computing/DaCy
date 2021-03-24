from .vaderSentiment_da import SentimentIntensityAnalyzer

def da_vader_getter(doc):
    """
    extract polarity using a Danish implementation of Vader.
    

    Example:
    Doc.set_extension("vader_da", getter=da_vader_getter)
    # fetch result
    doc._.vader_polarity
    """
    analyser = SentimentIntensityAnalyzer()
    polarity = analyser.polarity_scores(doc.text)
    return polarity

