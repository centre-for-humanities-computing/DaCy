def LIX_getter(doc):
    """
    extract LIX score

    Example:
    Doc.set_extension("LIX", getter=LIX_getter)
    
    fetch result:
    doc._.LIX
    """
    O = len(doc)
    P = len(list(doc.sents))
    L = len([t for t in doc if len(t) > 6])

    LIX = O / P + L * 100 / O
    return LIX