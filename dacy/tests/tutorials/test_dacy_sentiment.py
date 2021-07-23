
def test_tutorial():
    from dacy.sentiment import add_senda
    import spacy

    # an empty pipeline - replace it with your pipeline of choice
    nlp = spacy.blank("da")

    nlp = add_senda(nlp, force_extension=True)

    texts = ["Sikke en dejlig dag det er i dag", "Sikke noget forfærdeligt møgvejr det er i dag", "FC København og Brøndby IF i duel om mesterskabet"]

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.polarity)
        print(doc._.polarity_prop)

    from dacy.sentiment import add_berttone_subjectivity

    nlp = spacy.blank("da")
    nlp = add_berttone_subjectivity(nlp, force_extension=True)

    texts = ["Analysen viser, at økonomien bliver forfærdelig dårlig", 
            "Jeg tror alligevel, det bliver godt"]

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.subjectivity)
        print(doc._.subjectivity_prop)

    from dacy.sentiment import add_berttone_polarity
    nlp = add_berttone_polarity(nlp, force_extension=True) # force_extension let us overwrite the polarity from using senda

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.polarity)
        print(doc._.polarity_prop)

    from dacy.sentiment import add_bertemotion_emo, add_bertemotion_laden
    nlp = add_bertemotion_laden(nlp, force_extension=True)  # whether a text is emotionally laden
    nlp = add_bertemotion_emo(nlp, force_extension=True)    # what emotion is portrayed

    texts = ['bilen er flot', 
            'jeg ejer en rød bil og det er en god bil', 
            'jeg ejer en rød bil men den er gået i stykker', 
            "Ifølge TV udsendelsen så bliver vejret skidt imorgen",  
            "Fuck jeg hader bare Hitler. Han er bare så FUCKING træls!",
            "Har i set at Tesla har landet en raket på månen? Det er vildt!!",
            "Nu må vi altså få ændret noget",
            "En sten kan ikke flyve. Morlille kan ikke flyve. Ergo er morlille en sten!"]

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.laden)
        print("\t", doc._.emotion)

    from spacy.tokens import Doc
    from dacy.sentiment import da_vader_getter

    Doc.set_extension("vader_da", getter=da_vader_getter, force=True)

    # download model
    from spacy.cli import download
    
    download("da_core_news_sm")

    nlp = spacy.load("da_core_news_sm")
    texts = ['Jeg er så glad', 'jeg ejer en rød bil og det er en god bil', 'jeg ejer en rød bil men den er gået i stykker']

    docs = nlp.pipe(texts)

    for doc in docs:
        print(doc._.vader_da)