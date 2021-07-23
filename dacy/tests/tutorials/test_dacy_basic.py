

import dacy


def test_tutorial():
    # to see available models
    for model in dacy.models():
        print(model)

    # Loading the medium model
    nlp = dacy.load("da_dacy_medium_tft-0.0.0")
    # Models can also be loaded using the "small", "medium" or "large" shorthand 

    doc = nlp("DaCy er en hurtig og effektiv pipeline til dansk sprogprocessering bygget i SpaCy .")

    for entity in doc.ents:
        print(entity, ":", entity.label_)

    from spacy import displacy
    displacy.render(doc, style="ent")

    print("Token POS-tag")
    for token in doc:
        print(f"{token}: \t\t {token.tag_}")

    doc = nlp("DaCy er en effektiv pipeline til dansk fritekst.")
    from spacy import displacy
    displacy.render(doc)