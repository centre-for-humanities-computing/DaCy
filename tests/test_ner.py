import spacy


def test_ner():
    nlp = spacy.blank("da")
    # or nlp = dacy.load("da_dacy_small_tft-0.0.0", exclude=["ner"])
    nlp.add_pipe("dacy/ner")

    doc = nlp("Jeg hedder Peter og bor i København")
    assert doc.ents[0].text == "Peter"
    assert doc.ents[0].label_ == "PER"
    assert doc.ents[1].text == "København"
    assert doc.ents[1].label_ == "LOC"
