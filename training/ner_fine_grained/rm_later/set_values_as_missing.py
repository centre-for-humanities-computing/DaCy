import spacy, sys
from spacy.tokens import DocBin, Doc

# Below is testing to make it work with settings head to None
# https://github.com/explosion/spaCy/discussions/12307
nlp = spacy.load("da_core_news_lg")
db_in_dansk = DocBin().from_disk("assets/dansk.spacy")
docs = list(db_in_dansk.get_docs(nlp.vocab))


NER_doc = docs[3]
ents = NER_doc.ents

doc_text = NER_doc.text

REST_doc = nlp(doc_text)

REST_doc.set_ents([], default="missing")

doc_ents_iobs = [t.ent_iob_ for t in NER_doc]

spaces = [t.whitespace_ for t in REST_doc]
words = [t.text for t in REST_doc]
tags = [t.tag_ for t in REST_doc]
deps = [t.dep_ for t in REST_doc]
heads = [t.head for t in REST_doc]
lemmas = [t.lemma_ for t in REST_doc]

heads

REST_doc = Doc(
    vocab=nlp.vocab,
    words=words,
    spaces=spaces,
    tags=tags,
    deps=deps,
    # heads=heads,
    lemmas=lemmas,
    # ents=doc_ents_iobs,
)


for i, t in enumerate(REST_doc):
    print(t.head)


def set_values_as_missing(datasets_to_include, nlp_vocab):
    print(f"Datasets to be included in the training: {datasets_to_include}")

    # DANSK
    if "dansk" in datasets_to_include:
        print("Setting lemmas, deps, heads and tags as missing for DANSK")
        for partition in ["", "_train", "_dev", "_test"]:
            # Load in dansk partitions
            db_in_dansk = DocBin().from_disk(f"assets/dansk{partition}.spacy")
            docs = list(db_in_dansk.get_docs(nlp_vocab))

            # Set all values for lemmas, parser, tagger as missing
            # Implement code here from https://github.com/explosion/spaCy/discussions/12307
            # Delete this when not needed any longer: Token.lemma, .tag, .dep are already set to missing. .head IS NOT
            # ...
            # ...
            # ...
            new_docs = []
            for doc in docs:
                spaces = [t.whitespace_ for t in doc]
                words = [t.text for t in doc]
                ents = doc.ents
                new_doc = Doc(
                    vocab=nlp_vocab,
                    words=words,
                    spaces=spaces,
                    lemmas=None,
                    deps=None,
                    heads=None,
                    tags=None,
                )
                new_doc.ents = ents
                new_docs.append(new_doc)

            # Save as .spacy
            db_out_dansk = DocBin()
            for doc in new_docs:
                db_out_dansk.add(doc)
            db_out_dansk.to_disk(f"corpus/dansk{partition}.spacy")
            print(f"corpus/dansk{partition}.spacy saved successfully")

    # DaNE
    if "dane" in datasets_to_include:
        print("Setting ents as missing for DaNE")
        for partition in ["_train", "_dev", "_test"]:
            # Load in the dane partitions
            db_in_dane = DocBin().from_disk(f"assets/dane{partition}.spacy")
            docs = list(db_in_dane.get_docs(nlp_vocab))

            # Set all values ents values as missing
            for doc in docs:
                doc.set_ents([], default="missing")

            # Save as .spacy
            db_out_dane = DocBin()
            for doc in docs:
                db_out_dane.add(doc)
            db_out_dane.to_disk(f"corpus/dane{partition}.spacy")
            print(f"corpus/dane{partition}.spacy saved successfully")

    # Ontonotes
    if "ontonotes" in datasets_to_include:
        print("Setting lemmas, deps, heads and tags as missing for Ontonotes")
        db_in_ontonotes = DocBin().from_disk("assets/ontonotes.spacy")
        docs = list(db_in_ontonotes.get_docs(nlp_vocab))
        new_docs = []
        for doc in docs:
            spaces = [t.whitespace_ for t in doc]
            words = [t.text for t in doc]
            ents = doc.ents
            new_doc = Doc(
                vocab=nlp_vocab,
                words=words,
                spaces=spaces,
                lemmas=None,
                deps=None,
                heads=None,
                tags=None,
            )
            new_doc.ents = ents
            new_docs.append(new_doc)
        db_out_ontonotes = DocBin()
        for doc in new_docs:
            db_out_ontonotes.add(doc)
        db_out_ontonotes.to_disk("corpus/ontonotes.spacy")
        print("corpus/ontonotes.spacy saved successfully")


if __name__ == "__main__":
    datasets_to_include = str(sys.argv[1]).split("_")

    nlp_vocab = spacy.blank("da").vocab

    set_values_as_missing(datasets_to_include, nlp_vocab)
