"""
This includes a series of SpaCy Augmenters
"""

import spacy
from spacy.training import Example
import random
from spacy.training.augment import create_lower_casing_augmenter


def get_names():
    with open("lookup_tables/female_da.csv", "r") as f:
        names = f.read().split("\n")
    with open("lookup_tables/male_da.csv", "r") as f:
        names += f.read().split("\n")
    return set(filter(lambda x: x, names))


@spacy.registry.augmenters("ent_augmenter.v1")
def create_augmenter(ent_dict: dict, prob: float = 0.2):

    def augment(nlp, example):
        def __ent_replace(tok):
            """
            """
            ent = tok.ent_type_
            print(ent)
            if ent and (ent in ent_dict) and (random.random() < prob):
                return random.sample(ent_dict[ent], 1)[0]
            return tok.text

        example_dict = example.to_dict()
        example_dict["token_annotation"]["ORTH"] = [__ent_replace(tok) for tok
                                                    in example.reference]
        doc = nlp.make_doc() # TODO GENERATE Desired text
        yield example.from_dict(doc, example_dict)

    return augment


def create_augmenter_sponge(randomize: bool = False):
    def augment(nlp, example):
        text = example.text
        if randomize:
            # Randomly uppercase/lowercase characters
            chars = [c.lower() if random.random() < 0.5 else c.upper()
                             for c in text]
        else:
            # Uppercase followed by lowercase
            chars = [c.lower() if i % 2 else c.upper()
                             for i, c in enumerate(text)]
        # Create augmented training example
        example_dict = example.to_dict()
        doc = nlp.make_doc("".join(chars))
        example_dict["token_annotation"]["ORTH"] = [t.text for t in doc]
        # Original example followed by augmented example
        # yield example
        yield example.from_dict(doc, example_dict)

    return augment

dir(example.)
nlp = spacy.load("da_core_news_sm")
doc = nlp("Mit navn er Kenneth og Malte og Jakob og Kenneth.")
ent_dict = {"PER": get_names()}
example = Example(doc, doc)
lc_augmenter = create_augmenter_sponge(randomize=False)
res = next(lc_augmenter(nlp, example))
res.text
res
augment = create_augmenter(ent_dict, prob=1)
res
res = next(augment(nlp, res))
res.text

res

dir(example)
for example.y

Example.from_dict(doc, {})
doc.to_dict()

type(Example)
doc = nlp("He pretty quickly walks away")
example = Example.from_dict(doc, {"heads": [3, 2, 3, 0, 2]})
example.to_dict()


for tok in Example(doc, doc):
    tok.ent_type_
 Example(predicted, reference)
