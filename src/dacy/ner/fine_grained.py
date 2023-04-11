from typing import Callable, Literal

from spacy.lang.da import Danish
from spacy.language import Language
from spacy.tokens import Doc

import dacy


@Danish.factory(
    "dacy/ner-fine-grained",
    default_config={},
)
def create_finegrained_ner_component(
    nlp: Language,
    name: str,
    size: Literal["small", "medium", "large"] = "small",
    transformer_name: str = "ner-transformer",
    version: str = "0.1.0",
) -> Callable[[Doc], Doc]:
    """Create a fine grained NER component using the dacy models.

    Args:
        nlp: The spaCy language pipeline
        name: The name of the component
        size: The size of the model to use. Can be "small", "medium" or "large"
        transformer_name: The name of the transformer component which the NER moel will listen to
        version: The version of the model to use
    """

    nlp_ner = dacy.load(f"da_dacy_{size}_ner_fine_grained-{version}")
    nlp.add_pipe(factory_name="transformer", name=transformer_name, source=nlp_ner)
    name_, component = nlp_ner.components[-1]
    component.tok2vec.layers[0].layers[0].upstream_name = transformer_name  # type: ignore
    component.name = name  # type: ignore
    return component
