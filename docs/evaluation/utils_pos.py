def pos_accuracy(examples: list[Example]) -> dict[str, Any]:
    return Scorer().score_token_attr(examples, attr="pos") 