import collections


def softmax(x):
    import numpy as np
    return np.exp(x) / sum(np.exp(x))


def flatten_dict(d: dict, sep: str = "_") -> dict:
    """flattens a nested dictionary

    Args:
        d (dict): The nested dictionary to be flattened. Keys are assumed to be strings.
        sep (str, optional): The seperator between keys. Defaults to '_'.

    Returns:
        dict: the flattened dict.

    Example:
        >>> flatten({'a': 1, 'c': {'a': 2, 'b': {'x': 5, 'y' : 10}}, 'd': [1, 2, 3]})
        {'a': 1, 'c_a': 2, 'c_b_x': 5, 'd': [1, 2, 3], 'c_b_y': 10}
    """

    def __inner_flatten(d, parent_key, sep):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, collections.MutableMapping):
                items.extend(__inner_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return __inner_flatten(d, "", sep)
