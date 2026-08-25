# Installation

You can install `dacy` via pip from [PyPI]:

```bash
pip install dacy
```

## Advanced installation

The `0.2.0` DaCy models include a coreference resolution component that depends on `spacy-experimental`. 
This package requires Python <3.12 and requires as so it is an optional extra which can be installed as follows:

```bash
pip install "dacy[coref]"
```


[pip]: https://pip.pypa.io/en/stable/
[PyPI]: https://pypi.org/project/dacy/
[GitHub]: https://github.com/centre-for-humanities-computing/dacy