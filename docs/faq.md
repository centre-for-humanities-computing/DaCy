
# FAQ


## How do I test the code?

This package comes with a test suite implemented using [pytest].
In order to run the tests, you have to clone the repository and install the package.
This will also install the required tests dependencies
and test utilities defined in the extras_require section of the :code:`pyproject.toml`.

```bash
# clone the repository
git clone https://github.com/centre-for-humanities-computing/dacy

# install package and test dependencies
pip install -e ".[tests]"

# run all tests
python -m pytest
```

which will run all the test in the `tests` folder.

Specific tests can be run using:

```bash
python -m pytest tests/desired_test.py
```

If you want to check code coverage you can run the following:

```bash
python -m pytest --cov=src
```

## How is the documentation generated?

This package use [sphinx] to generate documentation. It uses the [Furo] theme with
custom styling.

To make the documentation you can run:


```bash
# install sphinx, themes and extensions
pip install -e ".[docs]"

# generate html from documentations
sphinx-build -b html docs docs/_build/html
```

## How do I cite this work?
If you use this library in your research, it would be much appreciated it if you would cite:

```
@inproceedings{f975f4ce65944e3ea958578003cee622,
    title = {{{DaCy}}: {{A}} Unified Framework for Danish {{NLP}}},
    booktitle = {Ceur Workshop Proceedings},
    author = {Enevoldsen, Kenneth and Hansen, Lasse and Nielbo, Kristoffer L.},
    date = {2021},
    series = {{{CEUR Workshop Proceedings}}},
    volume = {2989},
    pages = {206--216},
    publisher = {{ceur workshop proceedings}},
    issn = {1613-0073},
    keywords = {Danish NLP,Data Augmentation,Low-resource NLP,Natural Language Processing},
}
```

Or if you prefer APA:

```
Enevoldsen, K., Hansen, L., & Nielbo, K. L. (2021). DaCy: A unified framework for danish NLP. Ceur Workshop Proceedings, 2989, 206-216.
```

The papers is available publicly [here](http://ceur-ws.org/Vol-2989/short_paper24.pdf) and as a preprint [here](https://arxiv.org/abs/2107.05295).

[sphinx]: https://www.sphinx-doc.org/en/master/index.html
[Furo]: https://github.com/pradyunsg/furo
