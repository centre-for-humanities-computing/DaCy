Frequently Asked Questions
----------------------------


How do I test the code and run the test suite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DaCy comes with an extensive test suite. To run the tests, you should clone the repository, then install DaCy using `Poetry <https://python-poetry.org>`__.
This will also install the required development dependencies and test utilities defined in the pyproject.toml.


.. code-block::
   
   poetry install

   python -m pytest


which will run all the test in the :code:`tests` folder.

Specific tests can be run using:

.. code-block::

   python -m pytest tests/desired_test.py


If you want to check code coverage you can run the following:

.. code-block::

   pip install pytest-cov

   python -m pytest --cov=.


Does DaCy run on X?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DaCy is intended to run on all major OS, this includes Windows (latest version), MacOS (latest) and the latest version of Linux (Ubuntu). 
Similarly it also tested on python 3.7, 3.8, and 3.9.
Please note these are only the systems DaCy is being actively tested on, if you run on a similar system (e.g. an earlier version of Linux) DaCy
will likely run there as well, if not please create an issue.

How is the documentation generated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Augmenty uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme with custom styling.

To make the documentation you can run:

.. code-block::

  # install sphinx, themes and extensions
  pip install sphinx furo sphinx-copybutton sphinxext-opengraph

  # or using poetry 
  # poetry install

  # generate html from documentations

  make -C docs html


How do I cite this work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you use this library in your research, it would be much appreciated it if you would cite:

.. code-block::
   
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

Or if you prefer APA:

.. code-block::
   
    Enevoldsen, K., Hansen, L., & Nielbo, K. L. (2021). DaCy: A unified framework for danish NLP. Ceur Workshop Proceedings, 2989, 206-216.

The papers is available publicly `here <http://ceur-ws.org/Vol-2989/short_paper24.pdf>`__ and as a preprint `here <https://arxiv.org/abs/2107.05295>`__.