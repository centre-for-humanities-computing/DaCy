Using DaCy
==================

To use the model you first have to download either the small, medium or large model. To see a list
of all available models:

.. code-block:: python

   import dacy
   for model in dacy.models():
      print(model)
   # da_dacy_small_tft-0.0.0
   # da_dacy_medium_tft-0.0.0
   # da_dacy_large_tft-0.0.0

.. note::
   The name of the indicated language (:code:`da`), framework (:code:`dacy`), model size (e.g.
   :code:`small`), model type (:code:`tft`),and model version (:code:`0.0.0`)

From here we can now download a model simply using:

.. code-block:: python

   nlp = dacy.load("da_dacy_medium_tft-0.0.0")

Which will download the model to the :code:`.dacy` directory in your home directory.
If the model is already downloaded this will simply load the model. To download
the model to a specific directory:

.. code-block:: python

   # Just download
   dacy.download_model("da_dacy_medium_tft-0.0.0", your_save_path)
   # Download and load
   nlp = dacy.load_model("da_dacy_medium_tft-0.0.0", your_save_path)

Using this we can now apply DaCy to text using:

.. code-block:: python

   doc = nlp("DaCy er en hurtig og effektiv pipeline til dansk sprogprocessering bygget i SpaCy.")

.. seealso::

   DaCy is build using SpaCy, thus you will be able to find a lot of the required documentation for
   using the pipeline in their very well written documentation on their `website <https://spacy.io>`__

Tagging named entities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A named entity is a “real-world object” that’s assigned a name – for example, a person, a country, a product or a book title. 
DaCy can recognize organizations, persons, and location. As seen below is also includes a mixed category.

.. code-block:: python

   for entity in doc.ents:
      print(entity, ":", entity.label_)
   # DaCy : ORG
   # dansk : MISC
   # SpaCy : ORG

We can also plot these using:

.. code-block:: python

   from spacy import displacy
   displacy.render(doc, style="ent")


.. seealso::

   For more on named entity recognition see SpaCy's `documentation <https://spacy.io/usage/linguistic-features#named-entities>`__.


.. image:: ../img/ner.png
  :width: 800
  :alt: Named entity recognition using DaCy

Tagging parts-of-speech
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   print("Token POS-tag")
   for token in doc:
      print(f"{token}: {token.pos_}")
   # Token POS-tag
   # DaCy:              PROPN
   # er:                AUX
   # en:                DET
   # hurtig:            ADJ
   # og:                CCONJ
   # effektiv:          ADJ
   # pipeline:          NOUN
   # til:               ADP
   # dansk:             ADJ
   # sprogprocessering: NOUN
   # bygget:            VERB
   # i:                 ADP
   # SpaCy:             PROPN
   # .:                 PUNCT

.. seealso::

   For more on Part-of-speech tagging see SpaCy's `documentation <https://spacy.io/usage/linguistic-features#pos-tagging>`__.


Dependency parsing
^^^^^^^^^^^^^^^^^^^^^^
DaCy features a fast and accurate syntactic dependency parser. In DaCy this dependency parsing is also
used for sentence segmentation and detecting noun chunks.

You can see the dependency of DaCy using:

.. code-block:: python

   doc = nlp("DaCy er en effektiv pipeline til dansk fritekst.")
   
   from spacy import displacy
   displacy.render(doc)


.. image:: ../img/dep_parse.png
  :width: 800
  :alt: Dependency parsing using DaCy


.. seealso::

   For more on dependency parsing with DaCy, especially on how to navigate the tree, see SpaCy's `documentation <https://spacy.io/usage/linguistic-features#dependency-parse>`__.



More guides and tutorials
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. |colab_sent| image:: https://colab.research.google.com/assets/colab-badge.svg
   :width: 100pt
   :target: https://colab.research.google.com/github/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-sentiment.ipynb


.. |colab_clf| image:: https://colab.research.google.com/assets/colab-badge.svg
   :width: 100pt
   :target: https://colab.research.google.com/github/KennethEnevoldsen/DaCy/blob/main/tutorials/dacy-wrapping-a-classification-transformer.ipynb

DaCy also include a couple of additional tutorials which are available as a notebook on Google's Colab.

+--------------+-----------------------------------------------------------------+
| Google Colab | Content                                                         |
+==============+=================================================================+
| |colab_sent| |  A simple introduction to the new sentiment features in DaCy.   | 
+--------------+-----------------------------------------------------------------+
| |colab_clf|  | A guide on how to wrap an already fine-tuned transformer and    |
|              | add it to your SpaCy pipeline using DaCy helper functions.      |
+--------------+-----------------------------------------------------------------+

