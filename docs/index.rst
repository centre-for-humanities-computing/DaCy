DaCy
================================

.. image:: https://img.shields.io/github/stars/KennethEnevoldsen/DaCy.svg?style=social&label=Star&maxAge=2592000
   :target: https://github.com/KennethEnevoldsen/DaCy

DaCy is a Danish text processing pipeline built using SpaCy. At the time of writing it has achieved State-of-the-Art performance on part-of-speech (POS) tagging, 
named-entity recognition (NER) and Dependency parsing for Danish. 

This website contains the documentation for DaCy as well introduction for how to get started using DaCy for your project.

📰 News
---------------------------------

* 04/06/21

  -DaCy now have a stunningly looking documentation site 🌟

* 01/06/21

  - DaCy's tests now cover 99% of its codebase 🎉
  - DaCy's test suite is now being applied for all major operating systems instead of just linux 👩‍💻 

* 25/05/21

  - The new Danish Model Senda was added to DaCy

* 30/03/21

  - DaCy now includes a small model for efficient processing based on the Danish Ælæctra 🏃

* 24/03/21

  - DaCy included wrapped version on major Danish sentiment analysis software including the models by DaNLP. As well as code for wrapping any sequence classification model into its pipeline 🤩
  - Totorials is added to introduce the above functionality

* 25/02/21

  - DaCy launches with a medium-sized and a large language model obtaining state-of-the-art on Named entity recognition, part-of-speech tagging and dependency parsing for Danish 🇩🇰


Contents
---------------------------------
  
The documentation is organized in two parts:

- **Getting Started** contains the installation instructions on how to use DaCy.
- **Package References** contains the documentation of each public class and function.

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   installation
   usingdacy

.. toctree::
   :maxdepth: 3
   :caption: Package References

   download
   dacy.sentiment
   dacy.readability
   dacy.subclasses


.. toctree::
  GitHub Repository <https://github.com/KennethEnevoldsen/DaCy>

   


Indices and search
==================

* :ref:`genindex`
* :ref:`search`
