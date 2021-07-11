DaCy
================================

.. image:: https://img.shields.io/github/stars/centre-for-humanities-computing/DaCy.svg?style=social&label=Star&maxAge=2592000
   :target: https://github.com/centre-for-humanities-computing/DaCy

DaCy is a Danish text processing pipeline built using SpaCy. At the time of writing, it has achieved State-of-the-Art performance on part-of-speech (POS) tagging, 
named-entity recognition (NER) and Dependency parsing for Danish. 

This website contains the documentation for DaCy as well as an introduction to how to get started using DaCy for your project.

📰 News
---------------------------------

* 1.0.0 (09/07/21)

  -  DaCy version 1.0.0 releases as the first version to pypi! 📦
    - Including a series of augmenters with a few specifically designed for Danish
    - Code for behavioural tests of NLP pipelines
    - A new tutorial for both 📖
  - A new beautiful hand-drawn logo 🤩
  - A behavioural test for biases and robustness in Danish NLP pipelines 🧐
  - DaCy is now officially supported by the `Centre for Humanities Computing <https://chcaa.io/#/>`__ at Aarhus University
  - The first paper on DaCy; check it out as a preprint and code for reproducing it `here <https://github.com/centre-for-humanities-computing/DaCy/tree/main/papers/DaCy-A-Unified-Framework-for-Danish-NLP>`__! 🌟 
    
* 0.4.1 (03/06/21)

  - DaCy now has a stunningly looking documentation site 🌟

* 0.3.1 (01/06/21)

  - DaCy's tests now cover 99% of its codebase 🎉
  - DaCy's test suite is now being applied for all major operating systems instead of just Linux 👩‍💻 

* 0.2.2 (25/05/21)

  - The new Danish Model Senda was added to DaCy

* 0.2.1 (30/03/21)

  - DaCy now includes a small model for efficient processing based on the Danish Ælæctra 🏃

* 0.1.1 (24/03/21)

  - DaCy includes a wrapped version of major Danish sentiment analysis software including the models by DaNLP, as well as code for wrapping any sequence classification model into its pipeline 🤩
  - Tutorials is added to introduce the above functionality

* 0.0.1 (25/02/21)

  - DaCy launches with a medium-sized and a large language model obtaining state-of-the-art on Named entity recognition, part-of-speech tagging and dependency parsing for Danish 🇩🇰


Contents
---------------------------------
  
The documentation is organized in three parts:

- **Getting Started** contains the installation instructions, guides, and tutorials on how to use DaCy.
- **Performance** contains a series of performance metrics and comparisons of DaCy and other Danish NLP pipelines.
- **Package References** contains the documentation of each public class and function.

.. toctree::
   :maxdepth: 3
   :caption: Getting Started

   installation
   usingdacy

.. toctree::
   :maxdepth: 3
   :caption: Performance

   performance
   robustness


.. toctree::
   :maxdepth: 3
   :caption: Package References

   download
   dacy.datasets
   dacy.sentiment
   dacy.augmenters
   dacy.readability
   dacy.subclasses



.. toctree::
  GitHub Repository <https://github.com/centre-for-humanities-computing/DaCy>

   


Indices and search
==================

* :ref:`genindex`
* :ref:`search`
