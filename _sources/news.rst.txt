News and Updates
---------------------------------

**2.7.0 (15/05/23)** 

- Updated the DaCy models to version 0.2.0, including a small, medium and large

  - Added beta support for Coreference Resolution! 🤩
  - Added beta support for Named Entity Linking! 
  - Updated dependency parsing and part-of-speech tagging to use the latest version of the DDT treebank 🌳
  - Added a trainable lemmatizer, notably improving the lemmatization
  - All model are trained using the intersection between the CDT and the DDT treebanks (so actually trained on less data than before) 🤯

    - This includes the annotations from DaNED, DaCoref and DaNE

- Large model:

  - obtained state-of-the-art performance on:

    - Dependency parsing
    - Part-of-speech tagging
    - Morphological tagging
    - lemmatization (from 84.91 to 95.89!)

  - Improved performance on:

    - Reduced performance for NER down to 87.38 but we recommend either using the :code:`nlp.add_pipe("dacy/ner")` to add the SotA ScandiNER model to your pipeline or use one of the new fine-grained NER models.

  - Added support

    - Coreference Resolution, performance isn't great yet, but it's a start!
    - Named entity linking, with a precision of 0.86 but recall is still low due to a lacking knowledge base

- Medium model:

  - Consistent improvements across all tasks:

    - Notable performance gain for NER from an F1 of 81.79 to 85.82
    - Notable performance gain for lemmatization from an ACC 84.91 to 94

  - Added support

    - Coreference Resolution
    - Named entity linking

- Small model:

  - Consistent improvements across all tasks

  - Added support:

    - Coreference Resolution
    - Named entity linking

- Fixes a variety of issues:

  - `Removed custom requirements for the large model <https://github.com/centre-for-humanities-computing/DaCy/issues/218>`__
  - `Removing warning when loading the models <https://github.com/centre-for-humanities-computing/DaCy/issues/72>`__
  - Fixed annotatations in error in the DDT treebank where "'" was *never* followed by a space

- Removed support for DaCy model version 0.1.0, if you need to use these models you will have to use `DaCy <= 2.0.0`

- What is next?

  - A coreference resolution only model
  - Better named entity linking by

    - Improving the annotations of DaNED which currently annotate PERSONS using the QID reference to the name among other things
    - Improving the knowledge base which currently is the main source of low recall

  - Examine model generalization using DANSK and whether we can improve the generalization
  - and more!

**2.6.0 (10/04/23)**

- Added support three new models (small, medium, large) for fine-grained NER, which lets you do NER on up to 18 different entities! 🤩

  - You can add these models to your pipeline using :code:`nlp.add_pipe("dacy/ner-fine-grained", config={"size": "small"})`

**2.5.0 (10/04/23)**

- Removed support for `0.0.0` models. To use these models you will now have to use `DaCy=<2.4.3`

**2.3.0 (05/01/23)**

- New tutorial added for using DaCy with `textdescriptives`. You can find it `here <https://centre-for-humanities-computing.github.io/DaCy/using_dacy.html>`__

**2.2.10 (05/01/23)**

- Added support for spaCy 3.4.0

  - This required the wrapped component model to change name from e.g. :code:`dacy.ner` to :code:`dacy/ner`. As the :code:`.` is no longer allowed by spaCy.

- Added support for the state-of-the-art NER model by Dan Nielsen

  - You can add this model to your pipeline using :code:`nlp.add_pipe("dacy/ner")`


**2.0.0 (27/06/22): The Spandaur Update**

.. image:: _static/DALL-E_2022-07-27_A_minimalistic_2d_depiction_of_a_danish_cream_pastry.png
    :width: 100

- Added models for hate-speech detection and classification
- A large part of DaCy is now moved to seperate packages to allow for more versatility:

  - Now uses `spacy-wrap <https://github.com/KennethEnevoldsen/spacy-wrap>`__ for including existing models in DaCy.
  - Removed augmenters, they are now available through the external package `augmenty <https://kennethenevoldsen.github.io/augmenty/>`__
  - Removed the rule-based sentiment pipeline instead we recommend using `asent <https://github.com/KennethEnevoldsen/asent>`__

- Removed support for multiple installs, thus :code:`pip install dacy[all]` or :code:`dacy[large]` is no longer required. This should simplify installation processes and avoid errors
- Documentation

  - New `tutorial <https://centre-for-humanities-computing.github.io/DaCy/using_dacy.sentiment.html>`__ on using the sentiment models, including emotions detection, subjectivity detection and polarity classifcation.
  - New `tutorial <https://centre-for-humanities-computing.github.io/DaCy/using_dacy.hate_speech.html>`__ on using the hate speech classification and detection.
  - Multiple updated on function and package documentation

- Multiple bugfixes

**1.2.0 (04/11/21)**

- Removed DaNLP dependency, now DaNLP models is downloaded directly from Huggingface's model hub which is faster and more stable 🌟.
- Removed the readability module, we instead recommend you use the more extensive `textdescriptives <https://github.com/hlasse/TextDescriptives>`__ package developed by [HLasse](https://github.com/HLasse) and I for extracting readability and other text metrics.
- Added support for the configuring the default the model location with the environmental variable 'DACY_CACHE_DIR' thanks to a PR by `dhpullack <https://github.com/dhpollack>`__ 🙏.


**1.1.0 (23/07/21)**

- DaCy in now available on the `Huggingface model hub <https://huggingface.co/models?search=dacy>`__ 🤗 . Including detailed performance descriptions of biases and robustness.
- It also got a brand new online `demo <https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing>`__ - try it out!
- And more, including documentation update and prettier prints.

**1.0.0 (09/07/21)**

-  DaCy version 1.0.0 releases as the first version to pypi! 📦
    * Including a series of augmenters with a few specifically designed for Danish
    * Code for behavioural tests of NLP pipelines
    * And new tutorials for both 📖
- A new beautiful hand-drawn logo 🤩
- A behavioural test for biases and robustness in Danish NLP pipelines 🧐
- DaCy is now officially supported by the `Centre for Humanities Computing <https://chcaa.io/#/>`__ at Aarhus University
- The first paper on DaCy; check it out as a preprint and code for reproducing it `here <https://github.com/centre-for-humanities-computing/DaCy/tree/main/papers/DaCy-A-Unified-Framework-for-Danish-NLP>`__! 🌟 
  
**0.4.1 (03/06/21)**

- DaCy now has a stunningly looking documentation site 🌟

**0.3.1 (01/06/21)**

- DaCy's tests now cover 99% of its codebase 🎉
- DaCy's test suite is now being applied for all major operating systems instead of just Linux 👩‍💻 

**0.2.2 (25/05/21)**

- The new Danish Model Senda was added to DaCy

**0.2.1 (30/03/21)**

- DaCy now includes a small model for efficient processing based on the Danish Ælæctra 🏃

**0.1.1 (24/03/21)**

- DaCy includes a wrapped version of major Danish sentiment analysis software including the models by DaNLP, as well as code for wrapping any sequence classification model into its pipeline 🤩
- Tutorials is added to introduce the above functionality

**0.0.1 (25/02/21)**

- DaCy launches with a medium-sized and a large language model obtaining state-of-the-art on Named entity recognition, part-of-speech tagging and dependency parsing for Danish 🇩🇰
