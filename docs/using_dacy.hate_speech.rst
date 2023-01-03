********************
Hate Speech
********************

`Hate speech <https://en.wikipedia.org/wiki/Hate_speech>`__ in text is often defined
text that expresses hate or encourages violence towards a person or group based on
something such as race, religion, sex, or sexual orientation".

DaCy currently does not include its own tools for hate-speech analysis, but incorperates existing
state-of-the-art models for Danish. The hate-speech model used in DaCy is
trained by [DaNLP](https://github.com/alexandrainst/danlp). It exists of two models.
One for detecting wether a text is hate speech laden and one for classifying the type
of hate speech.

+---------------------------------------+----------+----------+----------------------------------------------------------------------------+--------------------------------+
| Name                                  | Creator  | Domain   | Output Type                                                                | Model                          |
+=======================================+==========+==========+============================================================================+================================+
| code:`dacy/hatespeech_detection`      | `DaNLP`_ | Facebook | `["not offensive", "offensive"]`                                           | `Ælæctra`_                     |
+---------------------------------------+----------+----------+----------------------------------------------------------------------------+--------------------------------+
| code:`dacy/hatespeech_classification` | `DaNLP`_ | Facebook | `["særlig opmærksomhed", "personangreb", "sprogbrug", "spam & indhold"]`   | `Danish BERT by Certainly.io`_ |
+---------------------------------------+----------+----------+----------------------------------------------------------------------------+--------------------------------+

.. _DaNLP: https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/sentiment_analysis.md
.. _Danish BERT by Certainly.io: https://huggingface.co/Maltehb/danish-bert-botxo
.. _Ælæctra: https://huggingface.co/Maltehb/aelaectra-danish-electra-small-cased 

.. admonition:: Other models for Hate Speech detection
   
   There exist others models for Danish hate-speech detection. We have chosen the BERT
   offensive model as it obtains a reasonable trade-off between good
   `performance and speed <https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/hatespeech.md#-benchmarks>`__
   and includes a classification for classifying the type of hate-speech. The other models includes:
   
   - `A&ttack <https://github.com/ogtal/A-ttack>`__
   - `ELECTRA Offensive <https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/hatespeech.md#-electra-offensive-electra>`__
   - `BERT HateSpeech <https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/hatespeech.md#-bert-hatespeech-bertdr>`__
   - `Guscode/DKbert-hatespeech-detection <https://huggingface.co/Guscode/DKbert-hatespeech-detection>`__

Usage
#########

.. |colab_tut| image:: https://colab.research.google.com/assets/colab-badge.svg
   :width: 140pt
   :target: https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/master/tutorials/hate-speech.ipynb

|colab_tut|


To add the emotion models to your pipeline simply run:

.. code-block:: python

   nlp = spacy.blank("da") # create an empty pipeline

   # add the hate speech models
   nlp.add_pipe("dacy/hatespeech_detection")
   nlp.add_pipe("dacy/hatespeech_classification")

This wil set the two extensions to the Doc object, :code:`is_offensive` and :code:`hate_speech_type`.
These shows whether a text is emotionally laden and what emotion it contains.

Both of these also come with :code:`*_prob`-suffix if you want to examine the
probabilites of the models.

Let's look at an example using the model:

.. code-block:: python

   texts = [
      "senile gamle idiot", 
      "hej har du haft en god dag"
   ]

   # apply the pipeline
   docs = nlp.pipe(texts)

   for doc in docs:
      # print model predictions
      print(doc._.is_offensive)
      # print type of hate-speech if it is hate-speech
      if doc._.is_offensive == "offensive":
         print("\t", doc._.hate_speech_type)

   # offensive
   #    sprogbrug
   # not offensive
