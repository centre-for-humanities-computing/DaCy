********************
Sentiment Analysis
********************

Sentiment analysis (or opinion mining) is a method used to determine whether text is
positive, negative or neutral. Sentiment analysis is e.g. used by businesses to monitor
brand and product sentiment in customer feedback, or in research to e.g. examine
`political biases <https://tidsskrift.dk/lwo/article/view/96014>`__.


Sentiment analysis can be split into rule-based and neural approaches. Rule-based
approaches typically used a dictionary of rated positive and negative words and employs
a series of rules such as negations to estimate whether a text is postive and negative.

Typically rules-based approaches are notably faster, but performs worse compared to their
neural counterpart especially on more complex sentiment such as sarcasm where it is hard
to defined clear rules. It is thus important to take this into consideration when
choosing between the models.

Overview of Models
###################

DaCy include a variety of models for sentiment analysis including whether a text
is subjective or objective, the polarity of the text (postive, negative or neutral) and
even the emotion of given text. Its rule-based component also allow more fine-grained
analysis examining positive or negative words along with the dependency structure. 


+---------------------+-----------+-----------------------------+-----------------------------------------------------------------------------+-----------------------------------------+
| Name                | Reference | Domain                      | Output Type                                                                 | Model Type                              |
+=====================+===========+=============================+=============================================================================+=========================================+
| `dacy/subjectivity` | `DaNLP`_  | Europarl and Twitter        | `["objective", "subjective"]`                                               | Neural (`Danish BERT by Certainly.io`_) |
+---------------------+-----------+-----------------------------+-----------------------------------------------------------------------------+-----------------------------------------+
| `dacy/polarity`     | `DaNLP`_  | Europarl and Twitter        | ['postive', 'neutral', 'negative']`                                         | Neural (`Danish BERT by Certainly.io`_) |
+---------------------+-----------+-----------------------------+-----------------------------------------------------------------------------+-----------------------------------------+
| `dacy/emotion`      | `DaNLP`_  | Social Media                | `["Emotional", "No emotion"] and ["Glæde/Sindsro", "Tillid/Accept", ... ]`  | Neural (`Danish BERT by Certainly.io`_) |
+---------------------+-----------+-----------------------------+-----------------------------------------------------------------------------+-----------------------------------------+
| `asent_da_v1`       | `Asent`_  | Microblogs and Social media | `Polarity score (continuous)`                                               | Rule-based                              |
+---------------------+-----------+-----------------------------+-----------------------------------------------------------------------------+-----------------------------------------+


.. _DaNLP: https://github.com/alexandrainst/danlp/blob/master/docs/docs/tasks/sentiment_analysis.md
.. _Asent: https://kennethenevoldsen.github.io/asent/index.html
.. _Danish BERT by Certainly.io: https://huggingface.co/Maltehb/danish-bert-botxo


Usage
#########


.. tab-set::


    .. tab-item:: Neural Models

      .. |colab_tut_neural| image:: https://colab.research.google.com/assets/colab-badge.svg
         :width: 140pt
         :target: https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/master/tutorials/sentiment-neural.ipynb

      |colab_tut_neural|

      DaCy currently does not include its own tools for sentiment extraction, but a couple of good tools already exists. DaCy providers wrappers for these to use them in the spaCy/DaCy framework. This allows you to get all of the best models in one place.

      to get started we will need to load the dacy and spacy:

      .. code-block:: python

         import dacy
         import spacy

      
      .. dropdown:: Subjectivity
         :open:

         The subjectivity model is a part of BertTone a model trained by
         `DaNLP <https://github.com/alexandrainst/danlp>`__. The models detect whether a
         text is subjective or not.

         To add the subjectivity model to your pipeline simply run:

         .. code-block:: python

            nlp = spacy.blank("da") # an empty spacy pipeline
            nlp.add_pipe("dacy.subjectivity")


         This will add the :code:`dacy.subjectivity` component to your pipeline, which adds
         two extensions to the Doc object,:code:`subjectivity_prob` and :code:`subjectivity`.
         These show the probabilities of a document being subjective and whether not a
         document is subjective or objective. Let's look at an example using the model:

         .. code-block:: python

            texts = [
               "Analysen viser, at økonomien bliver forfærdelig dårlig",
               "Jeg tror alligevel, det bliver godt",
            ]

            docs = nlp.pipe(texts)  # run the model

            for doc in docs:
               # print the model predictions
               print(doc._.subjectivity)
               print(doc._.subjectivity_prob)

            # objective
            # {'prob': array([1., 0.], dtype=float32), 'labels': ['objective', 'subjective']}
            # subjective
            # {'prob': array([0., 1.], dtype=float32), 'labels': ['objective', 'subjective']}


      .. dropdown:: Polarity

         Similar to the subjectivity model, the polarity model is a of the BertTone model. 
         This model classifies the polarity of a text, i.e. whether it is positve,
         negative or neutral.

         To add the polarity model to your pipeline simply run:

         .. code-block:: python

            nlp = spacy.blank("da") # an empty spacy pipeline
            nlp.add_pipe("dacy/polarity")

         This will add the :code:`dacy/polarity` component to your pipeline, which adds
         two extensions to the Doc object,:code:`polarity_prob` and :code:`polarity`.
         These show the probabilities of a document being positive/neutral/negative and
         the resulting classification. Let's look at an example using the model:

         .. code-block:: python

            texts = [
               "Analysen viser, at økonomien bliver forfærdelig dårlig",
               "Jeg tror alligevel, det bliver godt",
            ]

            # apply the pipeline
            docs = nlp.pipe(texts)  # run the texts through the pipeline

            for doc in docs:
               # print the model predictions
               print(doc._.polarity)
               print(doc._.polarity_prob)

            # negative
            # {'prob': array([0.002, 0.008, 0.99 ], dtype=float32), 'labels': ['positive', 'neutral', 'negative']}
            # positive
            # {'prob': array([0.981, 0.019, 0.   ], dtype=float32), 'labels': ['positive', 'neutral', 'negative']}

      .. dropdown:: Emotion

         The emotion model used in DaCy is trained by
         `DaNLP <https://github.com/alexandrainst/danlp>`__. It exists of two models.
         One for detecting wether a text is emotionally laden and one for classifying
         which emotion it is out of the following emotions:

         - "Glæde/Sindsro" (happiness)
         - "Tillid/Accept" (trust/acceptance)
         - "Forventning/Interrese" (interest)
         - "Overasket/Målløs" (surprise)
         - "Vrede/Irritation" (Anger)
         - "Foragt/Modvilje" (Contempt)
         - "Sorg/trist" (Sadness)
         - "Frygt/Bekymret" (Fear)

         To add the emotion models to your pipeline simply run:

         .. code-block:: python

            nlp = spacy.blank("da") # create an empty pipeline

            # add the emotion compenents to the pipeline
            nlp.add_pipe("dacy/emotionally_laden")
            nlp.add_pipe("dacy/emotion")

         This wil set the two extensions to the Doc object, :code:`laden` and :code:`emotion`.
         These shows whether a text is emotionally laden and what emotion it contains.
         Both of these also come with :code:`*_prob`-suffix if you want to examine the
         probabilites of the model.
         
         Let's look at an example using the model:

         .. code-block:: python

            texts = [
               "Ej den bil er såå flot",
               "Fuck det er bare så FUCKING træls!",
               "Har i set at Tesla har landet en raket på månen? Det er vildt!!",
               "der er et træ i haven"
            ]

            docs = nlp.pipe(texts)

            for doc in docs:
               print(doc._.emotionally_laden)
               # if emotional print the emotion
               if doc._.emotionally_laden == "emotional":
                  print("\t", doc._.emotion)

            # emotional
            #    tillid/accept
            # emotional
            #    sorg/trist
            # emotional
            #    overasket/målløs
            # no emotion




    .. tab-item:: Rule-based Models

      .. |colab_tut_rule| image:: https://colab.research.google.com/assets/colab-badge.svg
         :width: 140pt
         :target: https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/master/tutorials/sentiment-rule-based.ipynb

      |colab_tut_rule|

      if you wish to perform rule-based sentiment analysis using DaCy we recommend using
      `Asent <https://github.com/KennethEnevoldsen/asent>`__. Asent is a rule-based sentiment
      analysis library for performing sentiment analysis for multiple languages including
      Danish.

      To get started using Asent install it using:

      .. code-block:: bash

         pip install asent


      first we will need to set up the spaCy pipeline, which only need to include a method for
      creating sentences. You can use DaCy for this as it performs dependendency parsing, but
      it is notably faster to use a rule-based sentencizer. 

      .. code-block:: python

         import spacy
         import asent

         # load a spacy pipeline
         # equivalent to a dacy.load()
         # but notably faster
         nlp = spacy.blank("da")
         nlp.add_pipe("sentencizer")

         # add the rule-based sentiment model from asent.
         nlp.add_pipe("asent_da_v1")

         # try an example
         text = "jeg er ikke mega glad."
         doc = nlp(text)

         # print polarity of document, scaled to be between -1, and 1
         print(doc._.polarity)
         # neg=0.413 neu=0.587 pos=0.0 compound=-0.5448


      Asent also allow us to obtain more information such as the rated valence of a single
      token, whether a word is a negation or the valence of a words accounting for its context
      (polarity):


      .. code-block:: python

         for token in doc:
            print(f"{token._.polarity} | Valence: {token._.valence} | Negation: {token._.is_negation}")

         # polarity=0.0 token=jeg span=jeg | Valence: 0.0 | Negation: False
         # polarity=0.0 token=er span=er | Valence: 0.0 | Negation: False
         # polarity=0.0 token=ikke span=ikke | Valence: 0.0 | Negation: True
         # polarity=0.0 token=mega span=mega | Valence: 0.0 | Negation: False
         # polarity=-2.516 token=glad span=ikke mega glad | Valence: 3.0 | Negation: False
         # polarity=0.0 token=. span=. | Valence: 0.0 | Negation: False

      Here we see that words such as *"glad"* (happy) is rated positively (valence), but
      accounting for the negation *"ikke"* (not) it becomes negative. Furthermore, Asent also allows you to visualize the predictions: 

      .. admonition:: Learn more
         :class: hint

         If you want to learn more about how asent works check out the excellent `documentation <https://kennethenevoldsen.github.io/asent/introduction.html>`__.

         
      .. code-block:: python

         # visualize model prediction
         asent.visualize(doc, style="prediction")

      .. image:: _static/asent_prediction.png
         :width: 300
         :alt: Visualization of model prediction using Asent

      .. code-block:: python

         # visualize the analysis performed by the model:
         asent.visualize(doc, style="analysis")

      .. image:: _static/asent_analysis.png
         :width: 800
         :alt: Visualization of model analysis using Asent 


      .. seealso::

         Looking for the rule-based component :code:`dacy.sentiment.getters.da_vader_getter`?
         It removed in favor of `asent <https://github.com/KennethEnevoldsen/asent>`__.
         Asent contains the same functionality along while allowing for more customizability
         and includes visualizers. Asent is developed jointly with DaCy and are thus designed
         to be compatible.  

      .. admonition:: Other resources

         Danish has two other rule-based language models including 
         `AFINN <https://github.com/fnielsen/afinn>`, which does not implement any rules such as
         negations and `sentida <https://github.com/Guscode/Sentida>`__ which does use rules
         similarly to asent, but simplifies the 


