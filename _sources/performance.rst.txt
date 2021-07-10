General performance
==================================================================================

In the paper `DaCy: A Unified Framework for Danish NLP <https://github.com/centre-for-humanities-computing/DaCy/blob/main/papers/DaCy-A-Unified-Framework-for-Danish-NLP/readme.md>`__
we compare DaCy's models with other Danish language processing pipelines. This page represents only parts of the paper for the comprehensive introduction, we recommend reading the paper.

In the table below you see the performance of Danish language processing pipelines scored on the DaNE test set, including part-of-speech tagging (POS),
named entity recognition (NER) and dependency parsing.
The best score are highlighted with bold while the second best is underlined.
Empty cells indicate that the framework does not include the specific model.


.. image:: ../img/perf.png
  :width: 1000
  :alt: Performance of Danish NLP pipelines


The speed on the DaNLP model is as reported by the framework, which does not utilize batch input. 
However, given the model size, it can be expected to reach speeds comparable to DaCy medium. 


.. admonition:: What is LAS and UAS?
   :class: note

   Unlabelled attachment score (UAS) denotes the percentage of words that get assigned the correct head,
   while labelled attachment score (LAS) is the percentage of words that get assigned the correct head and label. 
   Unsure of what a head is? Then i recommend this `chapter <https://web.stanford.edu/~jurafsky/slp3/14.pdf>`__
   by Jurafsky and Martin.

From the table we see that DaCy large has state-of-the-art on all tasks, most notably on NER
and dependency parsing with DaCy Medium providing a good alternative especially when running on CPUs,
where one might even consider running the SpaCy large and if you are only interested in NER and speed Flair might 
provide a reasonable model.

Measuring performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically when measuring performance on these benchmark there is a tendency to feed the model the gold standard tokens. 
While this might seem reasonable at first as it allows for easier comparisons of modules and architectures, it does
inflate the performance metrics. What is more important it does not reflect what you are probably really interested in,
the performance you get when you apply the model. Thus here we measure the performance using the models own tokenizer or SpaCy tokenizer if it performs better.
Polyglot and Stanza performing better with their own tokenizers while the remaining models perform best with SpaCy's.

Benchmark performance is rarely everything though, so be sure to check out the more nuanced section on model `robustness and biases <https://centre-for-humanities-computing.github.io/DaCy/robustness.html>`__.