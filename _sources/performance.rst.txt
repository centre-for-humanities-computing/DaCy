General performance
==================================================================================

In the paper `DaCy: A Unified Framework for Danish NLP <https://github.com/centre-for-humanities-computing/DaCy/blob/main/papers/DaCy-A-Unified-Framework-for-Danish-NLP/readme.md>`__
we compare DaCy's models with other Danish language processing pipelines. This page represents only parts of the paper. For a more comprehensive evaluation we recommend reading the paper.

The table below shows the performance of Danish language processing pipelines scored on the DaNE test set, including part-of-speech tagging (POS),
named entity recognition (NER) and dependency parsing.
The best scores in each category are highlighted with bold and the second best is underlined.
Empty cells indicate that the framework does not include the specific model.


.. image:: ../img/perf.png
  :width: 1000
  :alt: Performance of Danish NLP pipelines


The speed of the DaNLP model is as reported by the framework, which does not utilize batch input. 
However, given the model size, it can be expected to reach speeds comparable to DaCy medium. 


.. admonition:: What is LAS and UAS?
   :class: note

   Unlabelled attachment score (UAS) denotes the percentage of words that get assigned the correct head,
   while labelled attachment score (LAS) is the percentage of words that get assigned the correct head and label. 
   For more information, read the following `chapter <https://web.stanford.edu/~jurafsky/slp3/14.pdf>`__
   by Jurafsky and Martin.

From the table we see that DaCy large obtains state-of-the-art on all tasks, most notably on NER
and dependency parsing. DaCy medium is a good alternative especially when running on CPU, where SpaCy large might also be considered.
If you are only interested in NER, and POS, Flair is also a viable option for CPU usage.

Measuring performance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typically when measuring performance on these benchmark there is a tendency to feed the model the gold standard tokens. 
While this allows for easier comparisons of modules and architectures, it inflates the performance metrics. Further, it does not proberly reflect what you are really interested in:
the performance you can expect when you apply the model. Therefore, we measure the performance using the models own tokenizer or SpaCy's tokenizer if it performs better.
Polyglot and Stanza performed better with their own tokenizers while the remaining models performed best with SpaCy's.

Benchmark performance only scratches the surface though, so be sure to check out the more nuanced section on model `robustness and biases <https://centre-for-humanities-computing.github.io/DaCy/robustness.html>`__.