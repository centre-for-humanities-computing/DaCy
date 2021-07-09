# DaCy: A Unified Framework for Danish NLP

## Abstract

Danish natural language processing (NLP) has in recent years obtained considerable improvements with the addition of multiple new datasets and models. However, at present, there is no coherent framework for applying state-of-the-art models for Danish. We present DaCy: a unified framework for Danish NLP built on SpaCy. DaCy uses efficient multitask models which obtains state-of-the-art performance on named entity recognition, part-of-speech tagging, and dependency parsing. DaCy contains tools for easy integration of existing models such as for polarity, emotion, or subjectivity detection. In addition, we conduct a series of tests for biases and robustness of Danish NLP pipelines through augmentation of the test set of DaNE. DaCy large compares favorably and is especially robust to long input lengths and spelling variations and errors. All models except DaCy small and large display consistent biases based on ethnicity and gender. We argue that for languages with limited benchmark sets, data augmentation can be particularly useful for obtaining more realistic and fine-grained performance estimates. We provide a series of augmenters as a first step towards more thorough evaluation of language models for low and medium resource languages and encourage further development.

## Reproducing the Paper

To reproduce the paper you will first need to install the required packages. Notice that this with require Python version 3.7.2 or higher. We suggest you make a virtual environment before doing this.

```
pip install -r requirements.txt
```

Do not that as we are testing a wide variety of packages not all package versions are compatible. Furthermore to test polyglot you will need to install additional dependencies outside og Python. You can follows the install instructions on their [website](https://polyglot.readthedocs.io/en/latest/Installation.html) for this. If you want to test the performance on GPU you will also need to [install](https://spacy.io/usage) the compatible cupy and spacy version for your CUDA driver.

After that simply running:

```
python dacy_paper_replication.py
```

Will generate a series of `.csv`s in `robustness` folder, under the name `robustness/{mdl}_augmentation_performance.csv`.
