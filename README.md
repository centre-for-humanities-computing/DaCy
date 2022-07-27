<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://github.com/centre-for-humanities-computing/DaCy/raw/main/docs/_static/icon_black_text.png" width="175" height="175" align="right" /></a>
# DaCy: An efficient NLP Pipeline for Danish

[![PyPI version](https://badge.fury.io/py/dacy.svg)](https://pypi.org/project/dacy/)
[![pip downloads](https://img.shields.io/pypi/dm/dacy.svg)](https://pypi.org/project/dacy/)
[![python version](https://img.shields.io/badge/Python-%3E=3.7-blue)](https://github.com/centre-for-humanities-computing/DaCy)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![license](https://img.shields.io/github/license/centre-for-humanities-computing/DaCy.svg?color=blue)](https://github.com/centre-for-humanities-computing/DaCy/blob/main/LICENSE)
[![spacy](https://img.shields.io/badge/built%20with-spaCy-09a3d5.svg)](https://spacy.io)
[![github actions pytest](https://github.com/centre-for-humanities-computing/DaCy/actions/workflows/pytest-cov-comment.yml/badge.svg)](https://github.com/centre-for-humanities-computing/Dacy/actions)
[![github actions docs](https://github.com/centre-for-humanities-computing/DaCy/actions/workflows/documentation.yml/badge.svg)](https://centre-for-humanities-computing.github.io/DaCy/)
<!-- 
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/af8637d94475ea8bcb6b6a03c4fbcd3e/raw/badge-dacy-pytest-coverage.json)
-->
[![CodeFactor](https://www.codefactor.io/repository/github/centre-for-humanities-computing/dacy/badge)](https://www.codefactor.io/repository/github/centre-for-humanities-computing/dacy)
[![Demo](https://img.shields.io/badge/Try%20the-Demo-important)](https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing.)



<!-- 
[![Known Vulnerabilities](https://snyk.io/test/github/KennethEnevoldsen/DaCy/badge.svg)](https://snyk.io/test/github/KennethEnevoldsen/DaCy)
<a href="https://doi.org/10.21105/joss.03153"><img alt="JOSS paper" src="https://joss.theoj.org/papers/10.21105/joss.03153/status.svg"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/trunajod">
[![Github All Releases](https://img.shields.io/github/downloads/centre-for-humanities-computing/dacy/total.svg)]()

-->

DaCy is a Danish natural language preprocessing framework made with SpaCy. Its largest pipeline has achieved State-of-the-Art performance on Named entity recognition, part-of-speech tagging and dependency parsing for Danish. Feel free to try out the [demo](https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing.). This repository contains material for using the DaCy, reproducing the results and guides on usage of the package. Furthermore, it also contains behavioural tests for biases and robustness of Danish NLP pipelines.

<!--
EASTER EGG:
https://www.youtube.com/watch?v=E7WQ1tdxSqI
-->


# 🔧 Installation
To get started using DaCy simply install it using pip by running the following line in your terminal:
```bash
pip install dacy
```


# 👩‍💻 Usage
To use the model you first have to download either the small, medium, or large model. To see a list of all available models:

```python
import dacy
for model in dacy.models():
    print(model)
# ...
# da_dacy_small_trf-0.1.0
# da_dacy_medium_trf-0.1.0
# da_dacy_large_trf-0.1.0
```

To download and load a model simply execute:
```python
nlp = dacy.load("da_dacy_medium_tfrf-0.1.0")
# or equivalently
nlp = dacy.load("medium")
```

Which will download the model to the `.dacy` directory in your home directory. 


To download the model to a specific directory:
```python
dacy.download_model("da_dacy_medium_trf-0.1.0", your_save_path)
nlp = dacy.load_model("da_dacy_medium_trf-0.1.0", your_save_path)
```

For more on how to use DaCy please check out our [documentation](https://centre-for-humanities-computing.github.io/DaCy/)

# 👩‍🏫 Tutorials and documentation

DaCy includes detailed documentation as well as a series of Jupyter notebook tutorial. If you do not have Jupyter Notebook installed, instructions for installing and running it can be found [here]( http://jupyter.org/install). All the tutorials are located in the `tutorials` folder.

|                                                                                                                                | Content                                                                             | Google Colab                                                                                                                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [🌟 Getting Started](https://centre-for-humanities-computing.github.io/DaCy/usingdacy.html)                                     | An introduction on how to use DaCy                                                  |                                                                                                                                                                                                         |
| [📖 Documentation](https://centre-for-humanities-computing.github.io/DaCy/)                                                     | The Documentation of DaCy                                                           |                                                                                                                                                                                                         |
| [😡😂 Sentiment](https://github.com/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-sentiment.ipynb)               | A simple introduction to the new sentiment features in DaCy.                        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-sentiment.ipynb)    |
| [🍒 Augmentation](https://github.com/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-augmentation.ipynb)          | A guide on how to augment text using the DaCy and SpaCy augmenters.                 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-augmentation.ipynb) |
| [⛑ Fairness and Robustness](https://github.com/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-robustness.ipynb) | A guide on how to use augmenters to measure model robustness and biases using DaCy. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/centre-for-humanities-computing/DaCy/blob/main/tutorials/dacy-robustness.ipynb)   |



# 🦾 Performance and Training

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dacy-a-unified-framework-for-danish-nlp/named-entity-recognition-on-dane)](https://paperswithcode.com/sota/named-entity-recognition-on-dane?p=dacy-a-unified-framework-for-danish-nlp)
  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dacy-a-unified-framework-for-danish-nlp/part-of-speech-tagging-on-dane)](https://paperswithcode.com/sota/part-of-speech-tagging-on-dane?p=dacy-a-unified-framework-for-danish-nlp)
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dacy-a-unified-framework-for-danish-nlp/dependency-parsing-on-dane)](https://paperswithcode.com/sota/dependency-parsing-on-dane?p=dacy-a-unified-framework-for-danish-nlp)

The following table shows the performance on the DaNE test set when compared to other models. Highest scores are highlighted with **bold** and second highest is <ins>underlined</ins>. 

<div align="center"><img src="img/perf.png"/></div>

Stanza uses the spacy-stanza implementation. The speed on the DaNLP model is as reported by the framework, which does not utilize batch input. However, given the model size, it can be expected to reach speeds comparable to DaCy medium. Empty cells indicate that the framework does not include the specific model.

<br /> 

<details>
  <summary> Training and reproduction </summary>

the folder `training` contains a SpaCy project which will allow for reproduction of the results. This folder also includes the evaluation metrics on DaNE and scripts for downloading the required data. For more information, please see the training [readme](training/readme.md).

Want to learn more about how DaCy initially came to be, check out this [blog post](https://www.kennethenevoldsen.com/post/new-fast-and-efficient-state-of-the-art-in-danish-nlp/).

</details>

<br /> 

## Robustness and Biases
DaCy compares the performance of Danish language processing pipeline under a large variaty of augmentations to test the robustness and biases hereof. To find out more please check the [website](https://centre-for-humanities-computing.github.io/DaCy/robustness.html).

# 🤔 Issues and Usage Q&A

To ask report issues or request features, please use the [GitHub Issue Tracker](https://github.com/centre-for-humanities-computing/DaCy/issues). Questions related to SpaCy are kindly referred to the SpaCy GitHub or forum.  Otherwise, please use the [discussion Forums](https://github.com/centre-for-humanities-computing/DaCy/discussions).



## Acknowledgements
DaCy is a result of great open-source software and contributors. It wouldn't have been possible without the work by the SpaCy team which developed and integrated the software. Huggingface for developing Transformers and making model sharing convenient. Multiple parties including Certainly.io and [Malte Hojmark-Bertelsen](https://github.com/MalteHB) for making their models publicly available. Alexandra Institute for developing and maintaining DaNLP which has made it easy to get access to Danish resources and even supplied some of the tagged data themselves.
