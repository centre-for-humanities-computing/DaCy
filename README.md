<a href="https://github.com/centre-for-humanities-computing/Dacy"><img src="https://github.com/centre-for-humanities-computing/DaCy/raw/main/docs/_static/icon_black_text.png" width="175" height="175" align="right" /></a>
# DaCy: An efficient NLP Pipeline for Danish

[![PyPI version](https://badge.fury.io/py/dacy.svg)](https://pypi.org/project/dacy/)
[![pip downloads](https://img.shields.io/pypi/dm/dacy.svg)](https://pypi.org/project/dacy/)
[![python version](https://img.shields.io/badge/Python-%3E=3.8-blue)](https://github.com/centre-for-humanities-computing/DaCy)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/centre-for-humanities-computing/DaCy/actions/workflows/tests.yml/badge.svg)](https://github.com/centre-for-humanities-computing/Dacy/actions)
[![github actions docs](https://github.com/centre-for-humanities-computing/DaCy/actions/workflows/documentation.yml/badge.svg)](https://centre-for-humanities-computing.github.io/DaCy/)
<!-- 
![github coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/KennethEnevoldsen/af8637d94475ea8bcb6b6a03c4fbcd3e/raw/badge-dacy-pytest-coverage.json)
-->
[![Demo](https://img.shields.io/badge/Try%20the-Demo-important)](https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing.)



DaCy is a Danish natural language preprocessing framework made with SpaCy. Its largest pipeline has achieved State-of-the-Art performance on Named entity recognition, part-of-speech tagging and dependency parsing for Danish. Feel free to try out the [demo](https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing.). This repository contains material for using DaCy, reproducing the results and guides on usage of the package. Furthermore, it also contains behavioural tests for biases and robustness of Danish NLP pipelines.

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


# 📖 Documentation

DaCy includes detailed documentation as well as a series of Jupyter notebook tutorials.
If you do not have Jupyter Notebook installed, instructions for installing and running
it can be found [here]( http://jupyter.org/install). All the tutorials are located in
the `tutorials` folder.


| Documentation              |                                                                                             |
| -------------------------- | ------------------------------------------------------------------------------------------- |
| 📚 **[Getting started]**    | Guides and instructions on how to use DaCy and its features.                                |
| 🦾 **[Performance]**        | A detailed description of the performance of DaCy and comparison with similar Danish models |
| 😎 **[Demo]**               | A simple Streamlit demo to try out the augmenters.                                          |
| 📰 **[News and changelog]** | New additions, changes and version history.                                                 |
| 🎛 **[API References]**     | The detailed reference for DaCy's API. Including function documentation                     |
| 🙋 **[FAQ]**                | Frequently asked questions                                                                  |


[Installation]: https://centre-for-humanities-computing.github.io/DaCy/installation.html
[Getting started]: https://centre-for-humanities-computing.github.io/DaCy/using_dacy.html
[api references]: https://centre-for-humanities-computing.github.io/DaCy/
[Demo]: https://huggingface.co/chcaa/da_dacy_medium_trf?text=DaCy+er+en+pipeline+til+anvendelse+af+dansk+sprogteknologi+lavet+af+K.+Enevoldsen%2C+L.+Hansen+og+K.+Nielbo+fra+Center+for+Humanities+Computing.
[News and changelog]: https://centre-for-humanities-computing.github.io/DaCy/news.html
[FAQ]: https://centre-for-humanities-computing.github.io/DaCy/faq.html
[Performance]: https://centre-for-humanities-computing.github.io/DaCy/performance.html





<br /> 

<details>
  <summary> Training and reproduction </summary>

the folder `training` contains a SpaCy project which will allow for reproduction of the results. This folder also includes the evaluation metrics on DaNE and scripts for downloading the required data. For more information, please see the training [readme](training/readme.md).

Want to learn more about how DaCy initially came to be, check out this [blog post](https://www.kennethenevoldsen.com/post/new-fast-and-efficient-state-of-the-art-in-danish-nlp/).

</details>

<br /> 


## 💬 Where to ask questions
To ask report issues or request features, please use the [GitHub Issue Tracker](https://github.com/centre-for-humanities-computing/DaCy/issues).
Questions related to SpaCy are kindly referred to the SpaCy GitHub or forum. Otherwise, please use the discussion Forums.

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[github issue tracker]: https://github.com/centre-for-humanities-computing/DaCy/issues
[github discussions]: https://github.com/centre-for-humanities-computing/DaCy/discussions






## Acknowledgements
DaCy is a result of great open-source software and contributors. It wouldn't have been possible without the work by the SpaCy team which developed and integrated the software. Huggingface for developing Transformers and making model sharing convenient. Multiple parties including Certainly.io and [Malte Hojmark-Bertelsen](https://github.com/MalteHB) for making their models publicly available. Alexandra Institute for developing and maintaining DaNLP which has made it easy to get access to Danish resources and even supplied some of the tagged data themselves.
