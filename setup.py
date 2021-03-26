import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="dacy",
    version="0.0.1",
    description="a Danish preprocessing pipeline trained in SpaCy. \
        At the time of writing it has achieved State-of-the-Art \
            performance on all Benchmark tasks for Danish",
    license="Apache License 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kenneth C. Enevoldsen",
    author_email="kennethcenevoldsen@gmail.com",
    url="https://github.com/KennethEnevoldsen/dacy",
    packages=setuptools.find_packages(),
    # external packages as dependencies
    install_requires=["spacy", "spacy-transformers", "tqdm", "danlp"],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 4 - Beta",
        # Indicate who your project is intended for
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: >=3.6",
    ],
    keywords="NLP danish",
)