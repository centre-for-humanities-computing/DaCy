from setuptools import setup


with open("dacy/about.py") as f:
    v = f.read()
    for line in v.split("\n"):
        if line.startswith("__version__"):
            __version__ = line.split('"')[-2]


def setup_package():
    setup(version=__version__)


if __name__ == "__main__":
    setup_package()
