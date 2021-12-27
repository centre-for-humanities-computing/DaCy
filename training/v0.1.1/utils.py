def fetch_dane_as_conllu():
    from danlp.datasets import DDT

    ddt = DDT()
    train, dev, test = ddt.load_as_conllu(predefined_splits=True)
    with open("assets/dane/dane_train.conllu", "w") as f:
        train.write(f)
    with open("assets/dane/dane_dev.conllu", "w") as f:
        dev.write(f)
    with open("assets/dane/dane_test.conllu", "w") as f:
        test.write(f)


if __name__ == "__main__":
    fetch_dane_as_conllu()
