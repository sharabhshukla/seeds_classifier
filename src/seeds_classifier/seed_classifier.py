import os

from joblib import load


class SeedClassifier:

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )

    def __init__(self, model_path=os.path.join(__location__, "models", "clf_pipe")):
        self._model = load(model_path)

    def classify(self, x_features):
        return self._model.predict(x_features)


if __name__ == "__main__":
    SeedClassifier()
