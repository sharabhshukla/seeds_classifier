# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for seeds_classifier.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""
import pytest

import numpy as np

from seeds_classifier.seed_classifier import SeedClassifier


@pytest.fixture
def test_setup():
    model = SeedClassifier()
    return model


def test_class_predictions(test_setup):
    x_features_set = np.random.rand(10, 7)
    predictions = test_setup.classify(x_features_set)
    assert predictions.shape == (10,)
