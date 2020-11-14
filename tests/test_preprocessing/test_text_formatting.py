import pytest
from treegoat import preprocessing
import numpy as np
from numpy import array_equal


@pytest.fixture
def text_transformer():
    return preprocessing.TextFormatting(top_words=100000,
                                        max_len=10)


@pytest.fixture
def training_data():
    return [
        "Lorem ipsum dolor sit amet",
        "hello consectetur adipiscing elit. Curabitur congue",
        "world consequat lorem a cursus. Aliquam sollicitudin"
    ]


@pytest.fixture
def test_data():
    return ["hello world"]


def test_fit_transform(text_transformer, training_data):
    transformed_data = text_transformer.fit_transform(training_data)
    assert array_equal(transformed_data,
                       np.array(
                           [
                            [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
                            [0, 0, 0, 0, 6, 7, 8, 9, 10, 11],
                            [0, 0, 0, 12, 13, 1, 14, 15, 16, 17],
                            ])
                       )


def test_transform(text_transformer, training_data):
    text_transformer.fit_transform(training_data)
    assert array_equal(text_transformer.transform(["hello world"]),
                       np.array([[0, 0, 0, 0, 0, 0, 0, 0, 6, 12]])
                       )
