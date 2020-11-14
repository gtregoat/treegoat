from abc import ABC
from abc import abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer

from ..preprocessing import TextFormatting
from ..utils import load_glove_embeddings


class TextClassificationPipeline(ABC):
    def __init__(self,
                 sequence_length: int,
                 embeddings_dim: int,
                 embeddings_path: str = None):
        """
        This class takes care of putting the text preprocessing, label encoding and model into a classification
        pipeline.
        Labels are one-hot encoded with sklearn's LabelBinarizer, text is tokenized with the TextFormatting class
        in preprocessing, and the model is the TextClassifier in model.

        :param sequence_length: int, maximum number of words to take into account (the rest is truncated). If there are
            less words, zero-padding is applied.
        :param embeddings_dim: int, dimension of the word representations. In the case of pre-trained word embeddings
            (this package supports GloVe), this corresponds to the dimension of the word embeddings.
        :param embeddings_path: str, path to pre-trained word embeddings. If left to None, the embeddings are
            learned, otherwise they are loaded and the embeddings layer is not re-trained.
        """
        self.label_encoder = LabelBinarizer()
        self.text_formatter = TextFormatting(max_len=sequence_length)
        self.sequence_length = sequence_length
        self.vector_dim = embeddings_dim
        self._model = None
        self.embeddings_path = embeddings_path
        self._embeddings = None
        self.fitted = False
        self.label_dim = None

    def _fit(self, x, y, **fit_kwargs):
        self.fitted = False
        self.label_dim = y.shape[1]
        model = self.model
        model.fit(x,
                  y,
                  **fit_kwargs)
        self.model = model
        self.fitted = True
        return model

    def fit(self, x: pd.Series, y: pd.Series, **fit_kwargs):
        """
        Fits the model to the training data x and its associated labels y. The model will
        be recorded in self.model.
        :param x: pandas Series, training data where each sample contains a string with the search words.
            they will be tokenized and fed to the model.
        :param y: pandas series, training labels. They will be one-hot encoded and fed to the model.
        :param fit_kwargs: any arguments to pass to Keras' fit method, e.g. epochs, batch_size.
        """
        x = self.text_formatter.fit_transform(x)
        y_one_hot = self.label_encoder.fit_transform(y)
        if y_one_hot.shape[1] == 1:
            y_one_hot = np.hstack((y_one_hot, 1 - y_one_hot))
        self._fit(x, y_one_hot, **fit_kwargs)

    def cv(self, x: pd.Series, y: pd.Series, n_splits: int, refit: bool = True, **fit_kwargs) -> List[list]:
        """Performs cross validation and returns the scores.

        :param x: pandas Series, training data where each sample contains a string with the search words.
            they will be tokenized and fed to the model.
        :param y: pandas series, training labels. They will be one-hot encoded and fed to the model.
        :param n_splits: int, number of splits for the cross validation. There will be as many scores as there are
            splits.
        :param refit: bool, whether to refit the model after evaluating it with cross-validation.
        :param fit_kwargs: any arguments to pass to Keras' fit method, e.g. epochs, batch_size.
        :return: list of list [[loss, acuracy]]
        """
        x = self.text_formatter.fit_transform(x)
        y_one_hot = self.label_encoder.fit_transform(y)
        if y_one_hot.shape[1] == 1:
            y_one_hot = np.hstack((y_one_hot, 1 - y_one_hot))
        skf = StratifiedKFold(n_splits=n_splits)
        scores = []
        for train_index, test_index in skf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
            self._fit(x, y_one_hot, **fit_kwargs)
            results = self.model.evaluate(x_test, y_test)
            scores.append(results)
        if refit:
            self._fit(x, y_one_hot, **fit_kwargs)
        return scores

    def _predict(self, x):
        x = self.text_formatter.transform(x)
        return self.model.predict(x)

    def predict(self, x: Union[List[str], pd.Series]) -> np.array:
        """Generates predictions using the trained model and preprocessing.

        :param x: pandas series or list of strings.
        :return: predicted label.
        """
        predictions = self._predict(x)
        return self.label_encoder.inverse_transform(predictions)

    def predict_proba(self, x):
        """Returns the raw prediction (all probabilities for all classes)"""
        return self._predict(x)

    def set_embeddings(self):
        """
        This function is used in the property self.embeddings.
        """

    @property
    def embeddings(self):
        if self._embeddings is None:
            if self.embeddings_path is not None:
                # This is used to set the embeddings to either None or to load them after
                # the tokenizer (self.text_formatter) is fitted, as its attribute 'word_index'
                # is needed during this step.
                self._embeddings = load_glove_embeddings(path=self.embeddings_path,
                                                         embedding_dim=self.vector_dim,
                                                         word_index=self.text_formatter.word_index)
            else:
                self._embeddings = None
        return self._embeddings

    @property
    def model(self):
        if self.fitted:
            model = self._model
        else:
            model = self.build()
        return model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def input_dim(self):
        return len(self.text_formatter.word_index) + 1

    @abstractmethod
    def build(self, *args, **kwargs):
        pass
