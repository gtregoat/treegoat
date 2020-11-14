from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import List, Union
import pandas as pd
import numpy as np


class TextFormatting:
    def __init__(self, top_words: int = 100000, max_len: int = 15):
        """Tokenizes and creates a sequence of words.
        This uses Keras Tokenizer and Keras pad_sequence.

        :param top_words: int, max number of words in vocabulary.
        :param max_len: int, maximum length of a sequence of words. If less words in a sample, will be 0-padded. If
                more words, will be truncated.
        """
        self.tokenizer = None
        self.top_words = top_words
        self.max_len = max_len

    def fit_transform(self, x: Union[List[str], pd.Series]) -> np.array:
        """
        Fits the preprocessing (tokenizing and padding) to the text data.

        :param x: list of strings to be tokenized. Can be a list of strings or a pandas Series
        :return: numpy array (n_samples, max_len) where max_len is the sequence length
            defined in the __init__
        """
        self.tokenizer, x = self.tokenize(x,
                                          self.top_words,
                                          self.max_len)
        return x

    def transform(self, text_data: Union[List[str], pd.Series]) -> np.array:
        """Transforms the text data according to the transformation learned in fit_transform.

        :param text_data: list of strings to be tokenized. Can be a list of strings or a pandas Series
        :return:  numpy array (n_samples, max_len) where max_len is the sequence length
            defined in the __init__
        """
        sequences = self.tokenizer.texts_to_sequences(text_data)
        return pad_sequences(sequences, maxlen=self.max_len)

    @property
    def word_index(self):
        return self.tokenizer.word_index

    @staticmethod
    def tokenize(x: Union[List[str], pd.Series], top_words: int, max_len: int):
        """Preprocesses the text so it can then be fed to an embedding layer.

        :param x: list of strings to be tokenized. Can be a list of strings or a pandas Series
        :param top_words: int, maximum number of words in the dictionary
        :param max_len: int, length of a sequence (i.e. len(train[i]).
        :return: tokenizer, processed train data (array of integers from tokenizer.texts_to_sequences),
        processed test data
        """
        tokenizer = Tokenizer(num_words=top_words)
        tokenizer.fit_on_texts(x)
        train_sequences = tokenizer.texts_to_sequences(x)
        return tokenizer, pad_sequences(train_sequences, maxlen=max_len)
