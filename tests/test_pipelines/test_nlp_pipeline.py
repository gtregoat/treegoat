import pytest
from treegoat.pipelines import nlp_pipeline
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import Model

DATA = [
        "Lorem ipsum dolor sit amet",
        "hello consectetur adipiscing elit. Curabitur congue",
        "world consequat lorem a cursus. Aliquam sollicitudin"
    ]


class ClassificationModel(nlp_pipeline.TextClassificationPipeline):
    def build(self) -> Model:
        """Builds and returns the Keras classification model.
        The model is a CNN LSTM with the attention mechanism. It can integrate pre-trained word embeddings.
        Loss is categorical cross-entropy and the Adam algorithm is used to minimize it. The accuracy on
        the training data is recorded at each epoch.
        :return: Keras model
        """
        inputs = layers.Input(shape=(self.sequence_length,))
        # Pre-trained embeddings
        if self.embeddings is not None:
            vector_dim = self.embeddings.shape[1]
            representation = layers.Embedding(
                input_dim=self.input_dim,
                output_dim=vector_dim,
                weights=[self.embeddings],
                input_length=self.sequence_length,
                trainable=False)(inputs)  # The embedding weights will remain fixed as there isn't much data per class
        else:
            assert self.vector_dim is not None, "If not using pretrained embeddings, an embedding dimension has " \
                                           "to be provided."
            embedded = layers.Embedding(
                input_dim=self.input_dim,
                output_dim=self.vector_dim,
                input_length=self.sequence_length,
                trainable=True)(inputs)  # In this case there are no pre-trained embeddings so training is required
            conv = layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(embedded)
            pool = layers.MaxPooling1D(pool_size=2)(conv)
            pool = layers.BatchNormalization()(pool)
            recurrent = layers.LSTM(units=100, return_sequences=True)(pool)
            # compute importance for each step (attention mechanism)
            attention = layers.Dense(1, activation='tanh')(recurrent)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(100)(attention)
            attention = layers.Permute([2, 1])(attention)
            # Complete text representation
            representation = layers.Multiply()([recurrent, attention])
        embedded = layers.Flatten()(representation)

        # Classify
        classification = layers.Dense(10, activation="relu")(embedded)
        classification = layers.Dense(self.label_dim, activation="softmax")(classification)

        # Create the model
        model = Model([inputs], classification)

        # Compile
        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
        return model


@pytest.fixture
def pipeline():
    return ClassificationModel(sequence_length=15,
                               embeddings_path=None,
                               embeddings_dim=50)


@pytest.fixture
def training_set():
    return pd.DataFrame({"text": DATA}), pd.Series([0, 1, 0])


def test_fit(pipeline, training_set):
    pipeline.fit(training_set[0].loc[:, "text"], training_set[1], batch_size=1)
    assert pipeline.model.history.history['acc'][0] > 0  # If so the model has been fitted


def test_fit_predict(pipeline, training_set):
    pipeline.fit(training_set[0].loc[:, "text"], training_set[1], batch_size=1)
    assert len(pipeline.predict(["hello world"])) == 1


def test_cv(pipeline, training_set):
    n_splits = 2
    scores = pipeline.cv(training_set[0].loc[:, "text"], training_set[1], batch_size=1, n_splits=n_splits)
    assert len(scores) == n_splits
