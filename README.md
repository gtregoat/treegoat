# treegoat
Helper functions for building machine learning pipelines from exploration to production.


## Structure
```text
- treegoat
    - analytics: functions to analyse data (e.g. comparing to a label, histograms...)
    - model_inspection: functions to explore a built model (e.g. looking at the rules of a decision tree with graphviz)
    - pipelines: inheritable classes to ease wrapping models in a production pipeline (e.g. building a text classification pipeline by inheriting the nlp pipline only requires defining a build function with the keras model)
    - preprocessing: scikit-learn like transformers (with fit_transform and transform methods) to transform data before feeding a model
    - utils: various functions useful in other parts of this library (e.g. loading GloVe word embeddings)
- tests
    - one test package per sub treegoat package
        - one test module per module
    - ...
```

### NLP pipeline
Build only the keras model and have the text formatting / tokenizing done behind the hood.

#### Example
##### Building the model
```python
from tensorflow.keras import layers
from tensorflow.keras import Model
from treegoat.pipelines import nlp_pipeline


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
```
##### Using the model
###### Fitting the model
```python
pipeline = ClassificationModel(sequence_length=15,
                               embeddings_path=None,
                               embeddings_dim=50)
pipeline.fit(x, y, batch_size=1, **any_other_args)
```
- x is a pandas DataFrame
- y is a pandas Series (labels will be one-hot encoded)
- any other arguments (e.g. batch size) will be passed to the keras fit method

###### Performing cross-validation
```python
n_splits = 2
scores = pipeline.cv(training_set[0].loc[:, "text"], training_set[1], batch_size=1, n_splits=n_splits)
```

###### Accessing the model
```python
pipeline.model
```
This will return the fitted model if the fit method has been called or the cv method has been called
with "refit=True". Otherwise, it will create a new instance of the model using the build method 
that has been custom-defined.

### Requirements
- scikit-learn
- pandas
- numpy
- pyspark
- tensorflow

### Running the tests
Running with pytest, go to the root and run:
```shell script
python -m pytest tests
```
