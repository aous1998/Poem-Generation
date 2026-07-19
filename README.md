# Poem Generation

A university coursework project (Neural Networks course, 2022) that trains a model to
generate Arabic poetry in the style of a specific poet.

## What it does

- Loads an Arabic poetry dataset (poet name, poem text, category) from Google Drive.
- Provides helper functions to filter poems by poet or by category (e.g. by country).
- Tokenizes a poet's collected poems line by line and builds n-gram input sequences
  for next-word prediction, padded to a fixed length.
- Trains an LSTM / Bidirectional LSTM model (Embedding -> LSTM -> Dense with softmax
  over the vocabulary) to predict the next word given a sequence.
- Generates new text by repeatedly predicting and appending the next word to a seed
  phrase.

A written report (`report (1).pdf`) covering the approach and results is included
alongside the notebook.

## Tech stack

- Python
- TensorFlow / Keras (LSTM, Bidirectional, Embedding)
- pandas, NumPy
- Google Colab / Google Drive (the notebook was written to run in Colab and pulls
  its dataset from Drive)

## Running it

The notebook is written for Google Colab and expects Google Drive access to fetch
the dataset. Running it outside Colab requires swapping the Drive-download cells
for a local copy of the dataset.
