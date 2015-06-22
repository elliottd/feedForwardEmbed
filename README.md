A Feed-forward Word Embedding Model
==========================================

A reimplementation of the feed-forward word embedding model of Bengio et al. 
(2000). The overall idea here is to learn a vector-based representation of
a word by training a model that predicts the next word in a fixed-length
sequence.

Dependencies
------------

* Python 2.7.2+
* Theano 0.70+
* numpy 1.8.2+
* [OpenBLAS](https://github.com/xianyi/OpenBLAS) for multi-core processing

Learning the embeddings
-----------------------

Run `python trainModel.py` to train a word embedding model over the
`raw_sentences.txt` file that accompanies this repository. 

The file will automatically be split into training, validation, and test data,
with `--unk=0` as a threshold for unknown words.  The model will be trained
over `--epochs=10` epochs, using a `--batch_size=100` instances, with a
sequence of `--ngram=3` tokens input and `--nlen=4` in total, meaning three
input tokens and one predicted token. The word embeddings will be
`--embed_size=50' dimensional, the hidden layer will have `--hidden_size=200`
dimensions. The optimiser will be `--optimiser=momentum`, with a
`--learning_rate=0.1` and a `--momentum=0.5`, `--decay=True` which lets the
learning rate decrease while training if validation perplexity does not
decrease for 10 epochs. The loss function is cross-entropy with
`--l2reg=0.0001` regularisation to the weights.

Run `python trainModel.py --help` to see the full list of available options,
including dropout, different optimisers, and initilising the model from a
previous saved state.

Find the nearest neighbours for a word
--------------------------------------

Run `python nearestWords.py --checkpoint checkpoints/$savedModel`, where
$savedModel is the directory name of a model that was saved to disk using
`trainModel`.

The nearest neighbour of a word is found by calculating the euclidean distance
between a candidate word and all other words in the embedding matrix.

Here are some of the nearest neighbours of a word-embedding model trained
on `raw_sentences.txt`. Your trained model may slightly differ due to
random variations in the data splitting and training shuffling.

| two     | women      | company    |
| ------- | ---------- | ---------- |
| four    | children   | department |
| three   | companies  | country    |
| five    | us         | program    |
| several | government | goverment  |
| old     | west       | directory  |

Predict the next word in a sequence
-----------------------------------

Run `python wordGeneration.py --checkpoint checkpoints/$savedModel`, where
$savedmodel contains a serialised model. A list of candidates is produced by
forward-propagating the word representations for the input tokens through the
model. The softmax probability distribution over the output layer produces an
ordered list of candidates for the next word.

An example of predicing the next word, given the sequence `i want them`.

    --->i want them
    i want them:
      to 0.265569995358
      . 0.182579378436
      , 0.0614877237544
      ? 0.0570122448425
      back 0.030883687082
      in 0.0214306911509
      out 0.0197824255555
      on 0.0155605708119
      for 0.0126796978053
      been 0.0121374341195

