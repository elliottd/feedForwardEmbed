import argparse
import sys
import math
import time
import os
import shutil
from time import gmtime, strftime

import numpy
from numpy import sqrt, zeros
from numpy.random import randn
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from loadData import LoadData

theano.config.optimizer="fast_compile"

class WordEmbedder:

  def __init__(self, args, vocab_size):
    self.args = args
    self.vocab_size = vocab_size
    self.names = ['embeddings', 'hidden-embeddings', 'output-hidden', 
                  'bias-hidden', 'bias-output']

    ''' The word embeddings is a matrix of dim:
          |vocabulary| x |embedding layer| '''
    emb = self.initWeights(vocab_size+1, 
                           args.embed_size, 
                           args.initialisation_checkpoint, 
                           self.names[0])

    ''' The embedding -> hidden layer weights is a matrix of dim:
         |embedding layer| * |input| x |hidden layer| '''
    whe = self.initWeights(args.embed_size * args.ngram, 
                           args.hidden_size, 
                           args.initialisation_checkpoint, 
                           self.names[1])

    ''' The hidden -> output layer weights is a matrix of dim:
          |hiden layer| x |vocabulary| '''
    woh = self.initWeights(args.hidden_size, 
                           vocab_size+1, 
                           args.initialisation_checkpoint, 
                           self.names[2])

    ''' Bias for the hidden layer '''
    bh = self.initBiases(args.hidden_size, 
                         args.initialisation_checkpoint, 
                         self.names[3])

    ''' Bias for the output layer '''
    bo = self.initBiases(vocab_size+1, 
                         args.initialisation_checkpoint, 
                         self.names[4])

    self.params = [emb, whe, woh, bh, bo]

    ''' Random Number Generator used to drop units '''
    self.srng = RandomStreams()

    ''' Theano types for the input, output, and hyper-parameters '''

    word_indices = T.imatrix() # Input sequences are a matrix of integers
                               # with one line per instance.

    # Reshape the input into a vector d = |1| x |embedding size * ngramlen|
    x = emb[word_indices].reshape((word_indices.shape[0], 
                                   args.embed_size * args.ngram)) 

    y = T.imatrix('y') # Output targets are a matrix of integers with
                       # one per instance. 

    # Alternative input type used for word-word similarities
    word = T.iscalar()
    single_x = emb[word]

    # weight update hyperparameters
    lr = T.scalar('lr')
    mom = T.scalar('mom')
    dropin = T.scalar('dropin')
    droph = T.scalar('droph')
    l1reg = T.scalar('l1reg')
    l2reg = T.scalar('l2reg')

    ''' Train model, loss, and pplx (includes dropout parameters in the model'''
    h, py_x = self.model(x, whe, woh, bh, bo,     # forward prop using
                           dropin=dropin, # dropout in the input
                           droph=droph)   # and hidden layer units

    cost = T.mean(T.nnet.categorical_crossentropy(py_x, y))

    # Add regularization weights to the cost. TODO: necessary for val/test?

    cost += sum([T.sum(x) for x in self.params]) * l1reg # L1 regularisation
    cost += sum([T.sum([T.sqr(x)]) for x in self.params]) * l2reg # L2

    pplx = T.pow(2, cost) # 2^normalised cross-entropy

    gradients = T.grad(cost, self.params)
    y_x = T.argmax(py_x, axis=1) # sample the max prob word from test model

    ''' Train the parameters of the model, given a sequence of words, a target
        word, and a learning rate. Selects the optimizer based on the user
        requirements.'''
    self.train = theano.function(inputs=[word_indices, y, lr, mom, dropin, droph, l1reg, l2reg],
                                 outputs=[cost, pplx],
                                 updates=self.optimizer(self.params, gradients, lr, mom),
                                 allow_input_downcast=True, on_unused_input='ignore')

    '''Validation is where we can test whether reducing the training loss
       is helping to reduce our loss on "unseen" data'''
    self.validate = theano.function(inputs=[word_indices, y, dropin, droph, l1reg, l2reg], 
                                    outputs=[cost, pplx, y_x], 
                                    allow_input_downcast=True, on_unused_input='ignore')

    '''Predict is used to retrieve the softmax vector over all predicted
       words, given an input sequence.'''
    self.predict = theano.function(inputs=[word_indices, dropin, droph, l1reg, l2reg], 
                                   outputs=py_x, 
                                   allow_input_downcast=True, on_unused_input='ignore')

    ''' Normalise the weights in the embedding layer to be within the unit
        sphere. TODO: understand why?'''
    self.normalize = theano.function(inputs=[],
                                     updates={emb:emb / T.sqrt((emb**2).sum(axis=1)).dimshuffle(0, 'x')})

  '''
  Initialise the weights from scratch of unpickle them from disk
  '''
  def initWeights(self, xdim, ydim, initialisation_checkpoint=None, name=None):
    if initialisation_checkpoint == None:
      return theano.shared(0.01 * randn(xdim, ydim).astype(theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (initialisation_checkpoint, name)).astype(theano.config.floatX))

  '''
  Initialise the bias from scratch of unpickle them from disk
  '''
  def initBiases(self, dims, initialisation_checkpoint=None, name=None):
    if initialisation_checkpoint == None:
      return theano.shared(zeros(dims, dtype=theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (initialisation_checkpoint, name)).astype(theano.config.floatX))

  '''
  The model is a simple multi-layer perceptron.
  h is the hidden layer, which has sigmoid activations from the embeddings
  py_x is the output layer, which has softmax activations from the hidden

  dropin: probabilty of dropping embedding units
  droph: probability of dropping hidden units
  '''
  def model(self, e, whe, woh, bh, bo, dropin=0., droph=0.):
    e = self.dropout(e, dropin)
    h = self.dropout(T.tanh(T.dot(e, whe) + bh), droph)
    py_x = T.nnet.softmax(T.dot(h, woh) + bo)
    return h, py_x

  '''
  Dropout units in the layer X with probability given by p.
  '''
  def dropout(self, X, p=0.):
    if T.gt(p, 0.):
        retain_prob = 1 - p
        X *= self.srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

  '''
  Selects an optimization function for minimizing the cost of the model
  '''
  def optimizer(self, params, grads, lr, mom):
    if self.args.optimizer == "sgd":
      return self.sgd_updates(params, grads, lr)
    if self.args.optimizer == "momentum":
      return self.momentum_updates(params, grads, lr, mom)
    if self.args.optimizer == "nesterov":
      return self.nesterov_updates(params, grads, lr, mom)
    if self.args.optimizer == "adagrad":
      return self.adagrad_updates(params, grads, lr)

  '''
  Stochastic gradient descent updates:
    weight = weight - learning_rate * gradient
  '''
  def sgd_updates(self, params, gradients, learning_rate):
    updates = []
    for p,g in zip(params, gradients):
      updates.append((p, p - learning_rate*g))
    return updates

  '''
  Momentum weight updates; initial velocity = 0
    weight = weight + velocity
    velocity = (momentum * velocity) - (learning_rate * gradient)
  '''
  def momentum_updates(self, params, gradients, learning_rate, momentum=0.5):
    assert T.lt(momentum, 1.0) and T.ge(momentum, 0)
    updates = []
    for p,g in zip(params,gradients):
      pvelocity = theano.shared(p.get_value()*0., broadcastable=p.broadcastable)
      updates.append((p, p + pvelocity))
      updates.append((pvelocity, momentum*pvelocity - learning_rate*g))
    return updates

  '''
  Nesterov weight updates; initial velocity = 0
    weight = weight + (momentum*velocity) + ((1-momentum) * velocity))
    velocity = (momentum * velocity) - (learning_rate * gradient)
  '''
  def nesterov_updates(self, params, gradients, learning_rate, momentum=0.5):
    assert T.lt(momentum, 1.0) and T.ge(momentum, 0)
    updates = []
    for p,g in zip(params,gradients):
      pvelocity = theano.shared(p.get_value()*0.)
      pprev = pvelocity
      updates.append((pvelocity, momentum*pvelocity - learning_rate*g))
      updates.append((p, p + (momentum*pprev) + ((1-momentum) * pvelocity)))
    return updates

  '''
  Adagrad is a per-parameter adaptive optimisation method. It works by keeping
  a record of the squared(gradients) at each update, and uses this record
  to perform relative weight modifications.

    cached_p = cached_p + gradient**2
    weight^new = weight - learning_rate*gradient / sqrt(cached_p+epsilon)
  '''
  def adagrad_updates(self, params, gradients, learning_rate):
    updates = []
    cached_p = []

    for p in params:
      eps_p = numpy.zeros_like(p.get_value(borrow=True), dtype=theano.config.floatX)
      cached_p.append(theano.shared(eps_p, borrow=True))

    for p,g,c in zip(params, gradients, cached_p):
      c_value = c + T.sqr(g)
      updates.append((p, p - learning_rate * g / T.sqrt(c_value+1e-8)))
      updates.append((c, c_value))
    return updates

  '''
  Serialise the model parameters to disk.
  '''
  def save(self, folder):
      for param, name in zip(self.params, self.names):
          numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

