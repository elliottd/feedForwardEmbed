import argparse
import sys
import math
import time
import os
from time import gmtime, strftime

import numpy
from numpy import *
from numpy.random import randn
from numpy import zeros
import theano
import theano.tensor as T
import scipy
import scipy.io
import matplotlib.pyplot as plt

from loadData import LoadData

class WordEmbedder:

  def __init__(self, args):
    self.args = args
    self.names = ['embeddings', 'hidden-embeddings', 'output-hidden', 
                  'bias-hidden', 'bias-output']

    ''' The word embeddings is a matrix of dim:
          |vocabulary| x |embedding layer| '''
    emb = self.initWeights(args.vocab_size+1, 
                           args.embed_size, 
                           args.checkpoint, 
                           self.names[0])

    ''' The embedding -> hidden layer weights is a matrix of dim:
         |embedding layer| * |input| x |hidden layer| '''
    whe = self.initWeights(args.embed_size * args.ngram, 
                           args.hidden_size, 
                           args.checkpoint, 
                           self.names[1])

    ''' The hidden -> output layer weights is a matrix of dim:
          |hiden layer| x |vocabulary| '''
    woh = self.initWeights(args.hidden_size, 
                           args.vocab_size+1, 
                           args.checkpoint, 
                           self.names[2])

    ''' Bias for the hidden layer '''
    bh = self.initBiases(args.hidden_size, 
                         args.checkpoint, 
                         self.names[3])

    ''' Bias for the output layer '''
    bo = self.initBiases(args.vocab_size+1, 
                         args.checkpoint, 
                         self.names[4])

    self.params = [emb, whe, woh, bh, bo]

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

    ''' The model and its cost function '''

    h, py_x = self.model(x, whe, woh, bh, bo) # forward prop from embeddings 
                                              # to the hidden layer and 
                                              # word prediction layer

    cost = -T.mean(T.log(py_x)[T.arange(x.shape[0]), y]) # -ve log-likelihood
    gradients = T.grad(cost, self.params)
    y_x = T.argmax(py_x, axis=1) # sample the max prob word

    dist = emb - single_x                   # calculate the squared distance
    ndist = T.sum(T.sqrt(dist **2), axis=1) # between word representations

    ''' Train the parameters of the model, given a sequence of words, a target
        word, and a learning rate. Selects the optimizer based on the user
        requirements.'''
    self.train = theano.function(inputs=[word_indices, y, lr, mom],
                                 outputs=cost, 
                                 updates=self.optimizer(self.params, gradients, lr, mom),
                                 allow_input_downcast=True)

    '''Validation is where we can test whether reducing the training loss
       is helping to reduce our loss on "unseen" data'''
    self.validate = theano.function(inputs=[word_indices, y], 
                                    outputs=[cost,y_x], 
                                    allow_input_downcast=True)

    '''Predict is used to retrieve the softmax vector over all predicted
       words, given an input sequence.'''
    self.predict = theano.function(inputs=[word_indices], 
                                   outputs=py_x, 
                                   allow_input_downcast=True)

    ''' Normalise the weights in the embedding layer to be within the unit
        sphere. TODO: understand why?'''
    self.normalize = theano.function(inputs=[],
                                     updates={emb:emb / T.sqrt((emb**2).sum(axis=1)).dimshuffle(0, 'x')})

    ''' Given one input word, return it's closest words in the embedding space'''
    self.distance = theano.function(inputs=[word], outputs=ndist)

  '''
  Initialise the weights from scratch of unpickle them from disk
  '''
  def initWeights(self, xdim, ydim, checkpoint=None, name=None):
    if checkpoint == None:
      return theano.shared(0.01 * randn(xdim, ydim).astype(theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

  '''
  Initialise the bias from scratch of unpickle them from disk
  '''
  def initBiases(self, dims, checkpoint=None, name=None):
    if checkpoint == None:
      return theano.shared(zeros(dims, dtype=theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

  '''
  The model is a simple multi-layer perceptron.
  h is the hidden layer, which has sigmoid activations from the embeddings
  py_x is the output layer, which has softmax activations from the hidden
  '''
  def model(self, e, whe, woh, bh, bo):
    h = T.nnet.sigmoid(T.dot(e, whe) + bh)
    py_x = T.nnet.softmax(T.dot(h, woh) + bo)
    return h, py_x

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
  Serialise the model parameters to disk.
  '''
  def save(self, folder):
      for param, name in zip(self.params, self.names):
          numpy.save(os.path.join(folder, name + '.npy'), param.get_value())

class Runner:

  def __init__(self, args):
    self.args = args

  def run(self):
    trainX, trainY, validX, validY, testX, testY, vocab = self.myLoadData()
    self.args.vocab_size = len(vocab)

    network = WordEmbedder(self.args)

    trainloss = []
    best_e = -1
    best_vl = numpy.inf

    idx2w  = dict((v,k) for k,v in vocab.iteritems())

    print("Training word-embedding model for %d epochs" % self.args.epochs)
    for i in range(self.args.epochs):
      tic = time.time()
      numpy.random.shuffle([trainX, trainY])

      '''
      Process the training data in minibatches
      '''
      for start, end in zip(range(0, len(trainX)+1, self.args.batch_size), 
                            range(self.args.batch_size, len(trainX)+1, self.args.batch_size)):
        x = trainX[start:end]
        y = trainY[start:end].T
        trainloss.append(numpy.mean(network.train(x, y, 
                                                  self.args.learning_rate,
                                                  self.args.momentum)))

        print '[train] epoch %i > %.2f%%,' % (i, (start+1)*100./len(trainX)),\
              'completed in %.2f (sec) <<\r' %(time.time()-tic),

        sys.stdout.flush()

      vx = validX
      vy = validY.T
      valloss, y_x = network.validate(vx,vy)

      # Save model parameters to disk if it improved validation log-likelihood
      if valloss < best_vl:
        best_vl = valloss
        best_e = i
        savetime = strftime("%d%m%Y-%H%M%S", gmtime())       
        try:
          os.mkdir("checkpoints")
        except OSError:
          pass # directory already exists
        os.mkdir("checkpoints/epoch%d_%.4f_%s/" % (best_e, best_vl, savetime))
        network.save("checkpoints/epoch%d_%.4f_%s/" % (best_e, best_vl, savetime))

      print "epoch %d took %.2f (s) [train] log-likelihood: %.4f [val] "\
            "log-likelihood: %.4f %s" % (i, time.time()-tic, 
            numpy.mean(trainloss), numpy.mean(valloss), 
            "(saved)" if best_e == i else "")

      ''' Decay the learning rate decay AND increase momentum if
          there is not improvement in the model for 10 epochs '''
      if self.args.decay and abs(best_e-i) >= 10: 
        self.args.learning_rate *= 0.5 
        self.args.momentum *= 1.2

        # sanity check the momentum value
        self.args.momentum = 0.99 \
          if self.args.momentum >= 1.0 \
          else self.args.momentum

        print "Decaying learning rate to %f, increasing momentum to %f" \
        % (self.args.learning_rate, self.args.momentum)

      if self.args.learning_rate < 1e-5: 
        break

    print
    print "Training complete! Best epoch %d [val] log-likelihood: %.4f"\
          % (best_e, best_vl)

  '''
  Load the data from the inputfile into memory, with train/val/test chunks
  '''
  def myLoadData(self):
    loader = LoadData(self.args)
    trainx, trainy, valx, valy, testx, testy, vocab = loader.run()
    return trainx, trainy, valx, valy, testx, testy, vocab      

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a simple tri-gram word embeddings model")

  parser.add_argument("--epochs", default=10, type=int)
  parser.add_argument("--batch_size", default=100, type=int)

  parser.add_argument("--ngram", default=3, type=int, help="Number of words in the input sequence")
  parser.add_argument("--embed_size", default=50, type=int)
  parser.add_argument("--hidden_size", default=200, type=int)

  parser.add_argument("--optimizer", default="momentum", type=str, help="Optimizer: sgd, momentum, nesterov")
  parser.add_argument("--learning_rate", default=0.1, type=float)
  parser.add_argument("--momentum", default=0.5, type=float)
  parser.add_argument("--decay", default=True, type=bool)

  parser.add_argument("--checkpoint", default=None, type=str, help="Path to a pickled model")
  
  parser.add_argument("--inputfile", type=str, help="Path to input text file")
  parser.add_argument("--nlen", type=int, help="n-gram lengths to extract from text. Default=4", default=4)
  parser.add_argument("--unk", type=int, help="unknown character cut-off", default=5)

  r = Runner(parser.parse_args())
  r.run()
