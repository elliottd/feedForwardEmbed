import argparse
import sys
import math
import time
import os
import shutil
from time import gmtime, strftime, time
from collections import OrderedDict
import signal

import numpy
from numpy import *
from numpy.random import randn
from numpy import zeros
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from loadData import LoadData
from WordEmbedder import WordEmbedder

class Runner:

  def __init__(self, args):
    self.args = args
    signal.signal(signal.SIGINT, self.sigint_handler)

  def run(self):
    trainX, trainY, validX, validY, testX, testY, vocab = self.prepareAndReadData()
    self.vocab_size = len(vocab)

    ''' Display the arguments for this run '''
    print
    for arg, val in OrderedDict(self.args.__dict__).iteritems():
      print("%s: %s" % (arg, str(val)))
    print

    ''' It is recommended to increase the capacity of a dropout network by
        n/p. See Appendix A of Srivastava et al. (2014) '''
    if self.args.droph != 0. or self.args.dropin != 0.:
      self.args.embed_size = int(math.floor(self.args.embed_size / (1-self.args.dropin)))
      self.args.hidden_size = int(math.floor(self.args.hidden_size / (1-self.args.droph)))
      print "Resized the number of units in the hidden layers"
      print "embed_size: %d" % self.args.embed_size
      print "hidden_size: %d" % self.args.hidden_size

    network = WordEmbedder(self.args, self.vocab_size)

    self.runtime = []
    self.runtime.append(time())
    self.best_e = -1
    self.best_vloss = numpy.inf
    self.best_vpplx = numpy.inf
    self.best_dir = None

    idx2w  = dict((v,k) for k,v in vocab.iteritems())

    print("Training word-embedding model for %d epochs" % self.args.epochs)
    for i in range(self.args.epochs):
      numpy.random.shuffle([trainX, trainY]) # reduce effects of data order
      tic = time()

      # Process the training data in minibatches
      trainloss = []
      trainpplx = []
      for start, end in zip(range(0, len(trainX)+1, self.args.batch_size), 
                            range(self.args.batch_size, len(trainX)+1, self.args.batch_size)):
        tictic = time() # we'll count how long this minibatch took
        x = trainX[start:end]
        y = numpy.eye(self.vocab_size+1)[trainY[start:end].T][0] # TODO: why [0]?
        loss, pplx = network.train(x, y, self.args.learning_rate,
                                         self.args.momentum,
                                         self.args.dropin,
                                         self.args.droph,
                                         self.args.l1reg,
                                         self.args.l2reg)
        trainloss.append(loss)
        trainpplx.append(pplx)

        print '[train] epoch %i >> batch %d/%d took %.2f (s) | batch ce: %.4f'\
              ' pplx %.4f | smoothed ce: %.4f pplx %.4f <\r' % (i, 
              start/self.args.batch_size, len(trainX)/self.args.batch_size, 
              time()-tictic, loss, pplx, numpy.mean(trainloss),
              numpy.mean(trainpplx)),

        sys.stdout.flush()

      loss = numpy.mean(trainloss)
      pplx = numpy.mean(trainpplx)
      self.runtime.append(time())

      # Process the validation data in minibatches
      valloss = []
      valpplx = []
      for start, end in zip(range(0, len(validX)+1, self.args.batch_size), 
                            range(self.args.batch_size, len(validX)+1, self.args.batch_size)):
        vx = validX[start:end]
        vy = numpy.eye(self.vocab_size+1)[validY[start:end].T][0] # TODO: why [0] ?
        # no dropout or regularisation here
        vloss, vpplx, vy_x = network.validate(vx, vy, 0., 0., 0., 0.) 
        valloss.append(vloss)
        valpplx.append(vpplx)

      vloss = numpy.mean(valloss)
      vpplx = numpy.mean(valpplx)

      # Save model parameters and arguments to disk 
      # if it improved validation perplexity

      if vpplx < self.best_vpplx:
        self.best_vpplx = vpplx
        self.best_e = i
        savetime = strftime("%d%m%Y-%H%M%S", gmtime())       
        try:
          os.mkdir("checkpoints")
        except OSError:
          pass # directory already exists
        os.mkdir("checkpoints/epoch%d_%.4f_%s/" % (self.best_e, self.best_vpplx, savetime))
        network.save("checkpoints/epoch%d_%.4f_%s/" % (self.best_e, self.best_vpplx, savetime))
        self.bestdir = "checkpoints/epoch%d_%.4f_%s/" % (self.best_e, self.best_vpplx, savetime)
        self.saveArguments(self.bestdir)

      print "epoch %d [train] took %.2f (s) avg. loss: %.4f avg. pplx: %.4f"\
            " [val] took %.2f (s) avg. loss: %.4f avg. pplx: %.4f %s" % (i, 
            self.runtime[-1]-self.runtime[-2], loss, pplx, time()-self.runtime[-1], vloss, 
            vpplx, "(saved)" if self.best_e == i else "")

      ''' Decay the learning rate decay if there is no improvement 
          in the model val pplx for 10 epochs '''
      if self.args.decay and abs(self.best_e-i) >= 10: 
        self.args.learning_rate *= 0.5 

        print "Decaying learning rate to %f" % self.args.learning_rate

        # Reload the network from the previous self.best position to
        # (possibly) speed up learning
        print "Reverting to the parameters are checkpoint %s" % self.bestdir
        self.args.initialisation_checkpoint = self.bestdir
        network = WordEmbedder(self.args)

      if self.args.learning_rate < 1e-5: 
        break

  def prepareAndReadData(self):
    '''
    Load the data from the onefile into memory, with train/val/test chunks
    '''
    loader = LoadData(self.args)
    return loader.run()

  def printSummary(self):
    print
    print "Trained in %.2f (s). Best epoch %d [val] smoothed pplx: %.4f"\
          % (self.runtime[-1]-self.runtime[0], self.best_e, self.best_vpplx)
          
  def saveArguments(self, directory):
    '''
    Save the command-line arguments, along with the method defaults,
    used to parameterise this run.
    '''    
    handle = open("%s/argparse.args" % directory, "w")
    for arg, val in self.args.__dict__.iteritems():
      handle.write("%s: %s\n" % (arg, str(val)))
    handle.close()
    shutil.copyfile("dictionary.pk", "%s/dictionary.pk" %  directory) # copy the dictionary

  def sigint_handler(self, signum, frame):
    '''
    Custom Ctrl+C handler that ensures the user sees the state of the best saved model.
    '''
    print
    print "Training halted by Ctrl+C"
    self.printSummary()
    sys.exit(0)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Train a simple tri-gram word embeddings model")

  parser.add_argument("--epochs", default=10, type=int)
  parser.add_argument("--batch_size", default=100, type=int)

  parser.add_argument("--ngram", default=3, type=int, help="Number of words in the input sequence")
  parser.add_argument("--embed_size", default=50, type=int)
  parser.add_argument("--hidden_size", default=200, type=int)

  parser.add_argument("--optimiser", default="momentum", type=str, help="Optimiser: sgd, momentum, nesterov, adagrad")
  parser.add_argument("--learning_rate", default=0.1, type=float)
  parser.add_argument("--momentum", default=0.5, type=float)
  parser.add_argument("--decay", default=True, type=bool, help="Decay learning rate if no improvement in loss?")

  parser.add_argument("--l1reg", default=0., type=float, help="L1 cost penalty. Default=0. (off)")
  parser.add_argument("--l2reg", default=0.0001, type=float, help="L2 cost penalty. Default=0.0001")
  parser.add_argument("--dropin", default=0., type=float, help="Prob. of dropping embedding units. Default=0.")
  parser.add_argument("--droph", default=0., type=float, help="Prob. of dropping hidden units. Default=0.")

  parser.add_argument("--initialisation_checkpoint", default=None, type=str, help="Path to a pickled model")
  
  parser.add_argument("--oneFile", type=str, help="Path to a single input text file. Will be split into train/val/test/", default="raw_sentences.txt")
  parser.add_argument("--trainFile", type=str, help="Path to the training data text file")
  parser.add_argument("--valFile", type=str, help="Path to the validation data text file")
  parser.add_argument("--testFile", type=str, help="Path to the test data raw text file.")
  parser.add_argument("--nlen", type=int, help="n-gram lengths to extract from text. Default=4", default=4)
  parser.add_argument("--unk", type=int, help="unknown character cut-off", default=0)

  r = Runner(parser.parse_args())
  r.run()
