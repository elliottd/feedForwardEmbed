import argparse
import numpy
import pprint
import cPickle

import theano
import theano.tensor as T

import WordEmbedder

theano.config.optimizer = "fast_compile"
theano.config.exception_verbosity = 'high'

class WordGenerationEmbedder:

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

    ''' Train model, loss, and pplx (includes dropout parameters in the model'''
    h, py_x = self.model(x, whe, woh, bh, bo)

    y_x = T.argmax(py_x, axis=1) # sample the max prob word from test model

    '''Predict is used to retrieve the softmax vector over all predicted
       words, given an input sequence.'''
    self.predict = theano.function(inputs=[word_indices], 
                                   outputs=[py_x], 
                                   allow_input_downcast=True, on_unused_input='ignore')
  def initWeights(self, xdim, ydim, checkpoint=None, name=None):
    '''
   Initialise the weights from scratch of unpickle them from disk
     '''
    if checkpoint == None:
      return theano.shared(0.01 * randn(xdim, ydim).astype(theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

  def initBiases(self, dims, checkpoint=None, name=None):
    '''
    Initialise the bias from scratch of unpickle them from disk
    '''
    if checkpoint == None:
      return theano.shared(zeros(dims, dtype=theano.config.floatX))
    else:
      return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

  def model(self, e, whe, woh, bh, bo):
    '''
    The model is a simple multi-layer perceptron.
    h is the hidden layer, which has sigmoid activations from the embeddings
    py_x is the output layer, which has softmax activations from the hidden
    '''
    h = T.tanh(T.dot(e, whe) + bh)
    py_x = T.nnet.softmax(T.dot(h, woh) + bo)
    return h, py_x

class WordGenerator:
 
  def __init__(self, args):
    self.args = args
    self.vocab = cPickle.load(open("%s/dictionary.pk" % (self.args.checkpoint), "rb"))
    args.vocab_size = len(self.vocab)
    self.model = WordGenerationEmbedder(args)

  def run(self):
    idx2w  = dict((v,k) for k,v in self.vocab.iteritems())
    print "vocab! will show you the model vocabulary\n"

    while True:
      text = raw_input("--->")
      if text == "vocab!":
        print "Model vocabulary"
        print(self.fmtcols(self.vocab.keys(), 6))
        continue

      testseq = self.words2indices(text)
      print "%s:" % text

      if testseq != None:
        predictions = self.model.predict([testseq]) # forward-prop through the model
        spred = sorted(zip(predictions[0][0],range(len(self.vocab))), reverse=True)
        for i in range(10):
          print idx2w[spred[i][1]], spred[i][0]
        print

  def words2indices(self, string):
    '''
    Converts a sequence of word tokens into a sequence of dictionary key
    identifiers. The WordEmbedding model operates over dictionary keys and
    so this transition needs to happen.
    '''
    string = string.lower()
    split = string.split()
    if len(split) != self.args.ngram:
      print "This model accepts inputs of exactly %d tokens" % self.args.ngram
      return None
    testseq = []
    for token in split:
      try:
        testseq.append(self.vocab[token])
      except KeyError:
        print "%s is not in the model vocabulary" % token
        return None
    return numpy.array(testseq)

  def fmtcols(self, mylist, cols):
    '''
    http://stackoverflow.com/questions/1524126/how-to-print-a-list-more-nicely/1524333#1524333
    '''
    lines = ("\t".join(mylist[i:i+cols]) for i in xrange(0,len(mylist),cols))
    return '\n'.join(lines)
    
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predict the next word, given a sequence of words")

  parser.add_argument("--checkpoint", default=None, type=str, help="Path to a pickled model")
  parsed = parser.parse_args()
  checkpoint_dir = parsed.checkpoint

  parser.add_argument("--embed_size", default=30, type=int)
  parser.add_argument("--hidden_size", default=100, type=int)
  parser.add_argument("--nlen", type=int, help="n-gram lengths to extract from text. Default=4", default=4)
  parser.add_argument("--unk", type=int, help="unknown character cut-off", default=5)
  parser.add_argument("--ngram", type=int, help="n-gram lengths to extract from text. Default=3", default=3)

  def convert_arg_line_to_args(arg_line):
    arg_line = arg_line.replace("\n","")
    split = arg_line.split(":")
    return "--%s %s " % (split[0], split[1].lstrip())

  argline = ""
  with open("%s/argparse.args" % checkpoint_dir) as f:
    for line in f:
      argline += convert_arg_line_to_args(line)

  parsed_args, unknown = parser.parse_known_args(argline.split())
  parsed_args.checkpoint = checkpoint_dir

  r = WordGenerator(parsed_args)
  r.run()
