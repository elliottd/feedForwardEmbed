import argparse
import cPickle
import numpy

import theano
import theano.tensor as T

class NearestEmbedder:

  def __init__(self, args):
    self.names = ['embeddings', 'hidden-embeddings', 'output-hidden', 
                  'bias-hidden', 'bias-output']

    ''' The word embeddings is a matrix of dim:
          |vocabulary| x |embedding layer| '''
    emb = self.initWeights(args.checkpoint, self.names[0])

    ''' The embedding -> hidden layer weights is a matrix of dim:
         |embedding layer| * |input| x |hidden layer| '''
    whe = self.initWeights(args.checkpoint, self.names[1])

    ''' The hidden -> output layer weights is a matrix of dim:
          |hiden layer| x |vocabulary| '''
    woh = self.initWeights(args.checkpoint, self.names[2])

    ''' Bias for the hidden layer '''
    bh = self.initBiases(args.checkpoint, self.names[3])

    ''' Bias for the output layer '''
    bo = self.initBiases(args.checkpoint, self.names[4])

    self.params = [emb, whe, woh, bh, bo]

    ''' Theano types for the input, output, and hyper-parameters '''

    # Alternative input type used for word-word similarities
    word = T.iscalar()
    single_x = emb[word]

    dist = emb - single_x                   # calculate the squared distance
    ndist = T.sum(T.sqrt(dist **2), axis=1) # between word representations

    ''' Given one input word, return it's closest words in the embedding space'''
    self.distance = theano.function(inputs=[word], outputs=ndist)

  '''
  Initialise the weights from scratch of unpickle them from disk
  '''
  def initWeights(self, checkpoint=None, name=None):
    return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

  '''
  Initialise the bias from scratch of unpickle them from disk
  '''
  def initBiases(self, checkpoint=None, name=None):
    return theano.shared(numpy.load("%s/%s.npy" % (checkpoint, name)).astype(theano.config.floatX))

class NearestWords:

  def __init__(self, args):
    self.n = args.n
    self.vocab = cPickle.load(open("%s/dictionary.pk" % args.checkpoint, "rb"))
    args.vocab_size = len(self.vocab)
    self.model = NearestEmbedder(args)

  def run(self):
    idx2w  = dict((v,k) for k,v in self.vocab.iteritems())

    print "Model vocabulary"
    print(self.fmtcols(self.vocab.keys(), 6))

    while True:
      text = raw_input("--->")

      if text == "vocab!":
        print "Model vocabulary"
        print(self.fmtcols(self.vocab.keys(), 6))
        continue

      testseq = self.words2indices(text)

      if testseq != None:
        nearest = self.model.distance(testseq)
        snearest = sorted(zip(nearest, range(len(self.vocab))))
        print "%d nearest words to %s:\n" % (self.n, text)
        for i in range(1,self.n+1): # Avoid self and +1
          print idx2w[snearest[i][1]], snearest[i][0]
        print

  '''
  Convert a word token into a dictionary index
  '''
  def words2indices(self, string):
    split = string.split()
    if len(split) != 1:
      return None
    testseq = []
    for token in split:
      try:
        testseq.append(self.vocab[token])
      except KeyError:
        print "%s is not in the model vocabulary"
        return None
    return testseq[0]

  '''
  Pretty print the dictionary.

  http://stackoverflow.com/questions/1524126/how-to-print-a-list-more-nicely/
  '''
  def fmtcols(self, mylist, cols):
    lines = ("\t".join(mylist[i:i+cols]) for i in xrange(0,len(mylist),cols))
    return '\n'.join(lines)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predict the N-most simlar words to a given word")

  parser.add_argument("--checkpoint", default=None, type=str, help="Path to a pickled model", required=True)
  parser.add_argument("--n", type=int, default=10, help="N nearest words. Default=10")
  
  n = NearestWords(parser.parse_args())
  n.run()
