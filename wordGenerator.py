import argparse
import numpy
import pprint

from feedForwardEmbedding import WordEmbedder
from loadData import LoadData

class WordGenerator:

  def __init__(self, args):
    y,z,a,b,c,d,self.vocab = LoadData(args).run()
    args.vocab_size = len(self.vocab)
    self.model = WordEmbedder(args)

  def words2indices(self, string):
    string = string.lower()
    split = string.split()
    if len(split) < 3:
      return None
    testseq = []
    for token in split:
      try:
        testseq.append(self.vocab[token])
      except KeyError:
        print "%s is not in the model vocabulary" % token
        return None
    return numpy.array(testseq)

  '''
  http://stackoverflow.com/questions/1524126/how-to-print-a-list-more-nicely/1524333#1524333
  '''
  def fmtcols(self, mylist, cols):
    lines = ("\t".join(mylist[i:i+cols]) for i in xrange(0,len(mylist),cols))
    return '\n'.join(lines)
    
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
      print "%s:" % text
      if testseq != None:
        predictions = self.model.predict([testseq])
        spred = sorted(zip(predictions[0],range(len(self.vocab))), reverse=True)
        for i in range(10):
          print idx2w[spred[i][1]], spred[i][0]
        print

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Predict the next word, given a sequence of three words")

  parser.add_argument("--checkpoint", default=None, type=str, help="Path to a pickled model")
  parser.add_argument("--embed_size", default=50, type=int)
  parser.add_argument("--hidden_size", default=200, type=int)
  parser.add_argument("--learning_rate", default=0.1, type=float)
  parser.add_argument("--momentum", default=0.5, type=float)
  parser.add_argument("--inputfile", type=str, help="Path to input text file")
  parser.add_argument("--nlen", type=int, help="n-gram lengths to extract from text. Default=4", default=4)
  parser.add_argument("--unk", type=int, help="unknown character cut-off", default=5)
  parser.add_argument("--ngram", type=int, help="n-gram lengths to extract from text. Default=4", default=3)
  parser.add_argument("--optimizer", default="momentum")
  
  r = WordGenerator(parser.parse_args())
  r.run()
