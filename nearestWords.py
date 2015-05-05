import argparse
import numpy
from feedForwardEmbedding import WordEmbedder
from loadData import LoadData

class NearestWords:

  def __init__(self, args):
    y,z,a,b,c,d,self.vocab = LoadData(args).run()
    args.vocab_size = len(self.vocab)
    self.model = WordEmbedder(args)

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
      if testseq != None:
        nearest = self.model.distance(testseq)
        snearest = sorted(zip(nearest, range(len(self.vocab))))
        print "Ten nearest words to %s:" % text
        for i in range(1,10):
          print idx2w[snearest[i][1]], snearest[i][0]
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
  parser.add_argument("--ngram", default=3, type=int)
  parser.add_argument("--optimizer", default="momentum")
  
  n = NearestWords(parser.parse_args())
  n.run()
