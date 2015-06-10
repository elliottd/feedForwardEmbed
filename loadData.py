import os
import argparse
import math
import cPickle

import numpy

class LoadData:

  def __init__(self, args):
    self.args = args
    self.vocab = dict() # maps tokens -> indices
    self.vocab["UNK"] = 0 # placeholder for thresholded tokens
    self.unkdict = dict() # counts occurrence of tokens
    self.unkdict["UNK"] = -1 # placeholder for thresholded tokens
    self.counter = 0 # dictionary counter

  def run(self):
    '''
    Create and return model input data from a line-delineated text file.
  
    1. Iterates through the input file and creates the vocabulary mapping
       and collections counts, which are later used for UNK thresholding.
  
    2. Iterates through the input file again and creates the x and y
       sequences, replacing UNKed tokens where necessary.
  
    3. Splits the x and y lists into train/val/test splits.
  
    Returns train{x,y}, val{x,y}, test{x,y}, and vocab
  
    TODO: combine the UNK counting and the input, target lists?
    '''
    self.constructVocabulary()

    if self.args.trainFile != None:
      trainx, trainy = self.constructNgrams(self.args.trainFile)
      valx, valy = self.constructNgrams(self.args.valFile)
      testx, testy = self.constructNgrams(self.args.testFile)
    else:
      xlist, ylist = self.constructNgrams(self.args.oneFile)
      trainx, trainy, valx, valy, testx, testy = self.splitData(xlist, ylist) 

    return trainx, trainy, valx, valy, testx, testy, self.vocab

  def constructVocabulary(self):
    '''
    Iterate once through the input texts and count up the
    occurrences of tokens.

    Behaviour depends on whether you supplied oneFile or
    a train/val/testFile on the command line.

    TODO: should the vocabulary include the val and test tokens?
    '''

    if self.args.trainFile == None:
      self.extendVocabulary(self.args.oneFile)
    else:
      self.extendVocabulary(self.args.trainFile)
      self.extendVocabulary(self.args.valFile)
      self.extendVocabulary(self.args.testFile)

    # Useful message about how much of the input vocabulary was truncated
    unkCounter = 0
    for x in self.unkdict:
      if self.unkdict[x] < self.args.unk:
        unkCounter += 1

    print "%s words occurred fewer then UNK %d times" % (unkCounter, self.args.unk)

    print "Pickling dictionary to dictionary.pk"
    cPickle.dump(self.vocab, open("dictionary.pk", "wb"))

  def extendVocabulary(self, textfile):
  
    with open(textfile) as f:
      for line in f:
        line = line.lstrip()
        line = line.rstrip()
        line = line.lower()
        splitline = line.split(" ") # split on a space
        for token in splitline:
          if token not in self.vocab: # add token to the vocab dictionary
            self.vocab[token] = self.counter
            self.unkdict[token] = 1
            self.counter += 1
          else:
           self.unkdict[token] += 1

  def constructNgrams(self, filename=None):
    '''
    Construct a set of input sequences and target words.
  
    An input sequence is a list of n unique indices from self.vocab,
    mapped from the actual underlying sequence of words. 
  
    A target is a list with a single unique index from self.vocab. 
  
    Each list is converted into a numpy.array([list]) to make it
    simple to work with sequence transposes in the word-embedding model.
  
    For example:
      x = [], y = []
      x[0] = ['the', 'man', 'said'] = numpy.array([12, 55, 210])
      y[0] = ['hello'] = numpy.array([94])
      x = numpy.array(x), y = numpy.array(y)

    TODO: Probably falls over with very large text files. Try yield?
    '''
    xs = []
    ys = []

    with open(filename) as f:
      for line in f:
        # Pre-process to remove whitesplit, lowercase, and split by spaces
        line = line.lstrip()
        line = line.rstrip()
        line = line.lower()
        splitline = line.split()
        fixedline = []
        for t in splitline:
          if self.unkdict[t] < self.args.unk:
            fixedline.append(self.vocab["UNK"])
          else:
            fixedline.append(self.vocab[t])

        # iterate through nlen-sized chunks of the input
        for j in range(0, len(fixedline)-self.args.nlen+1):
          # n/m-characters to training sequence
          xs.append(numpy.array(fixedline[j:j+self.args.nlen-1]))
          # m-1th character for the target
          ys.append(numpy.array([fixedline[j+self.args.nlen-1]]))

    return numpy.array(xs), numpy.array(ys)
          
  def splitData(self, xseq, yseq):
    '''
    Splits xseq and yseq into training, validation, and test data.
    '''
    t = 0.8
    v = 0.1
    
    trainsize = int(math.floor(t*len(xseq))) # determine the number of
    valsize = int(math.floor(v*len(xseq)))   # items in each split

    trainx = xseq[0:trainsize]
    trainy = yseq[0:trainsize]

    valx = xseq[trainsize:trainsize+valsize]
    valy = yseq[trainsize:trainsize+valsize]

    testx = xseq[trainsize+valsize:]
    testy = yseq[trainsize+valsize:]

    return trainx, trainy, valx, valy, testx, testy

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--oneFile", type=str, help="Path to a single input text file. Will be automatically split.", default="raw_sentences.txt")
  parser.add_argument("--trainFile", type=str, help="Path to the training data text file")
  parser.add_argument("--valFile", type=str, help="Path to the validation data text file")
  parser.add_argument("--testFile", type=str, help="Path to the test data raw text file.")
  parser.add_argument("--nlen", type=int, help="n-gram lengths to extract from text. Default=4", default=4)
  parser.add_argument("--unk", type=int, help="unknown character cut-off. Default=5", default=5)

  l = LoadData(parser.parse_args())
  l.run()
