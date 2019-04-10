from gensim.models import Word2Vec
from gensim import utils, matutils
import numpy as np
import math




def filter(model, inputFile, adjacents = 0):
    #filters the vocabulary of a vector space and only keeps that of words occurring in the inputFile
    restrict_vocab = set()
    for line in open(inputFile):
        words = line.split()
        for word in words:
            restrict_vocab.add(word)
    additional_vocab = set()
    if(adjacents > 0):
        for word in restrict_vocab:
            adjacents = model.most_similar(word, topn = adjacents)
            additional_vocal.add(adjacents)
    restrict_vocab = restrict_vocab.union(additional_vocab)
    limited = m.syn0norm if restrict_vocab is None else m.syn0norm[:restrict_vocab]
    
    return limited
    
if __name__ == '__main__':


    m = Word2Vec.load_word2vec_format("/Users/fa/workspace/repos/_codes/trunk/vectors-phrase.bin", binary=True) 
    m.init_sims(replace=True)
    hypoTesting(m["teacher"], m["school"], m["professional"])
    hypoTesting(m["teacher"] + m["nurse"], m["school"] + m["hospital"], m["professional"])
    similarityRank(m, "squirrel_monkey", ["stork"])
    similarityRank(m, "stork", ["baby"])
    similarityRank(m, "baby", ["stork"])
    m.vocab["baby"].__str__()
    m.vocab["stork"].__str__()
    