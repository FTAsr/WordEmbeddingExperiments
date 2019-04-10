from gensim.models import Word2Vec
from gensim import utils, matutils
import numpy as np
import math




def vLength(vector):
    length = math.sqrt(sum([n**2 for n in vector]))
    return length

def vSimilarity(v1, v2):
    vSimilarity = np.dot(matutils.unitvec(v1), matutils.unitvec(v2))
    return vSimilarity
    

def hypoTesting(v1, v2, vt):
    v = v1
    vv = v1 - v2
    directSim = vSimilarity(v, vt)
    deductiveSim = vSimilarity(vv, vt)
    if(directSim > deductiveSim):
        print "Hypo1: direct similarity wins: " + str(directSim) + " vs. " + str(deductiveSim)
    else:
        print "Hypo2: deductive similarity wins: " + str(deductiveSim) + " vs. " + str(directSim)


def similarityRank(m, targetWord, positive=[], negative=[], restrict_vocab=None):
    """
    Find the rank of targetWord in the vocabulary wrt. its similarity to the vector obtained from positve and negative lists
    """
    m.init_sims()
    positive = [
        (word, 1.0) for word in positive
    ]
    negative = [
        (word, -1.0) for word in negative
    ]
    all_words, mean = set(), []
    for word, weight in positive + negative:
        if word in m.vocab:
            mean.append(weight * m.syn0norm[m.vocab[word].index])
            all_words.add(m.vocab[word].index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    mean = matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
    limited = m.syn0norm if restrict_vocab is None else m.syn0norm[:restrict_vocab]
    dists = np.dot(limited, mean)
    best = matutils.argsort(dists, reverse=True)
    result = [(m.index2word[sim], float(dists[sim])) for sim in best]
    #print result[0]
    return [x for x, y in enumerate(result) if y[0] == targetWord]

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
    