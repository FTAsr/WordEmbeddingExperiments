#!/Users/fa/anaconda/bin/python
 
from gensim import corpora, models, similarities
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import matplotlib.pyplot as plt
import numpy as np
import bisect 
import math

 
import multiprocessing
import openpyxl 
import os
import sys
import subprocess
import docopt

class MySentences(object):
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for line in open(self.fname):
            yield line.split()

#def prepareWordContext(inputFile, wordContextFile, wordVocabFile, contextVocabFile, splitLine = False):
#    for sentence in MySentences(inputFile):
            
    
    
    
    
def trainFiltered(corpusFile, modelFile, _size, _window, _min_count, _sg, _iter, _negative, _sample): #trains a model on a text corpus and saves it in both binary and word2vec formats
    print("FA: trainFiltered(" + modelFile  + ")")
    #wantedVocab_set = set(line.strip() for line in open("CHILDESvocab.txt"))
    #print(len(wantedVocab_set))
    inp = corpusFile
    outp2 = modelFile 
    model = Word2Vec(size = _size, window = _window, min_count = _min_count, sg = _sg, iter = _iter, negative = _negative, sample = _sample) # an empty model, no training   
    sentences = MySentences(inp) # a memory-friendly iterator
    #discard_stopwords = lambda: ((word for word in sentence if word in wantedVocab_set) for sentence in sentences)
    print("building vocab started!")
    #model.build_vocab(discard_stopwords())
    model.build_vocab(sentences)
    print("building vocab finished!")
    vocab = sorted(model.vocab)
    print(vocab)
    print(len(vocab))
    sentences = MySentences(inp)
    print("training started!")
    ##important: if you filter out unwanted vocabulary prior to feeding it, the actual context window size would be unknown.
    #model.train(discard_stopwords())
    model.train(sentences)
    print("training finished!")
    vocab = sorted(model.vocab)
    print(len(vocab))
    model.init_sims(replace=True)
    model.save_word2vec_format(outp2, binary=False)
    return trainFiltered
    
    
def test(modelFile): #quick test of the model for sample words
    model1 = Word2Vec.load_word2vec_format(modelFile, binary=False)  # C text format
    model1.init_sims(replace=True)
    s1 = model1.most_similar(positive=['man'])
    print 'most similar top10 man'
    print(s1)
    #vocab = sorted(model1.vocab)
    #print(vocab)
    s2 = model1.similarity('woman', 'man')
    print 'similarity score: woman man'
    print(s2)
    s3 = model1["woman"]
    print 'word vector: woman'
    print(s3)
    
    return test
    
def betweenWithinSimilarities_paradigmatic(modelFile, categories, similarityFile):
    ## words from the same semantic category should be more similar than word from different semantic categories
    model= Word2Vec.load_word2vec_format(modelFile, binary=False)  # C text format
    #print("FA: pairwiseSimilarities(" + modelFile + ", " + similarityFile + ")")
    sameCategory = set()
    for category in categories.keys():
        words = categories[category]
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if (word1 in model.vocab) and (word2 in model.vocab):
                    sameCategory.add(word1 + "_" + word2)            
    print(sameCategory)
    outf = open(similarityFile, 'w')
    diffCatScore = 0.0
    sameCatScore = 0.0
    sameCatScoreNo = 0   
    diffCatScoreNo = 0  
    words = model.vocab
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            score = model.similarity(word1, word2)
            outf.write(word1 + " & " + word2 + " = " + str(score))
            #if(word1 == "glass"):# and (word2 in  ["plate", "monster", "smash", "cat", "chase"])):
            #    print("Paradigmatic Similarity for word1" + " & " + word2 + " = " + str(score))
            if word1 == word2:
                continue
            elif (word1 + "_" + word2) in sameCategory:
                sameCatScore = sameCatScore + score
                sameCatScoreNo +=  1
            else:
                diffCatScore = diffCatScore + score
                diffCatScoreNo +=  1
    avgScore = ( sameCatScore + diffCatScore )/(sameCatScoreNo+diffCatScoreNo)
    sameCatScore = sameCatScore * 1.0 / sameCatScoreNo
    diffCatScore = diffCatScore * 1.0 / diffCatScoreNo
    avgScore = ( sameCatScore*sameCatScoreNo + diffCatScore*diffCatScoreNo )/(sameCatScoreNo+diffCatScoreNo)
    #print ("average similarity: " + str( avgScore ))
    #print ("between category similarity: "  +   str(diffCatScore)) 
    #print ("within category similarity: " + str(sameCatScore) )
    #print ("Fisher score:" + str(sameCatScore - diffCatScore) )
    outf.write(
        "average similarity: " + str( avgScore )+"\n" +
        "between category similarity: "  +   str(diffCatScore) +"\n" +
        "within category similarity: " + str(sameCatScore) +"\n" +
        "Fisher score:" + str(sameCatScore - diffCatScore) +"\n" 
    )
    outf.close()
    ''''
    if (sameCatScore <= 0 or (sameCatScore < diffCatScore)):
        return avgScore, diffCatScore, sameCatScore, 0.0
    return avgScore, diffCatScore, sameCatScore, diffCatScore * 1.0 /sameCatScore
    '''
    return avgScore, diffCatScore, sameCatScore, (sameCatScore - diffCatScore) 
def findCategory(word, categories):
    for key, values in categories.iteritems():
        if word in values:
            return key
    print( "Warning:: no category found for the unidentified word " + word )
    return null
    
def betweenWithinSimilarities_syntagmatic(modelFile, categories, frames, similarityFile):
    ## words that [can] appear in each other's direct context (sentences) should be more similar than word appearing in different contexts
    model= Word2Vec.load_word2vec_format(modelFile, binary=False)  # C text format
    #print("FA: pairwiseSimilarities(" + modelFile + ", " + similarityFile + ")")
    sameContext = set()
    for frame in frames:
        cats = frame.split() 
        for i, cat1 in enumerate(cats):
            for j, cat2 in enumerate(cats):
                if ( i != j ): 
                    ## we want to consider coocurrence of the same-category words only if they actually co-occur in a frame, 
                    ## e.g.,  only if a frame like "NOUN-ANIM VERB-TRAN NOUN-ANIM" exists in the grammar, then we consider NOUN-ANIM and NOUN-ANIM as same-context categories.
                    sameContext.add(cat1 + "_" + cat2)            
    #print("SAME CONTEXT CATEGORIES:")
    #print(sameContext)
    outf = open(similarityFile, 'w')
    diffContScore = 0.0
    sameContScore = 0.0
    sameContScoreNo = 0   
    diffContScoreNo = 0  
    words = model.vocab
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            score = model.similarity(word1, word2)
            outf.write(word1 + " & " + word2 + " = " + str(score))
            #if(word1 == "glass" and (word2 in  ["plate", "monster", "smash", "cat", "chase"])):
            #    print("Syntagmatic Similarity for word1" + " & " + word2 + " = " + str(score))
            #print(word1 + " & " + word2 + " = " + str(score))
            cat1 = findCategory(word1, categories)
            cat2 = findCategory(word2, categories)
            if word1 == word2:
                continue
            elif (cat1 + "_" + cat2) in sameContext:
                sameContScore = sameContScore + score
                sameContScoreNo +=  1
            else:
                diffContScore = diffContScore + score
                diffContScoreNo +=  1
                #print("WORDS " + cat1 + ":" + word1 + " " + cat2 + ":" + word2 + " NEVER OCCUR IN THE SAME CONTEXT!")
    avgScore = ( sameContScore + diffContScore )/(sameContScoreNo+diffContScoreNo)
    sameContScore = sameContScore * 1.0 / sameContScoreNo
    diffContScore = diffContScore * 1.0 / diffContScoreNo
    avgScore = ( sameContScore*sameContScoreNo + diffContScore*diffContScoreNo )/(sameContScoreNo+diffContScoreNo)
    #print ("average similarity: " + str( avgScore ))
    #print ("between category similarity: "  +   str(diffContScore)) 
    #print ("within category similarity: " + str(sameContScore) )
    #print ("Fisher score:" + str(sameContScore - diffContScore) )
    outf.write(
        "average similarity: " + str( avgScore )+"\n" +
        "between category similarity: "  +   str(diffContScore) +"\n" +
        "within category similarity: " + str(sameContScore) +"\n" +
        "Fisher score:" + str(sameContScore - diffContScore) +"\n" 
    )
    outf.close()
    return avgScore, diffContScore, sameContScore, (sameContScore - diffContScore)
    
    
def spectrumSimilarities(modelFile, similarityFile):
    outf = open(similarityFile, 'w')
    model= Word2Vec.load_word2vec_format(modelFile, binary=False)  # C text format
    #print("FA: spectrumSimilarities(" + modelFile + ", " + similarityFile + ")")
    minScore = 0.0
    maxScore = 1.0   
    avgScore = 0.0
    varScore = 0  
    indx = 0
    words = model.vocab
    scores = list()
    for i, word1 in enumerate(words):
        for j, word2 in enumerate(words):
            score = model.similarity(word1, word2)
            outf.write(
                word1 + "\t" + word2 + "\t" + str(score) + "\n"
            )
            if word1 == word2:
                continue
            scores.append(score)
    outf.close()
    return np.min(scores), np.max(scores), np.mean(scores), np.var(scores)
    
    
    
    
    
def nounSimilarities():
    return 0
def verbSimilarities():
    return 0

        
        
class languageGenerator:
    def __init__(self): #, categories, frames, socabulary
        print("FA: languageGenerator()")
        self.categories = { 
            ## All words except "break" and "see" have only one category, therefore they should have same embeddings as their couple. 
            ## Based on this categorization, the embedding of "see" should be an average of "chase" and "smell" and the embedding of "break" should come out as an average of "move" and "smash".
            ## If bootstrapping occurs in a model training, we expect "see" would propagate some similarity between "chase" and "smell"; Same applies to "break".
            "NOUN-HUM": ["man", "woman"],
            "NOUN-ANIM": ["cat", "mouse"],
            "NOUN-INANIM": ["book", "rock"],
            "NOUN-AGRESS": ["dragon", "monster"],
            "NOUN-FRAG": ["glass", "plate"],
            "NOUN-FOOD": ["cookie", "sandwich"],
            "VERB-INTRAN": ["think", "sleep"],
            "VERB-TRAN": ["see", "chase"], 
            "VERB-AGPAT": ["move", "break"],
            "VERB-PERCEPT": ["smell", "see"],
            "VERB-DESTROY": ["break","smash"],
            "VERB-EAT": ["eat"] 
        }
        self.frames = {  
            "NOUN-HUM VERB-EAT NOUN-FOOD",
            "NOUN-HUM VERB-PERCEPT NOUN-INANIM",
            "NOUN-HUM VERB-DESTROY NOUN-FRAG",
            "NOUN-HUM VERB-INTRAN",
            "NOUN-HUM VERB-TRAN NOUN-HUM",
            "NOUN-HUM VERB-AGPAT NOUN-INANIM",
            "NOUN-HUM VERB-AGPAT",
            "NOUN-ANIM VERB-EAT NOUN-FOOD",
            "NOUN-ANIM VERB-TRAN NOUN-ANIM",
            "NOUN-ANIM VERB-AGPAT NOUN-INANIM",
            "NOUN-ANIM VERB-AGPAT",
            "NOUN-INANIM VERB-AGPAT",
            "NOUN-AGRESS VERB-DESTROY NOUN-FRAG",
            "NOUN-AGRESS VERB-EAT NOUN-HUM",
            "NOUN-AGRESS VERB-EAT NOUN-ANIM",
            "NOUN-AGRESS VERB-EAT NOUN-FOOD"
        }
        self.socabulary = self.generateSocabulary()
        self.vocabulary = set()
        for i in self.categories:
            self.vocabulary.update(set(self.categories[i])) 
        return 
    def generateSocabulary(self):
        ##make all possible sentences of the language 
        socabulary = set()
        for frame in self.frames:
            socabulary.update(self.fillin(frame, 0))   
        #print(socabulary)
        #print(len(socabulary))
        return socabulary
    def generateCorpus(self, corpusFile,  distribution = "uniform", size = 1000): 
        print("FA: corpusGenerator(" + corpusFile  + ")")
        socabulary = list(self.socabulary)
        ## distribution:  normal, zipf
        ## size: size of corpus in terms of the number of sentences
        cardinality = len(socabulary)
        print(socabulary[0])
        print(socabulary[cardinality-1])
        ## generate random samples until the sample includes at least one of each possible value:
        while(1):
            if distribution == "zipf":
                sample = self.truncatedRandZipf(cardinality,  0.5, size) # the bigger alpha the more peaky
            elif distribution == "uniform":
                sample = np.random.randint(0,  cardinality, size)
            elif distribution == "binomial":
                sample = np.random.binomial(cardinality,  0.5,  size)  
            sampleSet = set(sample)
            oneOfEach = set(range(0, cardinality))
            if oneOfEach.issubset(sampleSet):
                break
        print(len(sampleSet))
        print(sampleSet)
        print(len(oneOfEach))
        print(oneOfEach)
        print(distribution + " sample of size " + str(size) + " in range [0," + str(cardinality) + "):")
        print(str(min(sample))+ " -- " +  str(max(sample)))  
        freq = np.histogram(sample, bins = cardinality)
        plt.plot(range(0,cardinality), np.sort(freq[0]))
        corpus = []
        for index in sample:
            corpus.append(socabulary[index])
        corpus = '\n'.join(corpus)
        f = open(corpusFile,'w')
        f.write(corpus)
        f.close()
        return corpus
    def fillin(self, frame, index): #fill in the frame with words from the current index category
        slots = frame.split()
        if index == len(slots) : #all slots already filled with words
            return set([frame])
        sentences = set()
        for word in self.categories[slots[index]]:
            slots[index] = word
            sentences.update(self.fillin(" ".join(slots), index+1))
        return sentences
    ## Random zipf sampling in the range [0,n)
    ## Taken from http://stackoverflow.com/questions/31027739/python-custom-zipf-number-generator-performing-poorly
    def truncatedRandZipf(self, n, alpha, size): 
        # Calculate Zeta values from 1 to n: 
        '''
        tmp = [1. / (math.pow(float(i), alpha)) for i in range(1, n+1)]
        zeta = reduce(lambda sums, x: sums + [sums[-1] + x], tmp, [0])
        '''
        tmp = np.power( np.arange(1, n+1), -alpha )
        zeta = np.r_[0.0, np.cumsum(tmp)]
        # Store the translation map: 
        distMap = [x / zeta[-1] for x in zeta]
        # Generate an array of uniform 0-1 pseudo-random values: 
        u = np.random.random(size)
        # bisect them with distMap
        v = np.searchsorted(distMap, u)
        samples = [t-1 for t in v]
        return samples
    ## Using numpy zipf, scaling and flooring to obtain a pseudo-zipfian distribution in the range [0,n)
    def scaledRandZipf(self, n, alpha, size): 
        sample = np.random.zipf(2, size)
        sample = (sample/float(max(sample))) * (n - 1) ## scale the zipf sample: normalize and multiply by cardinality
        return (np.floor(sample)).astype(int) ## translate to int 
    def getVocabulary(self):
        return self.vocabulary
    
      
      
    
    

    
def plotting():

    corpusDistribution  = "uniform"
    corpusSize = 10000
    corpusFile =  "ElmanCorpus" + "_distribution" + corpusDistribution + "_size" +  str(corpusSize) + ".txt"
    lg = languageGenerator()
    #lg.generateCorpus(corpusFile, corpusDistribution  ,  corpusSize )
    #plt.show()
    
    ## parameters:
    _size = 5
    _window = 2
    _min_count = 1 
    _sg = 1
    _iter = 1
    _negative = 0
    _sample = 0
    _eig = 0.5
    #_cds = 0.75
    
    
    begin = 2
    end = 10
    beginW2V = 1
    endW2V = 2
    w2vResults_par =np.zeros(shape=(endW2V,end,4))
    w2vResults_syn =np.zeros(shape=(endW2V,end,4))
    w2vResults_spectrum =np.zeros(shape=(endW2V,end,4))
    svdResults_par =np.zeros(shape=(3,end,4))
    svdResults_syn =np.zeros(shape=(3,end,4))
    svdResults_spectrum =np.zeros(shape=(3,end,4))
    for _size in range(begin,end):
        
        for index, _eig in enumerate([0, 0.5, 1]):
            ## SVD
            parSetup = "_size" + str(_size) + "_window" + str(_window) + "_min_count" + str(_min_count) + "_sample" + str(_sample)
            modelFile = "SVD_ELMAN"+ parSetup + ".model" 
            similarityFile = "SVD_ELMAN"+ parSetup + ".similarity"
            cwd = os.path.dirname(os.path.realpath(__file__))
            os.chdir('/Users/fa/workspace/repos/_codes/omerlevy-hyperwords-688addd64ca2')
            subprocess.call(["./corpus2svd.sh",  "--thr" , str(_min_count), "--win" ,  str(_window), "--sub" , str(_sample), "--dim" , str(_size ) , "--eig", str(_eig),#"--w+c", 
                "/Users/fa/workspace/repos/_codes/elman/" + corpusFile,  "/Users/fa/workspace/repos/_codes/elman/svd" ])
            os.chdir(cwd)
    
            with file('/Users/fa/workspace/repos/_codes/elman/svd/vectors.txt', 'r') as fin: data = fin.read()
            with file(modelFile, 'w') as fout: fout.write(str(len(lg.getVocabulary())) + " " + str(_size ) + "\n" + data)

            #svdResults_par[ index, _size] = list(betweenWithinSimilarities_paradigmatic(modelFile, lg.categories,  similarityFile))
            #svdResults_syn[ index, _size] = list(betweenWithinSimilarities_syntagmatic(modelFile, lg.categories,lg.frames,  similarityFile))
            svdResults_spectrum[ index, _size] = list(spectrumSimilarities(modelFile, similarityFile))

        for _negative in range(beginW2V, endW2V):
            ## Word2Vec
            parSetup = "_size" + str(_size) + "_window" + str(_window) + "_min_count" + str(_min_count) + "_sample" + str(_sample)
            modelFile = "SGNS_ELMAN"+ parSetup + ".model" 
            similarityFile = "SGNS_ELMAN"+ parSetup + ".similarity"
            cwd = os.path.dirname(os.path.realpath(__file__))
            os.chdir('/Users/fa/workspace/repos/_codes/omerlevy-hyperwords-688addd64ca2')
            subprocess.call(["./corpus2sgns.sh",  "--thr" , str(_min_count), "--win" ,  str(_window), "--sub" , str(_sample), "--dim" , str(_size ),  "--neg" , str(_negative),#"--w+c", 
                "/Users/fa/workspace/repos/_codes/elman/" + corpusFile,  "/Users/fa/workspace/repos/_codes/elman/sgns" ])
            os.chdir(cwd)
    
            with file('/Users/fa/workspace/repos/_codes/elman/sgns/vectors.txt', 'r') as fin: data = fin.read()
            with file(modelFile, 'w') as fout: fout.write(str(len(lg.getVocabulary())) + " " + str(_size ) + "\n" + data)
        
            #w2vResults_par[ _negative, _size] = list(betweenWithinSimilarities_paradigmatic(modelFile, lg.categories,  similarityFile))
            #w2vResults_syn[ _negative, _size] = list(betweenWithinSimilarities_syntagmatic(modelFile, lg.categories,lg.frames,  similarityFile))
            w2vResults_spectrum[ _negative, _size] = list(spectrumSimilarities(modelFile, similarityFile))
     

            
    print("results ready for plotting!")
    #plt.plot(range(begin,end), svdResults[begin-1:end-1,0], label='AVG', color = 'black')
    #plt.plot(range(begin,end), svdResults[begin-1:end-1,1], label='Between', linewidth=4, color='red')
    #plt.plot(range(begin,end), svdResults[begin-1:end-1,2], label='Within', linewidth=4, color='green')
    #plt.plot(range(begin,end), svdResults[begin:end,3], label='SVD (Within - Between)', linestyle='--', color = 'blue')
     
    #plt.plot(range(begin,end), w2vResults[begin:end,0], label='AVG', color = 'black')
    #plt.plot(range(begin,end), w2vResults[begin:end,1], label='Between', linewidth=4, color='red')
    #plt.plot(range(begin,end), w2vResults[begin:end,2], label='Within', linewidth=4, color='green')
    #plt.plot(range(begin,end), w2vResults[begin:end,3], label='W2V Fisher',  color = 'blue')
    
    colors = ['red','green','yellow','orange','violet', 'purple', 'black','blue']
    
    
    plt.subplot(2, 1, 1)
    
    '''  
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_par[index, begin:end,3], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_par[_negative, begin:end,3], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Paradigmatic Task")
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_syn[index, begin:end,3], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_syn[_negative, begin:end,3], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Syntagmatic Task")
    plt.legend()
    plt.show()
    '''
 
    ''''
    plt.subplot(2, 3, 1)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_par[index, begin:end,1], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_par[_negative, begin:end,1], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Paradigmatic Task (Between)")
    plt.legend()
    
    plt.subplot(2, 3, 4)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_syn[index, begin:end,1], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_syn[_negative, begin:end,1], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Syntagmatic Task (Between)")
    plt.legend()
    
    
    plt.subplot(2, 3, 2)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_par[index, begin:end,2], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_par[_negative, begin:end,2], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Paradigmatic Task (Within)")
    plt.legend()
    
    plt.subplot(2, 3, 5)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_syn[index, begin:end,2], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_syn[_negative, begin:end,2], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Syntagmatic Task (Within)")
    plt.legend()
    
    plt.subplot(2, 3, 3)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_par[index, begin:end,3], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_par[_negative, begin:end,3], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Paradigmatic Task (Within - Between)")
    plt.legend()
    
    plt.subplot(2, 3, 6)
    for index, _eig in enumerate([0, 0.5, 1]):
        plt.plot(range(begin,end), svdResults_syn[index, begin:end,3], label= "SVD with " + str(_eig)+ " eig", linestyle='--', linewidth=2, color = colors[index-1]) 
    for _negative in range(beginW2V, endW2V):
        plt.plot(range(begin,end), w2vResults_syn[_negative, begin:end,3], label= "SG with " + str(_negative)+ " negative samples", linewidth=4, color = colors[_negative-1])
    plt.title("Syntagmatic Task (Within - Between)")
    plt.legend()
    
    plt.show()
    
    
    '''
    print(svdResults_spectrum)
    print(w2vResults_spectrum)
    
    plt.plot(range(begin,end), svdResults_spectrum[1, begin:end,0], label= "SVD MIN " , linestyle='--', linewidth=3, color = "red") 
    plt.plot(range(begin,end), svdResults_spectrum[1, begin:end,1], label= "SVD MAX " , linestyle='--', linewidth=3, color = "green") 
    plt.plot(range(begin,end), svdResults_spectrum[1, begin:end,2], label= "SVD AVG " , linestyle='--', linewidth=3, color = "blue") 
    plt.plot(range(begin,end), w2vResults_spectrum[1, begin:end,0], label= "SGNS MIN " , linewidth=4, color = "red") 
    plt.plot(range(begin,end), w2vResults_spectrum[1, begin:end,1], label= "SGNS MAX " , linewidth=4, color = "green") 
    plt.plot(range(begin,end), w2vResults_spectrum[1, begin:end,2], label= "SGNS AVG " , linewidth=4, color = "blue") 
    plt.title("(a)")
    plt.legend()
    
    plt.show()
    
    return plotting
    
def noPlotting(corpusSize, corpusSamples, resultFile):
    
   
    
    ## parameters:
    _size = 5
    _window = 2
    _min_count = 1 
    _sg = 1
    _iter = 1
    _negative = 0
    _sample = 0
    _eig = 0.5
    #_cds = 0.75
    
    
    
    counter = 0
    begin = 2
    end = 15
    beginW2V = 0 
    endW2V = 6
    w2vResults_par =np.zeros(shape=(endW2V,end,corpusSamples,4))
    w2vResults_syn =np.zeros(shape=(endW2V,end,corpusSamples,4))
    w2vResults_spectrum =np.zeros(shape=(endW2V,end,corpusSamples,4))
    svdResults_par =np.zeros(shape=(3,end,corpusSamples,4))
    svdResults_syn =np.zeros(shape=(3,end,corpusSamples,4))
    svdResults_spectrum =np.zeros(shape=(3,end,corpusSamples,4))
    for corpusIndex in range(0,corpusSamples):
        
        print "working on a sample corpus"
        corpusDistribution  = "uniform"
        corpusFile =  "data/ElmanCorpus" + "_distribution" + corpusDistribution + "_size" +  str(corpusSize) + "_" + str(corpusIndex) + ".txt"
        lg = languageGenerator()
        #lg.generateCorpus(corpusFile, corpusDistribution  ,  corpusSize )
        #plt.show()
        
        
        for _size in range(begin,end):
            print("\t\t\t****\t\t\tcounter = " +str(counter)+ "\tcorpusIndex = " + str(corpusIndex) + "\t_size = " + str(_size))
            
            for index, _eig in enumerate([0, 0.5, 1]):
                ## SVD
                parSetup = "_size" + str(_size) + "_window" + str(_window) + "_min_count" + str(_min_count) + "_sample" + str(_sample)
                modelFile = "SVD_ELMAN"+ parSetup + ".model" 
                similarityFile = "SVD_ELMAN"+ parSetup + ".similarity"
                cwd = os.path.dirname(os.path.realpath(__file__))
                os.chdir('/Users/fa/workspace/repos/_codes/omerlevy-hyperwords-688addd64ca2')
                subprocess.call(["./corpus2svd.sh",  "--thr" , str(_min_count), "--win" ,  str(_window), "--sub" , str(_sample), "--dim" , str(_size ) , "--eig", str(_eig),#"--w+c", 
                    "/Users/fa/workspace/repos/_codes/elman/" + corpusFile,  "/Users/fa/workspace/repos/_codes/elman/svd" ])
                os.chdir(cwd)
        
                with file('/Users/fa/workspace/repos/_codes/elman/svd/vectors.txt', 'r') as fin: data = fin.read()
                with file(modelFile, 'w') as fout: fout.write(str(len(lg.getVocabulary())) + " " + str(_size ) + "\n" + data)

                svdResults_par[ index, _size, corpusIndex] = list(betweenWithinSimilarities_paradigmatic(modelFile, lg.categories,  similarityFile))
                svdResults_syn[ index, _size, corpusIndex] = list(betweenWithinSimilarities_syntagmatic(modelFile, lg.categories,lg.frames,  similarityFile))
                #svdResults_spectrum[ index, _size, corpusIndex] = list(spectrumSimilarities(modelFile, similarityFile))
                counter=+1
                
            for _negative in range(beginW2V, endW2V):
                ## Word2Vec
                parSetup = "_size" + str(_size) + "_window" + str(_window) + "_min_count" + str(_min_count) + "_sample" + str(_sample)
                modelFile = "SGNS_ELMAN"+ parSetup + ".model" 
                similarityFile = "SGNS_ELMAN"+ parSetup + ".similarity"
                cwd = os.path.dirname(os.path.realpath(__file__))
                os.chdir('/Users/fa/workspace/repos/_codes/omerlevy-hyperwords-688addd64ca2')
                subprocess.call(["./corpus2sgns.sh",  "--thr" , str(_min_count), "--win" ,  str(_window), "--sub" , str(_sample), "--dim" , str(_size ),  "--neg" , str(_negative),#"--w+c", 
                    "/Users/fa/workspace/repos/_codes/elman/" + corpusFile,  "/Users/fa/workspace/repos/_codes/elman/sgns" ])
                os.chdir(cwd)
        
                with file('/Users/fa/workspace/repos/_codes/elman/sgns/vectors.txt', 'r') as fin: data = fin.read()
                with file(modelFile, 'w') as fout: fout.write(str(len(lg.getVocabulary())) + " " + str(_size ) + "\n" + data)
            
                w2vResults_par[ _negative, _size, corpusIndex] = list(betweenWithinSimilarities_paradigmatic(modelFile, lg.categories,  similarityFile))
                w2vResults_syn[ _negative, _size, corpusIndex] = list(betweenWithinSimilarities_syntagmatic(modelFile, lg.categories,lg.frames,  similarityFile))
                #w2vResults_spectrum[ _negative, _size, corpusIndex] = list(spectrumSimilarities(modelFile, similarityFile))
                counter=+1
                
                
    outf = open(resultFile, 'w')
    outf.write("All parameter manipulations [eig/neg, dimentionality, corpuSample]\n\n")
    outf.write("\nsvdResults_syn\n")
    outf.write(str(svdResults_syn))
    outf.write("\nsvdResults_par\n")
    outf.write(str(svdResults_par))
    outf.write("\nw2vResults_syn\n")
    outf.write(str(w2vResults_syn))    
    outf.write("\nw2vResults_par\n")
    outf.write(str(w2vResults_par))
    
     
    print "BEFORE", svdResults_syn
     
    svdResults_par= svdResults_par.mean(axis=2, dtype=None, out=None, keepdims=True)
    svdResults_syn= svdResults_syn.mean(axis=2, dtype=None, out=None, keepdims=True)
    
    w2vResults_par= w2vResults_par.mean(axis=2, dtype=None, out=None, keepdims=True)
    w2vResults_syn= w2vResults_syn.mean(axis=2, dtype=None, out=None, keepdims=True)
    
    
    print "AFTER",  svdResults_syn
    
    
   # print(svdResults_par[:,:,:,3])
            
   # print(w2vResults_syn[:,:,:,3])
            
   # print(w2vResults_par[:,:,:,3])
    
   
    svdSyn = np.round(svdResults_syn[:,:,:,3],3)
    svdPar = np.round(svdResults_par[:,:,:,3],3)
    w2vSyn = np.round(w2vResults_syn[:,:,:,3],3)
    w2vPar = np.round(w2vResults_par[:,:,:,3],3)
       
            
    print("Method\tPar Min\tPar Max\tPar AVG\tPar Var\tSyn Min\tSyn Max\tSyn AVG\tSyn Var\n")
    
    print("SVD\t" +  str(np.min(svdPar)) + "\t" + str(np.max(svdPar))+ "\t" + str(np.mean(svdPar))+ "\t" + str(np.var(svdPar))  + "\t"
        + str(np.min(svdSyn)) + "\t" + str(np.max(svdSyn))+ "\t" + str(np.mean(svdSyn))+ "\t" + str(np.var(svdSyn)))
    
    print("SGNS\t" + str(np.min(w2vPar)) + "\t" + str(np.max(w2vPar))+ "\t" + str(np.mean(w2vPar))+ "\t" + str(np.var(w2vPar)) + "\t"
        + str(np.min(w2vSyn)) + "\t" + str(np.max(w2vSyn))+ "\t" + str(np.mean(w2vSyn))+ "\t" + str(np.var(w2vSyn))  )    
        
        
    outf.write("\nSVD\t" +  str(np.min(svdPar)) + "\t" + str(np.max(svdPar))+ "\t" + str(np.mean(svdPar))+ "\t" + str(np.var(svdPar))  + "\t"
        + str(np.min(svdSyn)) + "\t" + str(np.max(svdSyn))+ "\t" + str(np.mean(svdSyn))+ "\t" + str(np.var(svdSyn)))
    
    outf.write("\nSGNS\t" + str(np.min(w2vPar)) + "\t" + str(np.max(w2vPar))+ "\t" + str(np.mean(w2vPar))+ "\t" + str(np.var(w2vPar)) + "\t"
        + str(np.min(w2vSyn)) + "\t" + str(np.max(w2vSyn))+ "\t" + str(np.mean(w2vSyn))+ "\t" + str(np.var(w2vSyn))  )
        
    outf.close()
        
    
        
    return noPlotting
    
if __name__ == '__main__':
    #plotting()
    #noPlotting(500, 5, "Exp1_5CorpusSamples_500.txt")
    #noPlotting(1000, 5, "Exp1_5CorpusSamples_1000.txt")
    #noPlotting(5000, 5, "Exp1_5CorpusSamples_5000.txt")
    noPlotting(10000, 5, "Exp1_5CorpusSamples_10000.txt")
    #noPlotting(20000, 5, "Exp1_5CorpusSamples_20000.txt")
    #noPlotting(30000, 5, "Exp1_5CorpusSamples_30000.txt")
    
    #noPlotting(500, 5, "Exp1_5CorpusSamplesZipf_500.txt")
    #noPlotting(1000, 5, "Exp1_5CorpusSamplesZipf_1000_wc.txt")
    #noPlotting(5000, 5, "Exp1_5CorpusSamplesZipf_5000_wc.txt")
    #noPlotting(10000, 5, "Exp1_5CorpusSamplesZipf_10000_wc.txt")
    #noPlotting(20000, 5, "Exp1_5CorpusSamplesZipf_20000_wc.txt")
    #noPlotting(30000, 5, "Exp1_5CorpusSamplesZipf_30000_wc.txt")
    
    