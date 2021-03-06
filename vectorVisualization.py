import sys
import codecs
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim import utils, matutils
import math
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
 
 
def visualize(wv, vocabulary, plotIndex): 
    tsvd = TruncatedSVD(n_components=2,  random_state=None)
    np.set_printoptions(suppress=True)
    Y = tsvd.fit_transform(wv[:]) # wv[100:110,:] only selected words in that range, but all original dimensions are used
    plt.subplot(2, 1, plotIndex)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(-1,  2)
    plt.ylim(-1, +1)
    plt.scatter(Y[:, 0], Y[:, 1])
    for label, x, y in zip(vocabulary[:], Y[:, 0], Y[:, 1]):#zip([vocabulary[x] for x in indices] , Y[indices, 0], Y[indices, 1]): 
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',color='blue', size='14')
 
 
def visualizeSelected(wv, vocabulary, selected, plotIndex): #selection contains the indices of word we want to visualize
    for word in selected:
        if not(word in vocabulary):
            print("Error:: this word does not exist in model vocabulary: " + word)
    category = [ vocabulary.index(word) for word in selected]  
    vocabulary1 = [vocabulary[x] for x in category] 
    wv1 = wv[category,:] 
    print(category)
    
    #tsne = TSNE(n_components=2, random_state=0)
    tsvd = TruncatedSVD(n_components=2,  random_state=None)
    
    np.set_printoptions(suppress=True)
    
    #Y = tsne.fit_transform(wv[:]) 
    Y = tsvd.fit_transform(wv[:])
    plt.subplot(2, 3, plotIndex)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.xlim(-1,  1.5)
    plt.ylim(-1, 1.5)
    plt.scatter(Y[category, 0], Y[category, 1])
    
    for label, x, y in zip(vocabulary1, Y[category, 0], Y[category, 1]):#zip([vocabulary[x] for x in indices] , Y[indices, 0], Y[indices, 1]): 
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    
    
def load_embeddings(file_name):
    with codecs.open(file_name, 'r', 'utf-8') as f_in:
        next(f_in)  #form model files generated by gensim.word2vec first line is a header denoting the dimensions
        vocabulary, wv = zip(*[line.strip().split(' ', 1) for line in f_in])
    print(vocabulary[0:10])
    print("here!")
    print(wv[0:10])
    wv = np.loadtxt(wv)
    return wv, vocabulary
 

if __name__ == '__main__':

    ''''
    m = Word2Vec.load_word2vec_format("/Users/fa/workspace/repos/_codes/CogSci/CHILDES_RAW_size200_window12_min_count1_sg1_iter10_vocabDefined.model",binary=False)
    #"/Users/fa/workspace/repos/_codes/trunk/vectors-phrase.bin", binary=True) 
    m.init_sims(replace=True)
    words = ["man", "woman", "boy", "girl", "king", "queen"]
    wv = [ m[word] for word in words ]
    print(wv[:])
    
    #If you want to visualize diffVecs too
    subtractionTags = ["man-boy", "woman-girl"]
    subtractionVecs = [m["man"] - m["boy"], m["woman"] - m["girl"]] # same result as: [matutils.unitvec(m["man"] - m["boy"]), matutils.unitvec(m["woman"] - m["girl"])]
    print(subtractionTags)
    print(subtractionVecs)
    words = words + subtractionTags
    wv = np.concatenate([wv, subtractionVecs])
    
    visualize(wv, words)
    ''' 
    
    #directory = "/Users/fa/workspace/repos/_codes/elman/"
    directory = "/Users/fa/workspace/repos/_codes/MODELS/"
    
    plt.suptitle('Testing different numbers of iterations')
    for plotIndex, fileName in enumerate([ 
        #"vectorsC.txt",
        #"vectorsW.txt"
        
        #"text8exp/art_sgns_vectorsB.txt",
        #"text8exp/art_sgns_vectorsC.txt",
        #"text8exp/art_sgns_vectorsW.txt",
        #"text8exp/art_cbow_vectorsB.txt",
        #"text8exp/art_cbow_vectorsC.txt",
        #"text8exp/art_cbow_vectorsW.txt"
        
        #"vectors_CHILDES.model",
        #"vectors_GooglePrepared.model",
        
        #"vectors_SGNS_1000Sent.txt",
        #"vectors_SGNS_10000Sent.txt",
        #"vectors_SVD_1000Sent.txt",
        "vectors_SVD_10000Sent.txt",
        
        #"vectors_SGNS_1000Sent_wc.txt",
        #"vectors_SGNS_10000Sent_wc.txt",
        #"vectors_SVD_1000Sent_wc.txt",
        #"vectors_SVD_10000Sent_wc.txt",
        
      
        
        #"CHILDES_2.model", #"CHILDES_250_6_1.model",
        #"vectors.model"
        #"ELMAN_size5_window1_min_count1_sg1_iter1_negative5_sample0_vocabDefined.model"
        #,"ELMAN_size5_window1_min_count1_sg1_iter5_negative5_sample0_vocabDefined.model"
        #,"ELMAN_size5_window1_min_count1_sg1_iter10_negative5_sample0_vocabDefined.model"
        #,"ELMAN_size5_window1_min_count1_sg1_iter20_negative5_sample0_vocabDefined.model"
        ]):
        '''
    plt.suptitle('Testing different numbers of negative samples')
    for plotIndex, fileName in enumerate([ "ELMAN_size20_window1_min_count1_sg1_iter20_negative1_sample0.99_vocabDefined.model"
        ,"ELMAN_size20_window1_min_count1_sg1_iter20_negative5_sample0_vocabDefined.model"
        ,"ELMAN_size20_window1_min_count1_sg1_iter20_negative10_sample0_vocabDefined.model"
        ,"ELMAN_size20_window1_min_count1_sg1_iter20_negative20_sample0_vocabDefined.model"
        ]):
        '''
        
        embeddings_file = directory + fileName
        wv, vocabulary = load_embeddings(embeddings_file) 
        print ("Visualizing from file" + embeddings_file)
        #print(vocabulary[0:10])   
        #selected = ['woman', 'man', 'break', 'eat', 'mouse', 'book', 'cat', 'cookie', 'sandwich' , 'sleep', 'think', 'see', 'rock']
        selected = ['woman', 'man', 'break', 'eat', 'mouse', 'book', 'cat', 'cookies', 'sandwich' , 'sleep', 'think', 'see', 'rock', 'dragon','monster','glass','plate','chase','move','smell','smash']
        #selected = ['mom', 'dad', 'parent', 'father', 'mother', 'brother', 'sister', 'kid', 'child'] #,  'baby', 'daughter', 'son', 'aunt', 'cousin'
        #visualize(wv, vocabulary, plotIndex+1) 
        #selected = [ 'frog', 'giraffe', 'lion', 'fish', 'red','yellow','green', 'brown','boy','girl','man','woman'] #'banana','lemon','strawberry', 'cucumber',
        visualizeSelected(wv, vocabulary, selected, plotIndex+1)
        plt.title(fileName, fontsize=18)
    plt.show()
    
  
    
    '''
    
    category2 = [ vocabulary.index(word) for word in ["boy", "girl"]]  
    vocabulary2 = [vocabulary[x] for x in category2] 
    wv2 = wv[category2,:]
    print(category2)
        
    vocabulary = vocabulary1 + vocabulary2
    wv = np.concatenate((wv1, wv2))
    print(vocabulary)
    print(wv)
    '''