import sys, numpy as np
import scipy
import scipy.spatial.distance as dist
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

###################################################################################################################
category_filehandle = open("categories.txt")#open(sys.argv[1])
similarity_filename = "ELMAN_size20_window10_min_count1_sg1_iter10_vocabDefined.similarity"
similarity_filehandle = open(similarity_filename) #open(sys.argv[2])

###################################################################################################################
print "Importing Category Definitions Data"
category_list = []
category_dict = {}
num_categories = 0
target_list = []
target_dict = {}
target_category_dict = {}
category_target_list_dict = {}
num_targets = 0

for line in category_filehandle:
    data = (line.strip().strip('\n').strip()).split()
    print(data)
    category = data[0]
    target = data[1]
    if not category in category_dict:
        category_dict[category] = num_categories
        category_list.append(category)
        category_target_list_dict[category] = []
        num_categories += 1
    if not target in target_dict:
        target_dict[target] = num_targets
        target_list.append(target)
        num_targets += 1
    target_category_dict[target] = category
    category_target_list_dict[category].append(target)
category_filehandle.close()

###################################################################################################################
print "Importing Similarity Data"
sim_dict = {}
master_sim_matrix = np.zeros([num_targets, num_targets], float)
for line in similarity_filehandle:
    data = (line.strip().strip('\n').strip()).split()
    word1 = data[0]
    word2 = data[1]
    sim = float(data[2])
    index1 = target_dict[word1]
    index2 = target_dict[word2]
    master_sim_matrix[index1, index2] = sim
    sim_dict[(word1, word2)] = sim
similarity_filehandle.close()

###################################################################################################################
print "Calculating Linkages"
category_sim_matrix_list = []
category_linkage_list = []
for i in range(num_categories):
    current_category = category_list[i]
    current_targets = category_target_list_dict[current_category]
    num_current_targets = len(current_targets)
    current_sim_matrix = np.zeros([num_current_targets, num_current_targets], float)
    for j in range(num_current_targets):
        word1 = current_targets[j]
        for k in range(num_current_targets):
            word2 = current_targets[k]
            sim = sim_dict[(word1,word2)]
            current_sim_matrix[k,j] = sim
            
    category_sim_matrix_list.append(current_sim_matrix)
    current_linkage = linkage(current_sim_matrix, 'complete', 'correlation')
    category_linkage_list.append(current_linkage)
    
###################################################################################################################
print "Showing Dendrograms"
for i in range(num_categories):
    current_category = category_list[i]
    current_targets = category_target_list_dict[current_category]
    current_linkage = category_linkage_list[i]
    

    plt.title(similarity_filename + ":" + current_category, fontsize=18)
    
    dendrogram(current_linkage,
           color_threshold=0.8,
           leaf_label_func=lambda x: current_targets[x],
           orientation='left',
           leaf_font_size=14, 
           )
    plt.show()