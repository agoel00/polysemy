# %%
import numpy as np 
import pandas as pd 
import networkx as nx
import pickle
import random 
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
import numpy as np
import re
import json
import pandas as pd
import sys
from collections import Counter
from operator import itemgetter
import csv
import time
import os
import torch

random.seed(42)

# %%
# control

# %%
with open('baseline-data/control/eng_word_elmo_embs.pkl','rb') as f:
    eng_elmo_embs_control = pickle.load(f)

with open('baseline-data/control/fr_word_elmo_embs.pkl','rb') as f:
    fr_elmo_embs_control = pickle.load(f)

with open('baseline-data/control/es_word_elmo_embs.pkl','rb') as f:
    es_elmo_embs_control = pickle.load(f)

with open('baseline-data/control/el_word_elmo_embs.pkl','rb') as f:
    el_elmo_embs_control = pickle.load(f)

with open('baseline-data/no-control/eng_word_elmo_embs.pkl','rb') as f:
    eng_elmo_embs_no_control = pickle.load(f)

with open('baseline-data/no-control/fr_word_elmo_embs.pkl','rb') as f:
    fr_elmo_embs_no_control = pickle.load(f)

with open('baseline-data/no-control/es_word_elmo_embs.pkl','rb') as f:
    es_elmo_embs_no_control = pickle.load(f)

with open('baseline-data/no-control/el_word_elmo_embs.pkl','rb') as f:
    el_elmo_embs_no_control = pickle.load(f)

# %%
np.array(eng_elmo_embs_control['exceed']).shape

# %%
from sklearn.decomposition import PCA 
import numpy as np
import re
import json
import pandas as pd
import sys
from collections import Counter
from operator import itemgetter
import csv
import time
import os

# %%
from nltk.corpus import wordnet as wn

# %%
D = 2
L = 8

# %%
pca = PCA(2)

# %%
def calc_baseline_results(words_dict):
    embs = []
    labels = []
    for word in words_dict.keys():
        embs.append(words_dict[word])
        labels.extend([word]*len(words_dict[word]))
    x = np.array(embs).reshape(-1, 1024)
    x = pca.fit_transform(x)
    
    number_of_embeddings, embeddings_dimension = x.shape

    histo_counters = []
    for j in range(L):
        # Map that stores the cells where a word appears into (word: [cell_i, ...,
        # cell_n])
        histoL_counter = {}
        # print('|--starting layer ', j)
        # Number of cells along each dimension at level j
        k = 2**j
        # Determines the cells in which each vertex lies
        # along each dimension since nodes lie in the unit
        # hypercube in R^d
        T = np.floor(x * k)
        T[np.where(T == k)] = k - 1
        T = T.astype(int)
        for vector_index, label in enumerate(labels):
            # Identify the cell into which the i-th
            # vertex lies and increase its value by 1
            cell = ",".join([str(element) for element in T[vector_index]])
            current_word = histoL_counter.get(label, set())
            current_word.add(cell)
            histoL_counter[label] = current_word

        histo_counters.append({word: len(cells)
                                for word, cells in histoL_counter.items()})
    total_histo_counter = {}
    # Using 'current_L' to calculate the pyramid with different L value faster
    for current_L in range(1, L):
        for l, counters in enumerate(histo_counters[:current_L]):
            coef = 1.0 / (2**(current_L - l))
            for word, cell_counter in counters.items():
                total_histo_counter[word] = total_histo_counter.get(
                    word, 0) + (coef * (float(cell_counter) / (embeddings_dimension * 2**l)))

            # Transform the dictionary for sorting and printing
            tmp = [(word, round(cell_counter, 3)) for word, cell_counter in total_histo_counter.items()]
            tmp = sorted(tmp, key=itemgetter(1))
            # with open(os.join(output_directory, "pca" + str(pca) + "_L" + str(current_L)), 'w') as f:
            #     f.write("\n".join([",".join([str(j)
            #                                   for j in i]) for i in tmp]))
            # print("\n".join([",".join([str(j)
            #                                      for j in i]) for i in tmp]))
    return total_histo_counter


# %%
en_baseline_res_control = calc_baseline_results(eng_elmo_embs_control)
en_baseline_res_no_control = calc_baseline_results(eng_elmo_embs_no_control)

# %%
en_words_wordnet_control = {
    word: len(wn.synsets(word)) for word in en_baseline_res_control.keys()
}

en_words_wordnet_no_control = {
    word: len(wn.synsets(word)) for word in en_baseline_res_no_control.keys()
}

# %%
from scipy.stats import spearmanr, pearsonr

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(en_words_wordnet_control.values()),
    list(en_baseline_res_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(en_words_wordnet_control.values()),
    list(en_baseline_res_control.values())
))

print('PearsonR | Non Control: ',
    pearsonr(
    list(en_words_wordnet_no_control.values()),
    list(en_baseline_res_no_control.values())
))

print('SpearmanR | Non Control: ',
    spearmanr(
    list(en_words_wordnet_no_control.values()),
    list(en_baseline_res_no_control.values())
))

# %%
fr_baseline_res_control = calc_baseline_results(fr_elmo_embs_control)
fr_baseline_res_no_control = calc_baseline_results(fr_elmo_embs_no_control)

fr_words_wordnet_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_baseline_res_control.keys()
}

fr_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_baseline_res_no_control.keys()
}

# %%
es_baseline_res_control = calc_baseline_results(es_elmo_embs_control)
es_baseline_res_no_control = calc_baseline_results(es_elmo_embs_no_control)

es_words_wordnet_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_baseline_res_control.keys()
}

es_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_baseline_res_no_control.keys()
}

# %%
el_baseline_res_control = calc_baseline_results(el_elmo_embs_control)
el_baseline_res_no_control = calc_baseline_results(el_elmo_embs_no_control)

el_words_wordnet_control = {
    word: len(wn.synsets(word, lang='ell')) for word in el_baseline_res_control.keys()
}

el_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='ell')) for word in el_baseline_res_no_control.keys()
}

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(fr_words_wordnet_control.values()),
    list(fr_baseline_res_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(fr_words_wordnet_control.values()),
    list(fr_baseline_res_control.values())
))

print('PearsonR | Non Control: ',
    pearsonr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_baseline_res_no_control.values())
))

print('SpearmanR | Non Control: ',
    spearmanr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_baseline_res_no_control.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(es_words_wordnet_control.values()),
    list(es_baseline_res_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(es_words_wordnet_control.values()),
    list(es_baseline_res_control.values())
))

print('PearsonR | Non Control: ',
    pearsonr(
    list(es_words_wordnet_no_control.values()),
    list(es_baseline_res_no_control.values())
))

print('SpearmanR | Non Control: ',
    spearmanr(
    list(es_words_wordnet_no_control.values()),
    list(es_baseline_res_no_control.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(el_words_wordnet_control.values()),
    list(el_baseline_res_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(el_words_wordnet_control.values()),
    list(el_baseline_res_control.values())
))

print('PearsonR | Non Control: ',
    pearsonr(
    list(el_words_wordnet_no_control.values()),
    list(el_baseline_res_no_control.values())
))

print('SpearmanR | Non Control: ',
    spearmanr(
    list(el_words_wordnet_no_control.values()),
    list(el_baseline_res_no_control.values())
))

# %%



