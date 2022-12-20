# %%
import numpy as np 
import networkx as nx 
from scipy.stats import pearsonr, spearmanr
import pickle
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from nltk.corpus import wordnet as wn

# %% [markdown]
# ## Syntactic

# %%
with open('./networks/syntactic/eng_dep_graphs_control.pkl', 'rb') as f:
    en_syn_control = pickle.load(f)

with open('./networks/syntactic/eng_dep_graphs_no_control.pkl', 'rb') as f:
    en_syn_no_control = pickle.load(f)

with open('./networks/syntactic/fr_dep_graphs_control.pkl', 'rb') as f:
    fr_syn_control = pickle.load(f)

with open('./networks/syntactic/fr_dep_graphs_no_control.pkl', 'rb') as f:
    fr_syn_no_control = pickle.load(f)

with open('./networks/syntactic/es_dep_graphs_control.pkl', 'rb') as f:
    es_syn_control = pickle.load(f)

with open('./networks/syntactic/es_dep_graphs_no_control.pkl', 'rb') as f:
    es_syn_no_control = pickle.load(f)

with open('./networks/syntactic/el_dep_graphs_control.pkl', 'rb') as f:
    el_syn_control = pickle.load(f)

with open('./networks/syntactic/el_dep_graphs_no_control.pkl', 'rb') as f:
    el_syn_no_control = pickle.load(f)


# %%
def calc_node_entropy(data_dict):
    results = {}

    for w, g in data_dict.items():
        word = w.split('_')[0]
        A = nx.adjacency_matrix(g).todense()
        degrees = [val for key, val in g.degree()]
        degrees = np.array(degrees)
        A2 = np.linalg.matrix_power(A,2)
        A2 = A2/A2.sum(axis=0)
        D = degrees.T @ A2
        D = D/D.sum()
        idx = list(g.nodes).index(word)
        d = D[:,idx]
        h = - (d * np.log(d))
        results[word] = np.asarray(h.item()).reshape(1)[0]
    return results

# %%
en_syntax_control_results = calc_node_entropy(en_syn_control)
en_syntax_no_control_results = calc_node_entropy(en_syn_no_control)

fr_syntax_control_results = calc_node_entropy(fr_syn_control)
fr_syntax_no_control_results = calc_node_entropy(fr_syn_no_control)

es_syntax_control_results = calc_node_entropy(es_syn_control)
es_syntax_no_control_results = calc_node_entropy(es_syn_no_control)

el_syntax_control_results = calc_node_entropy(el_syn_control)
el_syntax_no_control_results = calc_node_entropy(el_syn_no_control)

# %%
en_words_wordnet_control = {
    word: len(wn.synsets(word)) for word in en_syntax_control_results.keys()
}

en_words_wordnet_no_control = {
    word: len(wn.synsets(word)) for word in en_syntax_no_control_results.keys()
}

fr_words_wordnet_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_syntax_control_results.keys()
}

fr_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_syntax_no_control_results.keys()
}

es_words_wordnet_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_syntax_control_results.keys()
}

es_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_syntax_no_control_results.keys()
}


# %%
print('PearsonR | Control: ',
    pearsonr(
    list(en_words_wordnet_control.values()),
    list(en_syntax_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(en_words_wordnet_no_control.values()),
    list(en_syntax_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(en_words_wordnet_control.values()),
    list(en_syntax_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(en_words_wordnet_no_control.values()),
    list(en_syntax_no_control_results.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(fr_words_wordnet_control.values()),
    list(fr_syntax_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_syntax_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(fr_words_wordnet_control.values()),
    list(fr_syntax_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_syntax_no_control_results.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(es_words_wordnet_control.values()),
    list(es_syntax_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(es_words_wordnet_no_control.values()),
    list(es_syntax_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(es_words_wordnet_control.values()),
    list(es_syntax_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(es_words_wordnet_no_control.values()),
    list(es_syntax_no_control_results.values())
))

# %%


# %% [markdown]
# ## Semantic

# %%
import pickle

# %%
with open('./results/semantic/eng_control_sem_results.pkl', 'rb') as f:
    en_sem_control = pickle.load(f)

with open('./results/semantic/eng_no_control_sem_results.pkl', 'rb') as f:
    en_sem_no_control = pickle.load(f)

with open('./results/semantic/fr_control_sem_results.pkl', 'rb') as f:
    fr_sem_control = pickle.load(f)

with open('./results/semantic/fr_no_control_sem_results.pkl', 'rb') as f:
    fr_sem_no_control = pickle.load(f)

with open('./results/semantic/es_control_sem_results.pkl', 'rb') as f:
    es_sem_control = pickle.load(f)

with open('./results/semantic/es_no_control_sem_results.pkl', 'rb') as f:
    es_sem_no_control = pickle.load(f)

with open('./results/semantic/el_control_sem_results.pkl', 'rb') as f:
    el_sem_control = pickle.load(f)

with open('./results/semantic/el_no_control_sem_results.pkl', 'rb') as f:
    el_sem_no_control = pickle.load(f)


# %%
en_control = {
    k: en_syntax_control_results[k]*en_sem_control[k]
    for k in en_sem_control.keys()
}

en_no_control = {
    k: en_syntax_no_control_results[k]*en_sem_no_control[k]
    for k in en_sem_no_control.keys()
}

# %%
fr_control = {
    k: en_syntax_control_results[k]*en_sem_control[k]
    for k in en_sem_control.keys()
}

fr_no_control = {
    k: fr_syntax_no_control_results[k]*fr_sem_no_control[k]
    for k in fr_sem_no_control.keys()
}

# %%
es_control = {
    k: en_syntax_control_results[k]*en_sem_control[k]
    for k in en_sem_control.keys()
}

es_no_control = {
    k: es_syntax_no_control_results[k]*es_sem_no_control[k]
    for k in es_sem_no_control.keys()
}

# %%
# el_control = {
#     k: el_syntax_control_results[k]*el_sem_control[k]
#     for k in el_sem_control.keys()
# }

# el_no_control = {
#     k: el_syntax_no_control_results[k]*el_sem_no_control[k]
#     for k in el_sem_no_control.keys()
# }

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(en_words_wordnet_control.values()),
    list(en_control.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(en_words_wordnet_no_control.values()),
    list(en_no_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(en_words_wordnet_control.values()),
    list(en_control.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(en_words_wordnet_no_control.values()),
    list(en_no_control.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(fr_words_wordnet_control.values()),
    list(fr_control.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_no_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(fr_words_wordnet_control.values()),
    list(fr_control.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_no_control.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(es_words_wordnet_control.values()),
    list(es_control.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(es_words_wordnet_no_control.values()),
    list(es_no_control.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(es_words_wordnet_control.values()),
    list(es_control.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(es_words_wordnet_no_control.values()),
    list(es_no_control.values())
))

# %%



