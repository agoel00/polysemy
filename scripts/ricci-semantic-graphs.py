# %%
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import pickle

# %%
with open('./semantic/eng_control_networks.pkl', 'rb') as f:
    en_sem_control = pickle.load(f)

with open('./semantic/eng_no_control_networks.pkl', 'rb') as f:
    en_sem_no_control = pickle.load(f)

with open('./semantic/fr_control_networks.pkl', 'rb') as f:
    fr_sem_control = pickle.load(f)

with open('./semantic/fr_no_control_networks.pkl', 'rb') as f:
    fr_sem_no_control = pickle.load(f)

with open('./semantic/es_control_networks.pkl', 'rb') as f:
    es_sem_control = pickle.load(f)

with open('./semantic/es_no_control_networks.pkl', 'rb') as f:
    es_sem_no_control = pickle.load(f)

with open('./semantic/el_control_networks.pkl', 'rb') as f:
    el_sem_control = pickle.load(f)

with open('./semantic/el_no_control_networks.pkl', 'rb') as f:
    el_sem_no_control = pickle.load(f)


# %%
import tqdm

# %%
def calc_ricci(data_dict):
    results = {}
    
    for w, g in tqdm.tqdm(data_dict.items(), total=len(data_dict)):
        word = w.split('_')[0]
        orc = OllivierRicci(g, alpha=0.5)
        orc.compute_ricci_curvature()
        g = orc.G.copy()
        riccis = [v['ricciCurvature'] for k,v in dict(g.nodes(data=True)).items() if word in k]
        results[word] = 1/np.mean(riccis)
#         riccis = [i[2]['ricciCurvature'] for i in g.edges(data=True)]
#         numer = sum([1 for i in riccis if i<np.mean(riccis)])
#         denom = len(riccis)
#         results[word] = numer/denom

    return results

# %%
en_semantic_control_results = calc_ricci(en_sem_control)
en_semantic_no_control_results = calc_ricci(en_sem_no_control)

fr_semantic_control_results = calc_ricci(fr_sem_control)
fr_semantic_no_control_results = calc_ricci(fr_sem_no_control)

es_semantic_control_results = calc_ricci(es_sem_control)
es_semantic_no_control_results = calc_ricci(es_sem_no_control)

el_semantic_control_results = calc_ricci(el_sem_control)
el_semantic_no_control_results = calc_ricci(el_sem_no_control)

# %%
en_semantic_control_results

# %%
from nltk.corpus import wordnet as wn

# %%
en_words_wordnet_control = {
    word: len(wn.synsets(word)) for word in en_semantic_control_results.keys()
}

en_words_wordnet_no_control = {
    word: len(wn.synsets(word)) for word in en_semantic_no_control_results.keys()
}

fr_words_wordnet_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_semantic_control_results.keys()
}

fr_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='fra')) for word in fr_semantic_no_control_results.keys()
}

es_words_wordnet_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_semantic_control_results.keys()
}

es_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='spa')) for word in es_semantic_no_control_results.keys()
}

el_words_wordnet_control = {
    word: len(wn.synsets(word, lang='ell')) for word in el_semantic_control_results.keys()
}

el_words_wordnet_no_control = {
    word: len(wn.synsets(word, lang='ell')) for word in el_semantic_no_control_results.keys()
}


# %%
from scipy.stats import pearsonr, spearmanr

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(en_words_wordnet_control.values()),
    list(en_semantic_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(en_words_wordnet_no_control.values()),
    list(en_semantic_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(en_words_wordnet_control.values()),
    list(en_semantic_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(en_words_wordnet_no_control.values()),
    list(en_semantic_no_control_results.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(fr_words_wordnet_control.values()),
    list(fr_semantic_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_semantic_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(fr_words_wordnet_control.values()),
    list(fr_semantic_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(fr_words_wordnet_no_control.values()),
    list(fr_semantic_no_control_results.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(es_words_wordnet_control.values()),
    list(es_semantic_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(es_words_wordnet_no_control.values()),
    list(es_semantic_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(es_words_wordnet_control.values()),
    list(es_semantic_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(es_words_wordnet_no_control.values()),
    list(es_semantic_no_control_results.values())
))

# %%
print('PearsonR | Control: ',
    pearsonr(
    list(el_words_wordnet_control.values()),
    list(el_semantic_control_results.values())
))

print('PearsonR | No Control: ',
    pearsonr(
    list(el_words_wordnet_no_control.values()),
    list(el_semantic_no_control_results.values())
))

print('SpearmanR | Control: ',
    spearmanr(
    list(el_words_wordnet_control.values()),
    list(el_semantic_control_results.values())
))

print('SpearmanR | No Control: ',
    spearmanr(
    list(el_words_wordnet_no_control.values()),
    list(el_semantic_no_control_results.values())
))

# %%



