# %%
import pickle
import numpy as np
import stanza
import spacy_stanza
from nltk.corpus import wordnet as wn 
from spacy import displacy
import networkx as nx
import random

random.seed(42)

# %%
with open('Data/mono_poly_semcor_experiments_data_en.pkl', 'rb') as f:
    eng = pickle.load(f)

with open('Data/mono_poly_eurosense_experiments_data_el.pkl', 'rb') as f:
    greek = pickle.load(f)

with open('Data/mono_poly_eurosense_experiments_data_es.pkl', 'rb') as f:
    spanish = pickle.load(f)

with open('Data/mono_poly_eurosense_experiments_data_fr.pkl', 'rb') as f:
    french = pickle.load(f)

# %%
print(len(eng['no_control']))
print(len(french['no_control']))
print(len(spanish['no_control']))
print(len(greek['no_control']))

# %%
eng['no_control'] = dict(random.sample(eng['no_control'].items(), 15))
eng['control'] = {i: eng['control'][i] for i in eng['no_control'].keys()}

french['no_control'] = dict(random.sample(french['no_control'].items(), 15))
french['control'] = {i: french['control'][i] for i in french['no_control'].keys()}

spanish['no_control'] = dict(random.sample(spanish['no_control'].items(), 15))
spanish['control'] = {i: spanish['control'][i] for i in spanish['no_control'].keys()}

greek['no_control'] = dict(random.sample(greek['no_control'].items(), 15))
greek['control'] = {i: greek['control'][i] for i in greek['no_control'].keys()}


# %%
print(len(eng['no_control']))
print(len(french['no_control']))
print(len(spanish['no_control']))
print(len(greek['no_control']))

# %%
# stanza.download('fr')
# stanza.download('es')
# stanza.download('el')

# %%
en_nlp = spacy_stanza.load_pipeline('en')
fr_nlp = spacy_stanza.load_pipeline('fr')
es_nlp = spacy_stanza.load_pipeline('es')
gr_nlp = spacy_stanza.load_pipeline('el')

nlps = {
    'en': en_nlp,
    'es': es_nlp,
    'fr': fr_nlp,
    'el': gr_nlp
}

# %%
def build_syntactic_network(sents, lang='en'):
    nlp = nlps[lang]
    docs = [nlp(sent) for sent in sents]
    edges = []
    edges = []
    for doc in docs:
        for token in doc:
            for child in token.children:
                edges.append(
                    (
                        '{}'.format(token.lemma_),
                        '{}'.format(child.lemma_)
                    )
                )
    graph = nx.Graph(edges)
    return graph
    

# %%
# oppcontrol = polysame
# control = polybal
# nocontrol = polyrand

# %%
eng_dep_graphs_control = {}

for k,v in eng['control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents)
    print(nx.info(g))
    eng_dep_graphs_control[k] = g

eng_dep_graphs_no_control = {}

for k,v in eng['no_control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents)
    print(nx.info(g))
    eng_dep_graphs_no_control[k] = g
    

# %%
# nx.draw(eng_dep_graphs_control['tape_n'], with_labels=True)

# %%
# langs
# french=fra
# spanish=spa
# greek=ell

# %%
fr_dep_graphs_control = {}

for k,v in french['control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='fr')
    print(nx.info(g))
    fr_dep_graphs_control[k] = g

fr_dep_graphs_no_control = {}

for k,v in french['no_control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='fr')
    print(nx.info(g))
    fr_dep_graphs_no_control[k] = g
    

# %%
es_dep_graphs_control = {}

for k,v in spanish['control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='es')
    print(nx.info(g))
    es_dep_graphs_control[k] = g

es_dep_graphs_no_control = {}

for k,v in spanish['no_control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='es')
    print(nx.info(g))
    es_dep_graphs_no_control[k] = g
    

# %%
el_dep_graphs_control = {}

for k,v in greek['control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='el')
    print(nx.info(g))
    el_dep_graphs_control[k] = g

el_dep_graphs_no_control = {}

for k,v in greek['no_control'].items():
    sents = [i['sentence'] for i in v]
    # print(k, len(sents), sents[0])
    g = build_syntactic_network(sents, lang='el')
    print(nx.info(g))
    el_dep_graphs_no_control[k] = g
    

# %%
with open('./networks/syntactic/eng_dep_graphs_control.pkl', 'wb+') as f:
    pickle.dump(eng_dep_graphs_control, f)

with open('./networks/syntactic/eng_dep_graphs_no_control.pkl', 'wb+') as f:
    pickle.dump(eng_dep_graphs_no_control, f)
    

with open('./networks/syntactic/fr_dep_graphs_control.pkl', 'wb+') as f:
    pickle.dump(fr_dep_graphs_control, f)

with open('./networks/syntactic/fr_dep_graphs_no_control.pkl', 'wb+') as f:
    pickle.dump(fr_dep_graphs_no_control, f)
    


with open('./networks/syntactic/es_dep_graphs_control.pkl', 'wb+') as f:
    pickle.dump(es_dep_graphs_control, f)

with open('./networks/syntactic/es_dep_graphs_no_control.pkl', 'wb+') as f:
    pickle.dump(es_dep_graphs_no_control, f)
    


with open('./networks/syntactic/el_dep_graphs_control.pkl', 'wb+') as f:
    pickle.dump(el_dep_graphs_control, f)

with open('./networks/syntactic/el_dep_graphs_no_control.pkl', 'wb+') as f:
    pickle.dump(el_dep_graphs_no_control, f)
    

# %%
# print(nx.info(el_dep_graphs_no_control['καταιγίδα_n']))
# print(nx.info(el_dep_graphs_control['καταιγίδα_n']))

# %%


# %%



