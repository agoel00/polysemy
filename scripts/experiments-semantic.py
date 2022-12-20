# %%
import sys
from pathlib import Path

LEXSUBGEN_ROOT = str(Path().resolve().parent)

if LEXSUBGEN_ROOT not in sys.path:
    sys.path.insert(0, LEXSUBGEN_ROOT)

# %%
import os
os.chdir('LexSubGen')

# %%
from lexsubgen import SubstituteGenerator

# %%
CONFIGS_PATH = Path().resolve() / "configs"

# %%
# Loading substitute generator
sg = SubstituteGenerator.from_config(
    str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet_embs.jsonnet")
)

# %%
sents = ['I went to the bank'.split()]
ids = [4]

# %%
subs, _ = sg.generate_substitutes(
    sents,
    ids
)

# %%
subs

# %%
import pickle
from itertools import combinations

# %%
with open('../networks/syntactic/eng_dep_graphs_control.pkl', 'rb') as f:
    eng_control = pickle.load(f)

with open('../networks/syntactic/eng_dep_graphs_no_control.pkl', 'rb') as f:
    eng_no_control = pickle.load(f)

with open('../Data/mono_poly_semcor_experiments_data_en.pkl','rb') as f:
    eng_full = pickle.load(f)

# %%
"exceed" in eng_control['exceed_v'].nodes

# %%
eng_control_data = {word: eng_full['control'][word] for word in eng_control.keys()}
eng_no_control_data = {word: eng_full['no_control'][word] for word in eng_no_control.keys()}

# %%
len(eng_control_data), len(eng_no_control_data)

# %%
# for k,v in eng_control_data.items():
#     print(k)
#     for i in v:
#         print(i['sentence_words'], i['position'])

# %%
control_semantic_graphs = {}
no_control_semantic_graphs = {}

# %%
import networkx as nx
import tqdm

# %%
def build_semantic_network(eng_dict):
    networks = {}
    main_nodes = []
    for word, wdict in tqdm.tqdm(eng_dict.items(), total=len(eng_dict)):
        edges = []
        for i, w in enumerate(wdict):
            main_nodes.append('{}_{}'.format(word, i))
            # wsents.append(w['sentence_words'])
            # widxs.append(w['position'])
            wsubs, _ = sg.generate_substitutes(
                [w['sentence_words']],
                [w['position']]
            )
            for sub in wsubs[0]:
                edges.append(
                    (
                        '{}_{}'.format(word, i),
                        '{}'.format(sub)
                    )
                )
        g = nx.Graph(edges)
        g.add_edges_from(list(combinations(main_nodes, 2)))
        networks[word] = g
    return networks


# %%
eng_control_networks = build_semantic_network(eng_control_data)

# %%
with open('../networks/semantic/eng_control_networks.pkl', 'wb') as f:
    pickle.dump(eng_control_networks, f)

# %%
eng_no_control_networks = build_semantic_network(eng_no_control_data)

# %%
with open('../networks/semantic/eng_no_control_networks.pkl', 'wb') as f:
    pickle.dump(eng_no_control_networks, f)

# %%
eng_no_control_networks['exceed_v'].nodes

# %% [markdown]
# ## multilingual semantic graphs using fill mask pipeline
# ### french

# %%
with open('../networks/syntactic/fr_dep_graphs_control.pkl', 'rb') as f:
    fr_control = pickle.load(f)

with open('../networks/syntactic/fr_dep_graphs_no_control.pkl', 'rb') as f:
    fr_no_control = pickle.load(f)

with open('../Data/mono_poly_eurosense_experiments_data_fr.pkl','rb') as f:
    fr_full = pickle.load(f)

# %%
fr_control_data = {word: fr_full['control'][word] for word in fr_control.keys()}
fr_no_control_data = {word: fr_full['no_control'][word] for word in fr_no_control.keys()}

# %%
from transformers import pipeline

# %%
pipe = pipeline('fill-mask', model='flaubert/flaubert_base_uncased')

# %%
def build_noneng_semantic_network(noneng_dict, pipe, mask_token='<special1>'):
    networks = {}
    for word, wdict in tqdm.tqdm(noneng_dict.items(), total=len(noneng_dict)):
        edges = []
        main_nodes = []
        for i, w in enumerate(wdict):
            main_nodes.append('{}_{}'.format(word, i))
            # wsents.append(w['sentence_words'])
            # widxs.append(w['position'])
            wstrL = w['sentence'].split()
            wstrL[w['position']] = mask_token
            wstr = " ".join(i for i in wstrL)
            wsubs = pipe(wstr)
            wsubs = [i['token_str'] for i in wsubs]
            for sub in wsubs:
                edges.append(
                    (
                        '{}_{}'.format(word, i),
                        '{}'.format(sub)
                    )
                )
        g = nx.Graph(edges)
        g.add_edges_from(list(combinations(main_nodes, 2)))
        networks[word] = g
    return networks


# %%
fr_control_networks = build_noneng_semantic_network(fr_control_data, pipe=pipe, mask_token='<special1>')

# %%
fr_no_control_networks = build_noneng_semantic_network(fr_no_control_data, pipe=pipe, mask_token='<special1>')

# %%
with open('../networks/semantic/fr_control_networks.pkl', 'wb') as f:
    pickle.dump(fr_control_networks, f)

with open('../networks/semantic/fr_no_control_networks.pkl', 'wb') as f:
    pickle.dump(fr_no_control_networks, f)

# %% [markdown]
# ### spanish

# %%
with open('../networks/syntactic/es_dep_graphs_control.pkl', 'rb') as f:
    es_control = pickle.load(f)

with open('../networks/syntactic/es_dep_graphs_no_control.pkl', 'rb') as f:
    es_no_control = pickle.load(f)

with open('../Data/mono_poly_eurosense_experiments_data_es.pkl','rb') as f:
    es_full = pickle.load(f)

# %%
es_control_data = {word: es_full['control'][word] for word in es_control.keys()}
es_no_control_data = {word: es_full['no_control'][word] for word in es_no_control.keys()}

# %%
pipe = pipeline('fill-mask', model='dccuchile/bert-base-spanish-wwm-uncased')

# %%
es_control_networks = build_noneng_semantic_network(es_control_data, pipe=pipe, mask_token='[MASK]')

# %%
es_no_control_networks = build_noneng_semantic_network(es_no_control_data, pipe=pipe, mask_token='[MASK]')

# %%
with open('../networks/semantic/es_control_networks.pkl', 'wb') as f:
    pickle.dump(es_control_networks, f)

with open('../networks/semantic/es_no_control_networks.pkl', 'wb') as f:
    pickle.dump(es_no_control_networks, f)

# %% [markdown]
# ### greek

# %%
with open('../networks/syntactic/el_dep_graphs_control.pkl', 'rb') as f:
    el_control = pickle.load(f)

with open('../networks/syntactic/el_dep_graphs_no_control.pkl', 'rb') as f:
    el_no_control = pickle.load(f)

with open('../Data/mono_poly_eurosense_experiments_data_el.pkl','rb') as f:
    el_full = pickle.load(f)

# %%
el_control_data = {word: el_full['control'][word] for word in el_control.keys()}
el_no_control_data = {word: el_full['no_control'][word] for word in el_no_control.keys()}

# %%
pipe = pipeline('fill-mask', model='nlpaueb/bert-base-greek-uncased-v1')

# %%
el_control_networks = build_noneng_semantic_network(el_control_data, pipe=pipe, mask_token='[MASK]')

# %%
el_no_control_networks = build_noneng_semantic_network(el_no_control_data, pipe=pipe, mask_token='[MASK]')

# %%
with open('../networks/semantic/el_control_networks.pkl', 'wb') as f:
    pickle.dump(el_control_networks, f)

with open('../networks/semantic/el_no_control_networks.pkl', 'wb') as f:
    pickle.dump(el_no_control_networks, f)

# %%



