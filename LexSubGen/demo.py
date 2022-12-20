import stanza
import spacy_stanza
import networkx as nx
import matplotlib.pyplot as plt 
import gradio as gr 
import copy
import numpy as np

import sys
from pathlib import Path

LEXSUBGEN_ROOT = str(Path().resolve().parent)

if LEXSUBGEN_ROOT not in sys.path:
    sys.path.insert(0, LEXSUBGEN_ROOT)

from lexsubgen import SubstituteGenerator

CONFIGS_PATH = Path().resolve() / "configs"

sg = SubstituteGenerator.from_config(
    str(CONFIGS_PATH / "subst_generators" / "lexsub" / "xlnet_embs.jsonnet")
)

nlp = spacy_stanza.load_pipeline('en')
plt.switch_backend('Agg') 


def core(lines, word):
    plt.clf()
    word = " ".join([i.lemma_ for i in nlp(word)])
    sents = [i.strip() for i in lines.split('|')]

    # getting dep tree combined graph
    docs = [nlp(sent) for sent in sents]
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
    centrality = nx.eigenvector_centrality(graph)

    # calculating entropy of dep tree graph

    A = nx.adjacency_matrix(graph).todense()
    degrees = [val for key,val in graph.degree()]
    degrees = np.array(degrees)
    A2 = np.linalg.matrix_power(A,2)
    A2 = A2/A2.sum(axis=0)
    D = degrees.T @ A2
    D = D/D.sum()
    idx  = list(graph.nodes).index(word)
    d = D[:,idx]
    h = - (d * np.log(d))

    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')
    nx.draw(graph, pos, with_labels=True)


    fig = plt.gcf()

    # return fig, centrality[word]
    return fig, h.item()
    

iface = gr.Interface(
    fn=core,
    inputs=[gr.inputs.Textbox(lines=10, placeholder='Enter | separated sentences'), "text"],
    outputs=[gr.outputs.Image(type='plot'), "number"],
    server_port=7860
)
iface.launch()

#     ego = nx.ego_graph(graph, word, radius=2)
#     # pos = nx.nx_agraph.graphviz_layout(ego, prog='dot')
#     pos = nx.layout.spectral_layout(ego)
#     pos = nx.spring_layout(ego, pos=pos, iterations=50)

#     pos_shadow = copy.deepcopy(pos)
#     shift_amount = 0.006
#     for idx in pos_shadow:
#         pos_shadow[idx][0] += shift_amount
#         pos_shadow[idx][1] -= shift_amount
# # https://gist.github.com/jg-you/144a35013acba010054a2cc4a93b07c7
#     nx.draw_networkx_nodes(ego, pos_shadow, node_color='k', alpha=0.5)
#     nx.draw_networkx_nodes(ego, pos, node_color="#3182bd", linewidths=1)
#     nx.draw_networkx_labels(ego, pos)
#     nx.draw_networkx_edges(ego, pos, width=1)