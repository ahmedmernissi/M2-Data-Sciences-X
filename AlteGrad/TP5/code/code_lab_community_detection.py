"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G)
    D =  np.zeros((n,n))
    for i, node in enumerate(G.nodes()):
        D[i,i] = G.degree(node)

    L = D - A

    eigvals, eigvecs = eigs(L, k=100, which='SR')
    eigvecs = eigvecs.real

    km = KMeans(n_clusters=k)
    km.fit(eigvecs)

    clustering = dict()
    for i, node in enumerate(G.nodes()):
        clustering[node] = km.labels_[i]
    
    return clustering



############## Task 6

G = nx.read_edgelist("../datasets/CA-HepTh.txt",comments='#',
                        delimiter='\t', create_using=nx.Graph())
gcc_nodes = max(nx.connected_components(G),key=len)
gcc = G.subgraph(gcc_nodes)

clustering = spectral_clustering(gcc, k=50)

############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    modularity = 0
    clusters = {}
    m = G.number_of_edges()

    for node in clustering:
        if clustering[node] not in clusters:
            clusters[clustering[node]] = [node]
        else:
            clusters[clustering[node]].append(node)

    for cluster in clusters:
        nodes_in_cluster = clusters[cluster]

        subG = G.subgraph(nodes_in_cluster)
        l_c = subG.number_of_edges()

        d_c = 0

        for node in nodes_in_cluster:
            d_c += G.degree(node)

        modularity += (l_c/m) - (d_c/(2*m))**2
    
    return modularity



############## Task 8

print("Modularity spectral clustering:", modularity(gcc, clustering))

random_clustering = {}
for node in gcc.nodes():
    random_clustering[node] = randint(0,49)

print("Modularity random clustering:", modularity(gcc, random_clustering))