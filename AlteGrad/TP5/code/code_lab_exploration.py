"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1 ##############
G = nx.read_edgelist("../datasets/CA-HepTh.txt",comments='#',
                        delimiter='\t', create_using=nx.Graph())

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())



############## Task 2 ##############
print("Number of connected components", nx.number_connected_components(G))
gcc_nodes = max(nx.connected_components(G),key=len)
gcc = G.subgraph(gcc_nodes)
print("Fraction of nodes in gcc:",
            gcc.number_of_nodes()/G.number_of_nodes())
print("Fraction of edges in gcc:",
            gcc.number_of_edges()/G.number_of_edges())

############## Task 3 ##############
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print("Max degree:", np.max(degree_sequence))
print("Min degree:", np.min(degree_sequence))
print("Median degree:", np.median(degree_sequence))
print("Mean degree:", np.mean(degree_sequence))

############## Task 4 
hist = nx.degree_histogram(G)
plt.plot(hist)
plt.title("Degree distribution")
plt.xlabel("Degree")
plt.ylabel("Frequency")
#plt.xscale('log')
plt.yscale('log')
plt.show()