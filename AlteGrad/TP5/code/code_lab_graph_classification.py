"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


############## Task 9
# Generate simple dataset
def create_dataset():
    Gs = list()
    y = list()

    for i in range(3,103):
        Gs.append(nx.cycle_graph(i))
        y.append(0)

        Gs.append(nx.path_graph(i))
        y.append(1)

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(G_train), len(all_paths)))
    for i in range(len(G_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 10
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=2000):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    for i, G in enumerate(Gs_train):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            sampled_nodes = np.random.choice(nodes, 3, replace=False)
            subG = G.subgraph(sampled_nodes)

            if nx.is_isomorphic(subG, graphlets[0]):
                phi_train[i,0] += 1
            elif nx.is_isomorphic(subG, graphlets[1]):
                phi_train[i,1] += 1
            elif nx.is_isomorphic(subG, graphlets[2]):
                phi_train[i,2] += 1
            elif nx.is_isomorphic(subG, graphlets[3]):
                phi_train[i,3] += 1
            else:
                print("error")



    phi_test = np.zeros((len(G_test), 4))
    
    for i, G in enumerate(Gs_test):
        nodes = list(G.nodes())
        for _ in range(n_samples):
            sampled_nodes = np.random.choice(nodes, 3, replace=False)
            subG = G.subgraph(sampled_nodes)

            if nx.is_isomorphic(subG, graphlets[0]):
                phi_test[i,0] += 1
            elif nx.is_isomorphic(subG, graphlets[1]):
                phi_test[i,1] += 1
            elif nx.is_isomorphic(subG, graphlets[2]):
                phi_test[i,2] += 1
            elif nx.is_isomorphic(subG, graphlets[3]):
                phi_test[i,3] += 1
            else:
                print("error")

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)
K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test)



############## Task 11
# Initialize SVM and train
clf = SVC( kernel='precomputed') 
clf.fit(K_train_sp, y_train )
# Predict
y_pred_sp = clf.predict (K_test_sp)

clf = SVC( kernel='precomputed') 
clf.fit(K_train_gl, y_train )
# Predict
y_pred_gl = clf.predict (K_test_gl)

############## Task 12
print("Accuracy of shorest path kernel:", accuracy_score(y_test, y_pred_sp))

print("Accuracy of graphlet kernel:", accuracy_score(y_test, y_pred_gl))