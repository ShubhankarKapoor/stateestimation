#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 08:14:50 2020

@author: shub
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

which = 37 # IEEE 37-node or IEEE 906-node

if which == 37:
    from Network37 import *
    
# instantiate the graph
G = nx.Graph()
# add nodes
G.add_nodes_from(np.arange(0, which))
# add edges
G.add_edges_from((arcs))
# plot graph
plt.figure()
nx.draw(G, with_labels=True, font_weight='bold')

# get the path to nodes from the slack bus i.e. node 0
path_nodes = [nx.shortest_path(G, 0, i) for i in range(which)]
# convert to dictinary with vals as set of tuples denoting the path to nodes
path_to_all_nodes = {key:set(zip(val[:-1], val[1:])) for key,val in enumerate(path_nodes)}

# set(zip(path_nodes[:-1], path_nodes[1:]))