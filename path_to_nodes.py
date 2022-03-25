#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 08:14:50 2020

@author: shub
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def path_to_nodes(which, node_a = None, node_b = None):
    if which == 37: # IEEE 37-node or IEEE 906-node
        from Network37 import arcs
    if which == 906 or which == 907: # IEEE 37-node or IEEE 906-node
        from Network906 import arcs
        which = 906 + 1 # becasue there are 907 buses, see nw 906 script

    # instantiate the graph
    G = nx.Graph()
    # add nodes
    G.add_nodes_from(np.arange(0, which))
    # add edges
    if which == 37 or which == 906 or which == 907:
        G.add_edges_from((arcs))
    else: # use node_a and node_b for edge addition
        e = zip(node_a, node_b)
        G.add_edges_from(e)
    # uncomment below to plot graph
    # plt.figure()
    # nx.draw(G, with_labels=True, font_weight='bold')

    # get the path to nodes from the slack bus i.e. node 0
    path_nodes = [nx.shortest_path(G, 0, i) for i in range(which)]
    # convert to dictinary with vals as set of tuples denoting the path to nodes
    path_to_all_nodes = {key:set(zip(val[:-1], val[1:])) for key,val in enumerate(path_nodes)}

    # list made here so that no troubles converting set to list later
    path_to_all_nodes_list = {key:list(zip(val[:-1], val[1:])) for key,val in enumerate(path_nodes)}
    return path_to_all_nodes, path_to_all_nodes_list

# set(zip(path_nodes[:-1], path_nodes[1:]))