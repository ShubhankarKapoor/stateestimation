#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:25:13 2022

@author: shub
"""

# chose 2 cases for comparison
# use the following for plotting
# maybe pick paper and pen to decide what 2 cases you want to do
import networkx as nx  
s = 5
G = nx.grid_graph(dim=[s,s])
nodes = list(G.nodes)
edges = list(G.edges)
p = []
for i in range(0, s):
    for j in range(0, s):
        p.append([i,j])
for i in range(0, len(nodes)):
    G.nodes[nodes[i]]['pos'] = p[i]

pos = {}
for i in range(0, len(nodes)):
        pos[nodes[i]] = p[i]

from random import randint
val = []
for i in range(0, len(G.nodes())):
    val.append(randint(0,4))

nx.draw(G, pos, node_color=val)

# use the following for distribution of errors
# plot v perc hist for all methods
bins=[0, 1, 2.5,6.5]
# data = np.asarray(ll_la_perc_v)[np.where(np.asarray(ll_la_perc_v)<3)[0]]

hist, bin_edges = np.histogram(ll_both_feed_perc_v,bins) # make the histogram
fig,ax = plt.subplots()
ax.bar(range(len(hist)),hist,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

# plot p abs hist for all methods