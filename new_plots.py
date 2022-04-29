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
import numpy as np
import matplotlib.pyplot as plt
import pickle

SMALL_SIZE = 8
MEDIUM_SIZE = 15
BIGGER_SIZE = 22

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

# s = 5
# G = nx.grid_graph(dim=[s,s])
# nodes = list(G.nodes)
# edges = list(G.edges)
# p = []
# for i in range(0, s):
#     for j in range(0, s):
#         p.append([i,j])
# for i in range(0, len(nodes)):
#     G.nodes[nodes[i]]['pos'] = p[i]

# pos = {}
# for i in range(0, len(nodes)):
#         pos[nodes[i]] = p[i]

# from random import randint
# val = []
# for i in range(0, len(G.nodes())):
#     val.append(randint(0,4))

# nx.draw(G, pos, node_color=val)

if which == 37: # IEEE 37-node or IEEE 906-node
    from Network37 import arcs
if which == 906 or which == 907: # IEEE 37-node or IEEE 906-node
    from Network906 import arcs
    which = 906 + 1 # becasue there are 907 buses, see nw 906 script

val = abs_p_la*Sbase
# val = perc_v_la

# if vmin == None and np.all(val)!= None:
vmin = np.min(val)
# if vmin == None and np.all(val)!= None:
vmax = np.max(val)
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

# plotting heatmap on nodes
# just do it using proposed methodology for diff cases of meas available
# do it for voltage percentage and abs power
plt.figure()
with open('saved_pos.pkl', 'rb') as f:
    pos = pickle.load(f) # load the position of nodes for plotting
    pos_higher = {} # used to put label outside the nodes
    y_off = 0.05  # offset on the y axis
    
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)

if np.all(val) == None:
    # nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    nx.draw(G, pos=pos, with_labels=False, font_weight='bold')
    nx.draw_networkx_labels(G, pos_higher)
    plt.title('Test Feeder {}'.format(which))
else:
    # define vmin,vmax
    # plt.title('Voltage Percentage Error at each node in Test Feeder {}'.format(which))
    plt.title('')
    # cmap = plt.cm.Blues
    cmap = plt.cm.viridis
    # nx.draw(G, pos=pos, with_labels=True, font_weight='bold', node_color=val)
    nx.draw(G, pos=pos, vmin = vmin, vmax = vmax, with_labels=False, 
            font_weight='bold', node_color=val, cmap=cmap)
    nx.draw_networkx_labels(G, pos_higher)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin = vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)

# use the following for distribution of errors
# plot v perc hist for all methods
bins=[0, 1, 2.5,6.5] # 37
bins=[0, 0.5, 1,3] # 906
# data = np.asarray(ll_la_perc_v)[np.where(np.asarray(ll_la_perc_v)<3)[0]]

# hist, bin_edges = np.histogram(ll_both_feed_perc_v,bins) # make the histogram
# fig,ax = plt.subplots()
# ax.bar(range(len(hist)),hist,width=1,align='center',tick_label=
#         ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

fig = plt.figure()
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
hist1, bin_edges = np.histogram(ll_no_feed_perc_v,bins) # make the histogram
ax1.bar(range(len(hist1)),hist1,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist1)])
ax1.title.set_text('LN')

ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
hist2, bin_edges = np.histogram(ll_v_feed_perc_v,bins) # make the histogram
ax2.bar(range(len(hist2)),hist2,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist2)])
ax2.title.set_text('LV')

ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
hist3, bin_edges = np.histogram(ll_p_feed_perc_v,bins) # make the histogram
ax3.bar(range(len(hist3)),hist3,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist3)])
ax3.title.set_text('LP')

ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
hist4, bin_edges = np.histogram(ll_both_feed_perc_v,bins) # make the histogram
ax4.bar(range(len(hist4)),hist4,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist4)])
ax4.title.set_text('LB')

ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
hist5, bin_edges = np.histogram(ll_la_perc_v,bins) # make the histogram
ax5.bar(range(len(hist5)),hist5,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist5)])
ax5.title.set_text('LA')
# fig.subplots_adjust(wspace=0.8)
# fig.suptitle('Distribution of Voltage % Error in Test Feeder {}'.format(which))
fig.set_tight_layout(True)

# plot p abs hist for all methods
bins=np.asarray([0, 0.1, 1, 4])*Sbase # 37
bins=np.asarray([0, 2.5, 5, 10]) # 906
# data = np.asarray(ll_la_perc_v)[np.where(np.sasarray(ll_la_perc_v)<3)[0]]

# hist, bin_edges = np.histogram(ll_both_feed_ab,bins) # make the histogram
# fig,ax = plt.subplots()
# ax.bar(range(len(hist)),hist,width=1,align='center',tick_label=
#         ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist)])

fig = plt.figure()
ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
hist1, bin_edges = np.histogram(np.asarray(ll_no_feed_abs_p)*Sbase,bins) # make the histogram
ax1.bar(range(len(hist1)),hist1,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist1)])
plt.xticks( rotation=45)
ax1.title.set_text('LN')

ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
hist2, bin_edges = np.histogram(np.asarray(ll_v_feed_abs_p)*Sbase,bins) # make the histogram
ax2.bar(range(len(hist2)),hist2,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist2)])
plt.xticks( rotation=45)
ax2.title.set_text('LV')

ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
hist3, bin_edges = np.histogram(np.asarray(ll_p_feed_abs_p)*Sbase,bins) # make the histogram
ax3.bar(range(len(hist3)),hist3,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist3)])
plt.xticks( rotation=45)
ax3.title.set_text('LP')

ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
hist4, bin_edges = np.histogram(np.asarray(ll_both_feed_abs_p)*Sbase,bins) # make the histogram
ax4.bar(range(len(hist4)),hist4,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist4)])
plt.xticks( rotation=45)
ax4.title.set_text('LB')

ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)
hist5, bin_edges = np.histogram(np.asarray(ll_la_abs_p)*Sbase,bins) # make the histogram
ax5.bar(range(len(hist5)),hist5,width=1,align='center',tick_label=
        ['{} - {}'.format(bins[i],bins[i+1]) for i,j in enumerate(hist5)])
plt.xticks( rotation=45)
ax5.title.set_text('LA')
# fig.subplots_adjust(wspace=0.8)
# fig.suptitle('Distribution of Active Power Absolute Error in Test Feeder {}'.format(which))
fig.set_tight_layout(True)
