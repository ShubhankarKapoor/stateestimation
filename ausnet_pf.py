#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:43:33 2021

@author: shub
"""
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from power_flows.dist_flow_pyomo import dist_flow_pyomo, dist_flow_lossy
from power_flows.dist_flow_lossless import dist_flow_lossless
from ausnet_parser import *
from evolve_core_tools.evolve_core_tools.parser import (network_to_ejson, network_from_ejson, 
    measurements_from_ejson,  measurements_to_ejson, graph_to_ejson)
from evolve_core_tools.evolve_core_tools.network_graphs.processing import (
    set_full_graph_edge_direction,
    arbitrarily_remove_edges_to_remove_cycles,)

import json
import networkx as nx

# import pyximport
# pyximport.install()
# from power_flows.sgt_wrapper.sgt_wrapper import solve_power_flow_sgt, solve_network_sgt
# from power_flows.sgt_wrapper.test import foo
# from power_flows.power_flow_sgt import power_flow_sgt

# load network files and measurement files
NETWORK_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/tx_20_fdr.json" # new file
MEASUREMENT_SAMPLE_EJSON = "/home/shub/Documents/phd/distflow/json_files/ausnet_measurements.json"

with open(NETWORK_SAMPLE_EJSON) as f:
    ejson_nw = json.load(f)

with open(MEASUREMENT_SAMPLE_EJSON) as f:
    ejson_meas = json.load(f)

full_nw = network_from_ejson("loaded_nw", ejson_nw)
print("loaded nw")
print(full_nw)

# remove the nodes 'com_ground' and 'upstream' that were added after network_from_ejson
full_nw.graph.remove_node('com_ground')
full_nw.graph.remove_node('upstream')

# get network characterisitcs
R_line_unordered, X_line_unordered, LineData_Z_pu_unordered, arcs_all, \
transformer_edges, turns_ratio, count_lines, count_transformers, BusNum, = \
get_arcs_and_nw_info(ejson_nw, full_nw)

# get ordered arcs and bus arcs
arcs, bus_arcs = get_ordered_arcs(BusNum, arcs_all)

# assumed here no cycles
# run network checks
slack_node = BusNum[0] # 8183
# validate_nw_using_arcs(arcs_all, slack_node=slack_node) # cycles should be present
# validate_nw_using_arcs(arcs, slack_node=slack_node) # no cycles 
# validate_nw_using_json_file(ejson_nw, slack_node=slack_node) # no cycles
# validate_nw_using_json_file_to_network(ejson_nw, slack_node=slack_node) # cycles inteoduced because of introductioin of com_ground while converting it to nw

# get line characteristics
R_line, X_line, LineData_Z_pu = get_ordered_arcs_characterisitcs(arcs, 
                R_line_unordered, X_line_unordered, LineData_Z_pu_unordered)

# BusNum.remove(5594)
# get loads
P_Load, Q_Load, non_zib_nodes, zib_nodes, Vmag_true = get_load_meas_from_json(
    ejson_meas, full_nw, BusNum, timestamp=None)
# get voltage measurement as well
Vbase = 1 # voltage bus of slack bus
Sbase = 1000 #kVA Base apparent power for normalization

# run different powerflows from evolve
# slack_bus_node = 'node_8183'






'''
# getting sub network between 2 transformer edges
first_transformer_node_found = 0
transformer_nodes_in_order, idx_of_transformer_nodes = [], []
for bus in BusNum:
    for edge in transformer_edges:
        if bus == edge[0]:# should be the first node in the edge
            first_transformer_node_found+= 1
            print(bus, edge)
            transformer_nodes_in_order.append(bus)
            idx_of_transformer = BusNum.index(bus)
            idx_of_transformer_nodes.append(idx_of_transformer)
            break
    if first_transformer_node_found==3:
        break

# buses considered
first_node = idx_of_transformer_nodes[0]+1 # start from the second node of first transformer edge
second_node = idx_of_transformer_nodes[1]+1 # end at the first node of second transformer edge 
BusNummm = BusNum[first_node:second_node]

# network params of the updated nw
arcs2, bus_arcs2 = get_ordered_arcs(BusNummm, arcs_all)
R_line2, X_line2, LineData_Z_pu2 = get_ordered_arcs_characterisitcs(arcs, R_line_unordered, X_line_unordered, 
                                     LineData_Z_pu_unordered)

validate_nw_using_arcs(arcs2, slack_node=BusNummm[0]) # no cycles

slack_node = BusNummm[0]
for j in bus_arcs2:
    if len (bus_arcs2[j]["To"]) == 0 and j!=slack_node: # if the node doesnt have an incoming line
        last_node_index = BusNummm.index(j)
        print(bus_arcs2[j])
        break

BusNum = BusNummm[:last_node_index]
'''





'''
# removing these manually after running distflow, nodes with high errors
# BusNum = BusNum[105:]
BusNum = BusNum[0:75]
arcs, bus_arcs = get_ordered_arcs(BusNum, arcs_all)
R_line, X_line, LineData_Z_pu = get_ordered_arcs_characterisitcs(arcs, R_line_unordered, X_line_unordered, 
                                      LineData_Z_pu_unordered)

validate_nw_using_arcs(arcs, slack_node=BusNum[0]) # no cycles

slack_node = BusNum[0]
restructure_bus = 0
for j in bus_arcs:
    if len (bus_arcs[j]["To"]) == 0 and j!=slack_node: # if the node doesnt have an incoming line
        last_node_index = BusNum.index(j)
        print(bus_arcs[j])
        restructure_bus = 1
        break
if restructure_bus == 1: # this will help avoid using any prev last_node_index val
    BusNum = BusNummm[:last_node_index]
'''




'''
# network params of the updated nw
arcs, bus_arcs = get_ordered_arcs(BusNum, arcs_all)
R_line, X_line, LineData_Z_pu = get_ordered_arcs_characterisitcs(arcs, R_line_unordered, X_line_unordered, 
                                      LineData_Z_pu_unordered)

slack_node = BusNum[0]
for j in bus_arcs:
    if len (bus_arcs[j]["To"]) == 0 and j!=slack_node:
        last_node_index = BusNum.index(j)
        print(bus_arcs[j])
        
# check if the nw is good
validate_nw_using_arcs(arcs, slack_node=BusNum[0]) # no cycles
# consider the nodes before the last node index

# get new meas set on the updated bus
P_Load, Q_Load, non_zib_nodes, zib_nodes, Vmag_true = get_load_meas_from_json(
    ejson_meas, full_nw, BusNum, timestamp=None)

# check if bus belong to transformer edges
# slack node should be the second node of first transformer edge
# the last node can/ cannot be the first node of second transformer edge
for i in BusNum:
    for edge in transformer_edges:
        if i in edge:
            print(i, edge)
# run lindistflow
# LinDistFlowBackwardForwardSweep(P_Load,Q_Load, which=None, loss=1, pflow = 1, max_iter= 5)

# make a graph
G = nx.MultiDiGraph()
# add nodes
G.add_nodes_from(BusNum)
# add edges
G.add_edges_from((arcs))

# get nodes downstream of every node
node_of_interest = BusNum[0]
downstream = [n for n in nx.traversal.bfs_tree(G, node_of_interest) if n != node_of_interest]
'''
# use the above downstream nodes to check powerflow


# bus_arcs[i]["from"].remove((5978, 2247)) # coz 2247 isnt a part of the nw
# run distflow lossy
# dist_flow_lossy(full_nw, slack_bus_node, slack_voltage = 1, write_solver_output=True)

# distflow lossless
# bus_v = dist_flow_lossless(full_nw, root_node=slack_bus_node, v_root=1, calculate_junction_nodes=False, calculate_all_nodes=True)

# SGT
# power_flow_sgt(ejson_nw, ejson_meas)
