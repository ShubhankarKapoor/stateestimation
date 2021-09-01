#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 13:43:33 2021

@author: shub
"""

from power_flows.dist_flow_pyomo import dist_flow_pyomo, dist_flow_lossy
from power_flows.dist_flow_lossless import dist_flow_lossless

# import pyximport
# pyximport.install()
# from power_flows.sgt_wrapper.sgt_wrapper import solve_power_flow_sgt, solve_network_sgt
# from power_flows.sgt_wrapper.test import foo
# from power_flows.power_flow_sgt import power_flow_sgt

from evolve_core_tools.evolve_core_tools.parser import (network_to_ejson, network_from_ejson, 
    measurements_from_ejson,  measurements_to_ejson, graph_to_ejson)
from evolve_core_tools.evolve_core_tools.network_graphs.processing import (
    set_full_graph_edge_direction,
    arbitrarily_remove_edges_to_remove_cycles,)
import json
import networkx as nx

# load network files and measurement files
NETWORK_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_removed2.json" # new file
MEASUREMENT_SAMPLE_EJSON = "/home/shub/Documents/phd/distflow/json_files/ausnet_measurements.json"
# NETWORK_UPDATED_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_network_updated.json"

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

# assumed here no cycles
# run network checks
# validate_nw_using_json_file(ejson_nw_updated, slack_node=8183) # no cycles
# validate_nw_using_json_file_to_network(ejson_nw_updated, slack_node=8183) # cycles inteoduced because of introductioin of com_ground while converting it to nw

# topologically ordered nodes
# need it for forward backward calculations of power flow
bus_sorted = list(nx.topological_sort(full_nw.graph))
BusNum = [] # all buses with node in its name, ignores upstream & com_ground
for num in bus_sorted:
    if 'node' in num:
        BusNum.append(int(num.split("_")[1]))
        
# get loads
# P_Load, Q_Load, _, _ = get_load_meas_from_json(ejson_meas, full_nw, BusNum, timestamp=None)

# run different powerflows
slack_bus_node = 'node_8183'

# run distflow lossy
dist_flow_lossy(full_nw, slack_bus_node, slack_voltage = 1, write_solver_output=True)

# distflow lossless
bus_v = dist_flow_lossless(full_nw, root_node=slack_bus_node, v_root=1, calculate_junction_nodes=False, calculate_all_nodes=True)

# # SGT
# power_flow_sgt(ejson_nw, ejson_meas)