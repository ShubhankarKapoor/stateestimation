from evolve_core_tools.evolve_core_tools.parser import (network_to_ejson, network_from_ejson, 
    measurements_from_ejson,  measurements_to_ejson, graph_to_ejson)
from evolve_core_tools.evolve_core_tools.network_graphs.processing import (
    set_full_graph_edge_direction,
    arbitrarily_remove_edges_to_remove_cycles,)
import json
import numpy as np
import networkx as nx
import pandas as pd
from ausnet_parser import *

NETWORK_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_network.json"
MEASUREMENT_SAMPLE_EJSON = "/home/shub/Documents/phd/distflow/json_files/ausnet_measurements.json"
NETWORK_UPDATED_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_network_updated.json"

with open(NETWORK_SAMPLE_EJSON) as f:
    ejson_nw = json.load(f)

with open(MEASUREMENT_SAMPLE_EJSON) as f:
    ejson_meas = json.load(f)

# Example of using 'network_from_ejson' function (ejson_parser.py), that returns a new Network object
# com_ground added here, required some tracebility to see, coming from network in evolve core tools object
full_nw = network_from_ejson("loaded_nw", ejson_nw)
print("loaded nw")
print(full_nw)

# full_nw.graph.remove_node('com_ground')
# full_nw.graph.remove_node('upstream')

print(f"full_nw graph has {full_nw.graph.number_of_nodes()} nodes "
      f"and {full_nw.graph.number_of_edges()} edges.")

# get radial nw in ejson form
ejson_nw_updated = make_the_nw_radial(ejson_nw, slack_node = 8183)
# manually removed component 10423 which has node 5594 on its load

# Example of using 'network_from_ejson' function (ejson_parser.py), that returns a new Network object
updated_nw = network_from_ejson("loaded_nw_updated", ejson_nw_updated)
print("updated loaded nw")
print(updated_nw)

print(f"updated_nw graph has {updated_nw.graph.number_of_nodes()} nodes "
f"and {updated_nw.graph.number_of_edges()} edges")

# get network characterisitcs
R_line_unordered, X_line_unordered, LineData_Z_pu_unordered, arcs_all, \
transformer_edges, turns_ratio, count_lines, count_transformers, BusNum, = \
get_arcs_and_nw_info(ejson_nw_updated, updated_nw)

# get ordered arcs and bus arcs
arcs, bus_arcs = get_ordered_arcs(BusNum, arcs_all)

# get load values
# make sure 10423 which has node 5594 on its load is removed
# else will get a key error
# or a hack to load meas
BusNum.remove(5594)
P_Load, Q_Load, _, _ = get_load_meas_from_json(ejson_meas, updated_nw, BusNum, timestamp=None)

# validat the network
validate_nw_using_arcs(arcs_all, slack_node=8183) # cycles should be present
validate_nw_using_arcs(arcs, slack_node=8183) # no cycles 
validate_nw_using_json_file(ejson_nw_updated, slack_node=8183) # no cycles
validate_nw_using_json_file_to_network(ejson_nw_updated, slack_node=8183) # cycles inteoduced because of introductioin of com_ground while converting it to nw
