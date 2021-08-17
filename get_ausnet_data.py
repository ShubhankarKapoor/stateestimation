from evolve_core_tools.parser import network_from_ejson
from evolve_core_tools.parser import (network_to_ejson, network_from_ejson, 
    measurements_from_ejson,  measurements_to_ejson, graph_to_ejson)
from evolve_core_tools.network_graphs.processing import (
    set_full_graph_edge_direction,
    arbitrarily_remove_edges_to_remove_cycles,)
import json
import numpy as np
import networkx as nx


NETWORK_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_network.json"
MEASUREMENT_SAMPLE_EJSON = "/home/shub/Documents/phd/distflow/json_files/ausnet_measurements.json"
NETWORK_UPDATED_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_network_updated.json"

with open(NETWORK_SAMPLE_EJSON) as f:
    ejson_nw = json.load(f)

with open(MEASUREMENT_SAMPLE_EJSON) as f:
    ejson_meas = json.load(f)

# Example of using 'network_from_ejson' function (ejson_parser.py), that returns a new Network object
full_nw = network_from_ejson("loaded_nw", ejson_nw)
print("loaded nw")
print(full_nw)

reduced_nw = full_nw.copy()

arbitrarily_remove_edges_to_remove_cycles(reduced_nw.graph, inplace=True)
set_full_graph_edge_direction(reduced_nw.graph, inplace=True)

print("removed edges and set directions")

print(f"full_nw graph has {full_nw.graph.number_of_nodes()} nodes "
      f"and {full_nw.graph.number_of_edges()} edges.")
print(f"Reduced_nw graph has {reduced_nw.graph.number_of_nodes()} nodes "
f"and {reduced_nw.graph.number_of_edges()} edges")

reduced_json = graph_to_ejson(reduced_nw.graph, to_json=False)

print("parced reduced graph to ejson")
restored_nw = network_from_ejson('restored',reduced_json)
print(f"restored_nw graph has {restored_nw.graph.number_of_nodes()} nodes "
f"and {restored_nw.graph.number_of_edges()} edges")

# convert xy and other array values to list before saving json file
for key, component in reduced_json["components"].items():
    if 'node' in key:
        reduced_json["components"][key]["Node"]["xy"] = reduced_json["components"][key]["Node"]["xy"].tolist()
        # break
    
    if 'Line' in component:
        component_dct = component['Line']
        component_dct['z'] = component_dct['z'].tolist()
        component_dct['z0'] = component_dct['z0'].tolist()
    
    if 'Transformer' in component:

        component_dct = component['Transformer']
        component_dct["v_winding_base"] = component_dct["v_winding_base"].tolist()
        component_dct['z'] = component_dct['z'].tolist()
        component_dct['z0'] = component_dct['z0'].tolist()        
        # break
        #     component_dct.pop('imped_mod')
    if 'Load' in component:
        component_dct = component['Load']
        component_dct["s_nom"] = component_dct["s_nom"].tolist()
        # break
    
# write it to a json file
with open(NETWORK_UPDATED_SAMPLE_EJSON, 'w') as fp:
    json.dump(reduced_json, fp, indent=2)

# check the newly updated json file
with open(NETWORK_UPDATED_SAMPLE_EJSON) as f:
    ejson_nw_updated = json.load(f)
    
# Example of using 'network_from_ejson' function (ejson_parser.py), that returns a new Network object
updated_nw = network_from_ejson("loaded_nw_updated", ejson_nw_updated)
print("updated loaded nw")
print(updated_nw)

print(f"updated_nw graph has {updated_nw.graph.number_of_nodes()} nodes "
f"and {updated_nw.graph.number_of_edges()} edges")


# # nx.find_cycle(full_nw.graph)
# result =arbitrarily_remove_edges_to_remove_cycles(full_nw.graph, inplace=True)
# xx = set_full_graph_edge_direction(full_nw.graph, inplace=True)

# arbitrarily_remove_edges_to_remove_cycles(full_nw.graph, inplace=True)
# set_full_graph_edge_direction(full_nw.graph, inplace=True)

# nw_json = network_to_ejson(full_nw)

# network_to_ejson(xx)
# print("Number of edges", xx.number_of_edges())

'''
measurements_from_ejson(ejson_meas, updated_nw)
print("Loaded measurement data")

# test node
nodes = ['node_9']
for node_id in nodes:
    meas = updated_nw.nodes[node_id].meas
    if updated_nw.nodes[node_id].meas:
        print(f"{len(meas)} meters are associated with Node {node_id}")
        meas_df = next(iter(meas.values())).data
        print(meas_df.head(5))
'''
# topologically ordered nodes
# need it for forward backward calculations of power flow
bus_sorted = list(nx.topological_sort(reduced_nw.graph))
bus_all = []
for num in bus_sorted:
    if 'node' in num:
        bus_all.append(int(num.split("_")[1]))

BusNum = []
line_num = []

for k, component in ejson_nw_updated['components'].items():
    
    # if 'Node' in component:
    if 'node' in k:
        # print(k)
        # get the node numbers from the string node name
        BusNum.append(int(k.split("_")[1]))
        # print(k, component)
        # break

# check if the number of nodes from ejson file and topologically sorted graph
# are same
if len(BusNum) != len(bus_all):
    print('something fishy!!!!!')

R_line_unordered, X_line_unordered, LineData_Z_pu_unordered = {}, {}, {}
arcs = []
count, count2 = 0, 0
for k, component in ejson_nw_updated['components'].items():

    if 'Line' in component:
        count +=1
        # print(k)
        # break
        # get the line number from the string line name
        # line_num.append(int(k.split("_")[1]))        

        # there is no line_1
        component_dct = component['Line']

        if 'cons' in component_dct:
            # print('yo', component_dct["cons"][0]["node"], component_dct["cons"][1]["node"])
            first_node = int(component_dct["cons"][0]["node"].split("_")[1])
            second_node = int(component_dct["cons"][1]["node"].split("_")[1])
            # make sure arcs are ordered
            arcs.append((first_node, second_node))
            # get R_line_unordered, X_line_unordered and imp
            R, X = component_dct['z'][0], component_dct['z'][1]
            R_line_unordered[(first_node, second_node)] = R
            X_line_unordered[(first_node, second_node)] = X
            # component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']
            LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j

    if 'Transformer' in component:
        count2+=1
        component_dct = component['Transformer']
        if 'cons' in component_dct:
            first_node = int(component_dct["cons"][0]["node"].split("_")[1])
            second_node = int(component_dct["cons"][1]["node"].split("_")[1])
            arcs.append((first_node, second_node))        
        
            # component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']

            # need to see how to model transformers as line
            # not sure
            # reason to do beacsue they behave as edges in this system
            # alternatively can remove all transformers and related nodes
            # below isn't correct impedance, it s just to run the code for sorted arcs
            
            R, X = component_dct['z'][1][0], component_dct['z'][1][1]
            R_line_unordered[(first_node, second_node)] = R
            X_line_unordered[(first_node, second_node)] = X
            # component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']
            LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j

        #     break
        # break

##################################################
# bus_arcs
##################################################

# bus_arcs[0] = {"To":[],"from":[(0,1)]}

# add to and then from for bus nodes in list of ordered arcs
# avoid repitition
arcs_all = [] # ordered arcs
bus_arcs = {} # ordered with busnodes

for i in bus_all: # use ordered bus
    t = []
    f = []

    for ii in arcs:
        if i == ii[0]:
            f.append(ii)
            if ii not in arcs_all:
                arcs_all.append(ii)
        if i == ii[1]:
            t.append(ii)
            if ii not in arcs_all:
                arcs_all.append(ii)            
        # break
    bus_arcs[i] = {"To":t,"from":f}
    # break
# bus_arcs[1]["To"] = [(0,1)]

# ordered arcs characteristics
R_line, X_line, LineData_Z_pu = {}, {}, {}

for arc in arcs_all:
    R_line[arc] = R_line_unordered[arc]
    X_line[arc] = X_line_unordered[arc]
    LineData_Z_pu[arc] = LineData_Z_pu_unordered[arc]

###############################################################################
# check for cycles
###############################################################################
for key, val in bus_arcs.items():
    if len(val["To"])>1:
        print(key, val)

# see unconnected nodes
unconnected_nodes = []
uconnected_bus_count = 0
for bus in BusNum:
    bus_found = 0
    for arc in arcs:
        if bus == arc[1]:
            bus_found = 1
            break
    if bus_found == 0:
        uconnected_bus_count+=1
        unconnected_nodes.append(bus)

###############################################################################
# check if there is gap between line numbers
###############################################################################
for i in range(len(line_num)-1):
    if abs(line_num[i] - line_num[i+1])!=1:
        # print(line_num[i] - line_num[i+1],line_num[i], line_num[i+1])
        pass
    
# nw = network_from_ejson("loaded_nw", ejson_nw)

# ##################################################
# # arcs
# ##################################################
# arcs = []
# for i in range(0,busNo-1):
#     arcs.append(bus_arcs[i+1]["To"][0])