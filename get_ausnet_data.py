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
# com_ground added here, required some tracebility to see, coming from network in evolve core tools object
full_nw = network_from_ejson("loaded_nw", ejson_nw)
print("loaded nw")
print(full_nw)

reduced_nw = full_nw.copy()

_, edges_removed = arbitrarily_remove_edges_to_remove_cycles(reduced_nw.graph, inplace=True)
set_full_graph_edge_direction(reduced_nw.graph, inplace=True)

# get the removed edges without 'com_ground'
# most of the cycles introduced by 'com_ground'
count_com = 0
removed_edges_without_com = []
for i, val in enumerate(edges_removed):
    if 'com_ground' in val:
        count_com+=1
    else:
        removed_edges_without_com.append(val)
# num_edges includes sum of of infeeders, loads, lines and  transformers
# num lines + transformers = num_nodes - 1

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
bus_all = [] # all buses with node in its name, ignores upstream & com_ground
for num in bus_sorted:
    if 'node' in num:
        bus_all.append(int(num.split("_")[1]))

BusNum = []
line_num = []

# get all the buses from the json file
# Busnum would be less than number_of_nodes() becasue it doesn't count the 
# upstream and com_ground nodes
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

# unordered arcs characteristics
R_line_unordered, X_line_unordered, LineData_Z_pu_unordered = {}, {}, {}
arcs = []
count_lines, count_transformers = 0, 0
for k, component in ejson_nw_updated['components'].items():

    if 'Line' in component:
        count_lines +=1
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
        count_transformers+=1
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

def find_unconnected_nodes(BusNum, arcs):
    # returns the lsit of unconnected nodes and the total number
    # see unconnected nodes
    unconnected_nodes = []
    unconnected_bus_count = 0
    for bus in BusNum:
        bus_found = 0
        for arc in arcs: # iterate over arcs
            if bus == arc[1]: # see if the bus is being connected with a previous node
                bus_found = 1
                break
        if bus_found == 0:
            unconnected_bus_count+=1
            unconnected_nodes.append(bus)
    return unconnected_nodes, unconnected_bus_count

# unconnected nodes pre processing
unconnected_nodes_pre, unconnected_bus_count_pre = find_unconnected_nodes(BusNum, arcs)
# the only node that should be unconnected from the above loop
# should be the slack node in an ideal world

###############################################################################
# see if you can connect unconnected nodes using the removed edges
###############################################################################
unconnected_nodes_in_removed_edges = [] # nodes that are still unconnected after searching for them in removed edges
for j in unconnected_nodes_pre:
    found = 0
    count_more_than_to = 0
    edges_feeding_to_node = []
    for i in removed_edges_without_com:
        if 'node_'+str(j) in i[1]: # see the line that is feeding to node j
            # print(j,i)
            found = 1 # when to for missing node is found
            count_more_than_to+=1
            first_node = int(i[0].split("_")[1])
            second_node = int(i[1].split("_")[1])
            edges_feeding_to_node.append((first_node, second_node))
            arcs.append((first_node, second_node)) # add it to the list of arcs
    if found == 0: # when to for missing node isn't found, it should only happen for slack bus
        print('omg! should be only once', j,i)
        unconnected_nodes_in_removed_edges.append(j)
    # add the removed edges back to the list of edges
    else:
        if count_more_than_to == 1: # when unconnected node has 1 feed in line
            bus_arcs[j]['To'] = edges_feeding_to_node
        if count_more_than_to > 1: # if the unconnected nodes have multiple to
            # you should only use 1 of them for radial distribution
            # but at this time we'll add all
            print('Multiple to for node:',j, i, edges_feeding_to_node)
            bus_arcs[j]['To'] = edges_feeding_to_node
        # the only other case is when the value is 0, not required to deal 
        # wtih this case. Has been taken care in a way when found == 0

# unconnected nodes post processing with updated arcs
unconnected_nodes_post, unconnected_bus_count_post = find_unconnected_nodes(BusNum, arcs)

if unconnected_bus_count_post != len(unconnected_nodes_in_removed_edges):
    print('Something aint working')
if unconnected_nodes_post != unconnected_nodes_in_removed_edges:
    print('Check again, should be the same nodes')

def check_for_multiple_sources_to_node():
    pass
###############################################################################
# check for cycles
# Remove double to, make sure the removed ones first coincide with the removed nodes
###############################################################################
count_to = 0 # nodes having multiple lines feeding them
for key, val in bus_arcs.items():
    if len(val["To"])>1:
        count_to+=1
        print(key, val)
        if len(val["To"])>2:
            print('Damn SON')
print(count_to)

slack_node = 8183
unconnected_nodes_post.remove(slack_node) # remove slack node from unconnected nodes
# nodes downstream of unconected nodes and are leaf nodes
# 2349 is downstream of one of the uncoonected nodes: 3212
# removing 3312 won't affect 2349 becasue it has another upstream node: 2703
downstream_leaf_nodes = [130, 7730, 8023]
unconnected_nodes_post.extend(downstream_leaf_nodes)

# Remove the nodes still unconnected
for i in unconnected_nodes_post:
    for j in bus_all:
        if i == j:
            bus_all.remove(i)
            bus_arcs.pop(i)# remove bus_arcs as well
 
# make sure bus_all is still ordered
bb = []
for num in bus_sorted:
    if 'node' in num:
        bb.append(int(num.split("_")[1]))

indices_of_removed_nodes = []
for i in unconnected_nodes_post:
    for j, val in enumerate(bb):
        if i == val:
            indices_of_removed_nodes.append(j)
indices_of_removed_nodes = np.sort(np.asarray(indices_of_removed_nodes))
# they look in order, checked

# Remove arcs corresponding to unconnectred nodes
for i in unconnected_nodes_post:
    # print(i)
    for j in arcs:
        if i in j:
            print(i, j)
            arcs.remove(j)

# also remove the nodes downstream to the unconnceted nodes
# ceheck if the number of removed nodes matches the previously existing number
if len(BusNum) != len(bus_all) + len(unconnected_nodes_post):
    print('Mate, check it again')

# might have to create new to and from with updated arcs
arcs_all = [] # recreate ordered arcs
bus_arcs = {} # recreate ordered with busnodes

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


###############################################################################
# check again for cycles
# see if removing nodes reduced the number of nodes with double feeders
# Remove double to, make sure the removed ones first coincide with the removed nodes
###############################################################################
count_to = 0 # nodes having multiple lines feeding them
nodes_with_multiple_source, edges_removed = [], []
for key, val in bus_arcs.items():
    if len(val["To"])>1: # multiple source to a node
        count_to+=1
        nodes_with_multiple_source.append(key)
        print(key, val)
        if len(val["To"])>2:
            print('Damn SON')
        else: # when len is 2
            edge_removed = bus_arcs[key]["To"].pop(0) # remove the first edge
            edges_removed.append(edge_removed)

# remove it from arcs
for edge in edges_removed:
    for arc in arcs_all:
        if edge == arc:
            print(edge, arc)
            arcs_all.remove(arc)
            break

# final check  for multiple source to node
count_to = 0 # nodes having multiple lines feeding them
for key, val in bus_arcs.items():
    if len(val["To"])>1: # multiple source to a node
        count_to+=1
        nodes_with_multiple_source.append(key)
        print(key, val)

# think how you are gonna 'to'

# num_edges = num_nodes -1
# and then see what else can be removed


# make sure to get the order correct
# runt tests again to see if the final network is connected and usable
# total_edges = total_nodes - 1
# total_edges = num_lines + num_transformers


# make sure there is a path to every node from slack bus
# and there should be only way from slack node to any node in the system
# i think you have a function path_to_node, could use that
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