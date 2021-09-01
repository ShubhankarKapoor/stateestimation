from evolve_core_tools.parser import network_from_ejson
from evolve_core_tools.parser import (network_to_ejson, network_from_ejson, 
    measurements_from_ejson,  measurements_to_ejson, graph_to_ejson)
from evolve_core_tools.network_graphs.processing import (
    set_full_graph_edge_direction,
    arbitrarily_remove_edges_to_remove_cycles,)
import json
import numpy as np
import networkx as nx
import pandas as pd

NETWORK_SAMPLE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_removed2.json"
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
# it should be: num lines + transformers = num_nodes - 1

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

# check length before and after removing edges
if len(ejson_nw['components'])-len(ejson_nw_updated['components']) == len(edges_removed):
    print('Number of edges removed match!')
else:
    print('check the edges removed, not looking good....!')

# all of this below could go in a function in different file
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# topologically ordered nodes
# need it for forward backward calculations of power flow
bus_sorted = list(nx.topological_sort(updated_nw.graph))
BusNum = [] # all buses with node in its name, ignores upstream & com_ground
for num in bus_sorted:
    if 'node' in num:
        BusNum.append(int(num.split("_")[1]))

bus_all = []
line_num = []

# get all the buses from the json file
# bus_all would be less than number_of_nodes() becasue it doesn't count the 
# upstream and com_ground nodes
for k, component in ejson_nw_updated['components'].items():
    # if 'Node' in component:
    if 'node' in k:
        # print(k)
        # get the node numbers from the string node name
        bus_all.append(int(k.split("_")[1]))

# check if the number of nodes from ejson file and topologically sorted graph
# are same
if len(bus_all) != len(BusNum):
    print('something fishy!!!!!')

# unordered arcs characteristics
R_line_unordered, X_line_unordered, LineData_Z_pu_unordered = {}, {}, {}
turns_ratio = {} # for transformers
arcs_all, transformer_edges = [], []
count_lines, count_transformers = 0, 0
for k, component in ejson_nw_updated['components'].items():

    if 'Line' in component:
        count_lines +=1
        component_dct = component['Line']

        if 'cons' in component_dct:
            first_node = int(component_dct["cons"][0]["node"].split("_")[1])
            second_node = int(component_dct["cons"][1]["node"].split("_")[1])
            arcs_all.append((first_node, second_node))
            # get R_line_unordered, X_line_unordered and imp
            length = component_dct['length']
            R, X = component_dct['z'][0] * length, component_dct['z'][1] * length
            R_line_unordered[(first_node, second_node)] = R
            X_line_unordered[(first_node, second_node)] = X
            LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j

    if 'Transformer' in component:
        count_transformers+=1
        component_dct = component['Transformer']
        if 'cons' in component_dct:
            first_node = int(component_dct["cons"][0]["node"].split("_")[1])
            second_node = int(component_dct["cons"][1]["node"].split("_")[1])
            arcs_all.append((first_node, second_node))        
            transformer_edges.append((first_node, second_node))
            # get turns ratio for transformers
            turns_ratio[(first_node, second_node)] = component_dct['nom_turns_ratio']
            # need to see how to model transformers as line
            # not sure
            # reason to do beacsue they behave as edges in this system
            # below isn't correct impedance, it s just to run the code for sorted arcs
            R, X = component_dct['z'][1][0], component_dct['z'][1][1]
            R_line_unordered[(first_node, second_node)] = R
            X_line_unordered[(first_node, second_node)] = X
            # component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']
            LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j

total_edges_before_proc = count_lines + count_transformers

##################################################
# bus_arcs
##################################################

# bus_arcs[0] = {"To":[],"from":[(0,1)]}
# add to and then from for bus nodes in list of ordered arcs
arcs = [] # ordered arcs
bus_arcs = {} # ordered with busnodes

for i in BusNum: # use ordered bus
    t = []
    f = []

    for ii in arcs_all:
        if i == ii[0]:
            f.append(ii)
            if ii not in arcs:
                arcs.append(ii)
        if i == ii[1]:
            t.append(ii)
            if ii not in arcs:
                arcs.append(ii)            
    bus_arcs[i] = {"To":t,"from":f}
# bus_arcs[1]["To"] = [(0,1)]

def find_unconnected_nodes(bus_all, arcs_all):
    # returns the list and number of unconnected nodes
    # see unconnected nodes
    unconnected_nodes = []
    unconnected_bus_count = 0
    for bus in bus_all:
        bus_found = 0
        for arc in arcs_all: # iterate over arcs
            if bus == arc[1]: # see if the bus is being connected with a previous node
                bus_found = 1
                break
        if bus_found == 0:
            unconnected_bus_count+=1
            unconnected_nodes.append(bus)
    return unconnected_nodes, unconnected_bus_count

# unconnected nodes pre processing
unconnected_nodes_pre, unconnected_bus_count_pre = find_unconnected_nodes(bus_all, arcs_all)
# the only node that should be unconnected from the above loop
# should be the slack node in an ideal situation

###############################################################################
# see if you can connect unconnected nodes using the removed edges
###############################################################################
unconnected_nodes_after_using_removed_edges = [] # nodes that are still unconnected after searching for them in removed edges
newly_added_arcs = [] # list of new arcs being added to the existing arcs
for j in unconnected_nodes_pre:
    found = 0
    count_more_than_to = 0
    edges_feeding_to_node = []
    for i in removed_edges_without_com:
        if 'node_'+str(j) in i[1]: # see the line that is feeding to node j
            found = 1 # when the feed in line to node is found
            count_more_than_to+=1
            first_node = int(i[0].split("_")[1])
            second_node = int(i[1].split("_")[1])
            edges_feeding_to_node.append((first_node, second_node))
            arcs_all.append((first_node, second_node)) # add it to the list of arcs
            newly_added_arcs.append((first_node, second_node))
    if found == 0: # when to for missing node isn't found, it should only happen for slack bus
        print('omg! should be only once', j,i)
        unconnected_nodes_after_using_removed_edges.append(j)
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

# add newly added arcs to the json file -->> later save as ausnet_removed.json
# need to get line characterisitcs for the edges added from original ejson
pre_addn_length = len(ejson_nw_updated['components'])
for arc in newly_added_arcs:
    for k, component in ejson_nw['components'].items():

        if 'Line' in component:
            component_dct = component['Line']

            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                if first_node == arc[0] and second_node == arc[1]:
                    count_lines +=1    
                    # get R_line_unordered, X_line_unordered and imp
                    length = component_dct['length']
                    R, X = component_dct['z'][0] * length, component_dct['z'][1] * length                    
                    R_line_unordered[(first_node, second_node)] = R
                    X_line_unordered[(first_node, second_node)] = X
                    LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j
                    ejson_nw_updated['components'][k]=component # add it to update ejson
                    break
        if 'Transformer' in component:
            component_dct = component['Transformer']
            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                if first_node == arc[0] and second_node == arc[1]:
                    count_transformers+=1
                    transformer_edges.append((first_node, second_node))
                    # below isn't correct impedance, it s just to run the code for sorted arcs
                    R, X = component_dct['z'][1][0], component_dct['z'][1][1]
                    R_line_unordered[(first_node, second_node)] = R
                    X_line_unordered[(first_node, second_node)] = X
                    # component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']
                    LineData_Z_pu_unordered[(first_node, second_node)] = R + X*1j
                    ejson_nw_updated['components'][k]=component # add it to update ejson
                    break

# check length before and after adding edges
if len(ejson_nw_updated['components']) - pre_addn_length == len(newly_added_arcs):
    print('Number of edges added match!')
else:
    print('check the edges added, not looking good....!')

# unconnected nodes post processing with updated arcs
unconnected_nodes_post, unconnected_bus_count_post = find_unconnected_nodes(bus_all, arcs_all)

if unconnected_bus_count_post != len(unconnected_nodes_after_using_removed_edges):
    print('Something aint working')
if unconnected_nodes_post != unconnected_nodes_after_using_removed_edges:
    print('Check again, should be the same nodes')

def check_for_multiple_sources_to_node():
    pass
###############################################################################
# check for multiple lines feeding a node
###############################################################################
count_to = 0 # nodes having multiple lines feeding them
for key, val in bus_arcs.items():
    if len(val["To"])>1:
        count_to+=1
        # print(key, val)
        if len(val["To"])>2:
            print('Damn SON! So many lines')
# print(count_to)

slack_node = 8183
unconnected_nodes_post.remove(slack_node) # remove slack node from unconnected nodes
# also remove the nodes downstream to the unconnceted nodes
# nodes downstream of unconected nodes are leaf nodes apart from one: 2349
# 2349 is downstream of one of the uncoonected nodes: 3212
# removing 3212 won't affect 2349 becasue it has another upstream node: 2703
downstream_leaf_nodes = [130, 7730, 8023]
unconnected_nodes_post.extend(downstream_leaf_nodes) # update nodes to be removed

# Remove the unconnected nodes from buses
for i in unconnected_nodes_post:
    for j in BusNum:
        if i == j:
            BusNum.remove(i)
            bus_arcs.pop(i)# remove the nodes from bus_arcs as well

# remove these nodes from ejson_updated before storing as json file -->> ausnet_removed.json
# remove the components from ejson_nw_updated

pre_addn_length = len(ejson_nw_updated['components'])
for node in unconnected_nodes_post:
    for k, component in ejson_nw_updated['components'].items():
        if 'node' in k:
            if 'node_' + str(node) == k:
                # remove it from the json file
                ejson_nw_updated['components'].pop(k)
                break

# check length before and after removing edges
if pre_addn_length - len(ejson_nw_updated['components']) == len(unconnected_nodes_post):
    print('Number of edges removed match!')
else:
    print('check the edges removed, not looking good....!')

# make sure BusNum is still ordered
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
# they look in order, checked manually

# Remove arcs corresponding to unconnectred nodes
arcs_removed = []
for i in unconnected_nodes_post:
    # print(i)
    for j in arcs_all:
        if i in j:
            # print(i, j)
            arcs_all.remove(j)
            arcs_removed.append(j)
# remove these arcs from ejson_updated before storing as json file -->> ausnet_removed.json
# done later

# could check here if the arcs removed belongs to transformer edges
# but it shouldn't affect the system as you removed the nodes as well
transformer_edge_removed = 0
for i in arcs_removed:
    for j in transformer_edges:
        if i == j: # you removed a transformer edge
            transformer_edge_removed = 1
            print('You have removed a transformer edge. This is TROUBLE!', i, j)

if transformer_edge_removed == 1:
    print('Transformer edge removed mate, fix it!')
else:
    print('No transformer edge removed')
    
# ceheck if the number of removed nodes matches the previously existing number
if len(bus_all) != len(BusNum) + len(unconnected_nodes_post):
    print('Mate, check it again! Nodes not removed properly')

# create new to and from with updated buses and arcs
arcs = [] # recreate ordered arcs
bus_arcs = {} # recreate ordered with busnodes

for i in BusNum: # use ordered bus
    t = []
    f = []

    for ii in arcs_all:
        if i == ii[0]:
            f.append(ii)
            if ii not in arcs:
                arcs.append(ii)
        if i == ii[1]:
            t.append(ii)
            if ii not in arcs:
                arcs.append(ii)            
        # break
    bus_arcs[i] = {"To":t,"from":f}

###############################################################################
# check again for multiple lines feeding a node
# see if removing nodes reduced the number of nodes with double lines
# Remove double lines to a node
# lines and edges have been interchangibily used in the comments
###############################################################################
count_to = 0 # nodes having multiple lines feeding them
nodes_with_multiple_source, edges_to_removed = [], []
for key, val in bus_arcs.items():
    if len(val["To"])>1: # multiple lines to a node
        count_to+=1
        nodes_with_multiple_source.append(key)
        # print(key, val)
        if len(val["To"])>2:
            print('Damn SON! So many lines')
        else: # when len is 2
            # remove an edge when multiple edges are fed to a node
            edge_removed = bus_arcs[key]["To"].pop(0) # remove the first edge
            edges_to_removed.append(edge_removed)
            arcs.remove(edge_removed)
            
# remove the edges from 'for' part of bus
for edge_removed in edges_to_removed:
    for key, val in bus_arcs.items():
        if edge_removed in val["from"]:
            bus_arcs[key]['from'].remove(edge_removed)

# create new ordered arcs characteristics
R_line, X_line, LineData_Z_pu = {}, {}, {}
for arc in arcs:
    R_line[arc] = R_line_unordered[arc]
    X_line[arc] = X_line_unordered[arc]
    LineData_Z_pu[arc] = LineData_Z_pu_unordered[arc]

# all arcs to be removed
arcs_removed.extend(edges_to_removed)

# remove these arcs from ejson_updated before storing as json file -->> ausnet_removed.json
pre_addn_length = len(ejson_nw_updated['components'])
for arc in arcs_removed:
    for k, component in ejson_nw_updated['components'].items():

        if 'Line' in component:
            component_dct = component['Line']

            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                if first_node == arc[0] and second_node == arc[1]:
                    # remove the component
                    ejson_nw_updated['components'].pop(k)
                    break
        if 'Transformer' in component:
            component_dct = component['Transformer']
            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                if first_node == arc[0] and second_node == arc[1]:
                    # remove the component
                    ejson_nw_updated['components'].pop(k)
                    break

# check length before and after removing edges
if pre_addn_length - len(ejson_nw_updated['components']) == len(arcs_removed):
    print('Number of edges removed match!')
else:
    print('check the edges removed, not looking good....!')
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# function could be defiend here with returning updated json and req params

NETWORK_REDUCE_EJSON =  "/home/shub/Documents/phd/distflow/json_files/ausnet_removed2.json"
# write updated new a json file
# with open(NETWORK_REDUCE_EJSON, 'w') as fp:
#     json.dump(ejson_nw_updated, fp, indent=2)
# manually removed component 10423 which has node 5594 on its load

# similarly have a function for measurement
measurements_from_ejson(ejson_meas, updated_nw)
print("Loaded measurement data")

# sort out measurement in dataframe
zib_nodes, non_zib_nodes = [], []

df_P = pd.DataFrame() # empty df to store non-zib P loads
df_Q = pd.DataFrame() # empty df to store non-zib Q loads
for node in BusNum:
    node_id = 'node_'+ str(node)
    meas = updated_nw.nodes[node_id].meas
    if updated_nw.nodes[node_id].meas:
        # print(f"{len(meas)} meters are associated with Node {node_id}")
        meas_df = next(iter(meas.values())).data
        non_zib_nodes.append(node)

        # concatenate different loads in one df
        new_node_P = meas_df['P']
        new_node_P = meas_df['P'].rename(columns={new_node_P.columns[0]: node})
        df_P = pd.concat((df_P, new_node_P), axis=1).sort_index()
        new_node_Q = meas_df['Q']
        new_node_Q = meas_df['Q'].rename(columns={new_node_Q.columns[0]: node})
        df_Q = pd.concat((df_Q, new_node_Q), axis=1).sort_index()        
    else:
        zib_nodes.append(node)

# convert time index to readable datetime format for interpolation
df_P.index = pd.to_datetime(df_P.index,unit='s')
df_Q.index = pd.to_datetime(df_Q.index,unit='s')

# count the nans in every df row and pick the one with the least for testing
num_nans = df_P.isnull().sum(axis=1)
idx_min = num_nans.idxmin()
# idx_min = '2018-01-28 11:55:00'

# perform interpolation to fill up NANs
df_P = df_P.interpolate(method='time', limit_direction='both')
df_Q = df_Q.interpolate(method='time', limit_direction='both')

# get P_Load and Q_Load
P_Load,Q_Load = {}, {}
for node in BusNum:
    if node in non_zib_nodes:
        P_Load[node] = df_P[node][idx_min]
        Q_Load[node] = df_Q[node][idx_min]
    if node in zib_nodes:
        P_Load[node] = 0
        Q_Load[node] = 0

'''
from collections import defaultdict
f = open('/home/shub/Downloads/ausnet_meas.json/ausnet_meas.json')
data = json.load(f)
p = {}
for node, value in list(data['measurements'].items()):
    dps = defaultdict(dict)
    if 'PLG' in value:
        for meas in value['PLG']:
            for load, v in zip(meas['element'], meas['data']):
                dps[load][meas['start_time']] = v
    for k, v in dps.items():
        p[k] = v
df2 = pd.DataFrame.from_dict(p).sort_index()
'''

###############################################################################
# Final check on the obtained network
###############################################################################
# check if edges removed arent removing transformer edges
transformer_edge_removed = 0
for i in edges_to_removed:
    for j in transformer_edges:
        if i == j: # you removed a transformer edge
            transformer_edge_removed = 1
            print('You have removed a transformer edge. This is TROUBLE!', i, j)

if transformer_edge_removed == 1:
    print('Transformer edge removed mate, fix it!')
else:
    print('No transformer edge removed')

# final check to see if there are multiple lines to node
count_to = 0 # nodes having multiple lines feeding them
for key, val in bus_arcs.items():
    if len(val["To"])>1: # multiple source to a node
        count_to+=1
        nodes_with_multiple_source.append(key)
        print(key, val)
if count_to == 0:
    print('No multiple lines to one node, mate!')

# num_edges = num_nodes -1
# total_edges = num_lines + num_transformers
if len(arcs) == len(BusNum) - 1 and len(BusNum) == len(bus_arcs):
    print('All G!')

###############################################################################
# validation of the updated network
###############################################################################
def check_path_to_every_node_from_slack(G, buses, slack_node):
    ''' 
        G: netowrk object
        buses: all nodes in the graph except slack node
        slack_node: slack node
    '''
    # check paths to evrey node from the slack node
    # there should be one path from slack node to evry node
    count_nodes = 0 # to make sure it iterates over all nodes
    path_lengths = {}
    all_nodes_are_good = 1 # a flag to see if nodes satisfy above 2 condition
    for node in buses:
        count_nodes+=1
        # gets all the path from source to target
        path_length = len(list(nx.all_simple_paths(G, source=slack_node, target=node)))
        path_lengths[slack_node, node] = path_length
        # should have the length as 1 to satisfy the above 2 conditons
        if path_length == 1:
            pass
        elif path_length == 0:
            all_nodes_are_good = 0
            print('Unconnected node: ', node)
        else:
            all_nodes_are_good = 2
            print('Multiple paths to node: ', node)        

    if all_nodes_are_good == 1:
        all_nodes_connected_to_slack = True
        print('All nodes are connected to slack node and have single path')
    else:
        all_nodes_connected_to_slack = False
        print('Either unconnected node or multiple path to nodes')
    return all_nodes_connected_to_slack
                                        
def validate_nw_using_arcs(arcs, slack_node):
    ''' validates network by using arcs'''
    G = nx.DiGraph(arcs) # define the graph using the updated arcs

    # find cycles
    try:
        cycle = list(nx.find_cycle(G, orientation="ignore"))
        cycles = True
    except nx.exception.NetworkXNoCycle:
        cycles = False
        print('No Cycles, mate!')

    # check connectivity of the graph
    if nx.is_connected(G.to_undirected()) == True:
        graph_connectivity = True
        print('Graph is all connected!')
    else:
        graph_connectivity = False
        print('Trouble, We got unconnected graphs!')
    
    buses = list(G.nodes()) # get all nodes/ buses
    # remove the slack bus
    buses.remove(slack_node) # no need to check connectivity of slack node
    # check paths to evrey node from the slack node
    # there should be one path from slack node to evry node
    all_nodes_connected_to_slack = check_path_to_every_node_from_slack(G, buses, slack_node)

    return cycles, graph_connectivity, all_nodes_connected_to_slack

def validate_nw_using_json_file(json_file, slack_node):
    ''' validates json network by getting arcs from the file'''
    arcs_all, transformer_edges = [], []
    count_lines, count_transformers = 0, 0
    for k, component in json_file['components'].items():
    
        if 'Line' in component:
            count_lines +=1
            component_dct = component['Line']
    
            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                arcs_all.append((first_node, second_node))
    
        if 'Transformer' in component:
            count_transformers+=1
            component_dct = component['Transformer']
            if 'cons' in component_dct:
                first_node = int(component_dct["cons"][0]["node"].split("_")[1])
                second_node = int(component_dct["cons"][1]["node"].split("_")[1])
                arcs_all.append((first_node, second_node))        
                transformer_edges.append((first_node, second_node))

    cycles, graph_connectivity, all_nodes_connected_to_slack = validate_nw_using_arcs(arcs_all, slack_node)
    return cycles, graph_connectivity, all_nodes_connected_to_slack

def validate_nw_using_json_file_to_network(json_file, slack_node):
    ''' validates json network by converting it to network obj'''
    nw = network_from_ejson("loaded_nw", json_file)
    G = nw.graph
        # find cycles
    try:
        cycle = list(nx.find_cycle(G, orientation="ignore"))
        cycles = True
    except nx.exception.NetworkXNoCycle:
        cycles = False
        print('No Cycles, mate!')

    # check connectivity of the graph
    if nx.is_connected(G.to_undirected()) == True:
        graph_connectivity = True
        print('Graph is all connected!')
    else:
        graph_connectivity = False
        print('Trouble, We got unconnected graphs!')
    
    buses = list(G.nodes()) # get all nodes/ buses
    # check paths to evrey node from the slack node
    # there should be one path from slack node to evry node
    slack_node = 'node_'+str(slack_node)
    # remove the slack bus
    buses.remove(slack_node) # no need to check connectivity of slack node    
    all_nodes_connected_to_slack = check_path_to_every_node_from_slack(G, buses, slack_node)
    return cycles, graph_connectivity, all_nodes_connected_to_slack

validate_nw_using_arcs(arcs_all, slack_node=8183)
validate_nw_using_json_file(ejson_nw, slack_node=8183)
validate_nw_using_json_file_to_network(ejson_nw, slack_node=8183)