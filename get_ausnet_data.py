from evolve_core_tools.parser import network_from_ejson
from evolve_core_tools.parser import (
    network_to_ejson, network_from_ejson, measurements_from_ejson,  measurements_to_ejson)
import json
import numpy as np

NETWORK_SAMPLE_EJSON = '/home/shub/Documents/phd/distflow/ausnet_network.json'

 
with open(NETWORK_SAMPLE_EJSON) as f:
    ejson_nw = json.load(f)

# Example of using 'network_from_ejson' function (ejson_parser.py), that returns a new Network object
full_nw = network_from_ejson("loaded_nw", ejson_nw)
print("loaded nw")
print(full_nw)
 
BusNum = [] # double check if it type array or list
line_num, arcs = [], []

for k, component in ejson_nw['components'].items():
    
    # if 'Node' in component:
    if 'node' in k:
        print(k)
        # get the node numbers from the string node name
        BusNum.append(int(k.split("_")[1]))
        # print(k, component)
        # break

    # if 'Line' in component:

    #     # there is no line_1
    #     component_dct = component['Line']
        
    #     if 'cons' in component_dct:
    #         # print('yo', component_dct["cons"][0]["node"], component_dct["cons"][1]["node"])
    #         # if len(component_dct["cons"])!=2:
    #         #        print(k, len(component_dct["cons"]))
    #         break
    #     break

    #     if 'imped_mod' in component_dct:

    #         # print(k)

    #         component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']

    #         component_dct['z0'] = component_dct['imped_mod']['ZZ0']['z0']

    #         component_dct.pop('imped_mod')

    # if 'Transformer' in component:

    #     component_dct = component['Transformer']

    #     component_dct['nom_turns_ratio'] = component_dct['v_base'][0] / component_dct['v_base'][1]

    #     component_dct['v_winding_base'] = component_dct.pop('v_base')

    # if 'Load' in component:

    #     component_dct = component['Load']

    #     component_dct['wiring'] = 'wye'

    # if 'Infeeder' in component:

    #     component_dct = component['Infeeder']

    #     component_dct['v_setpoint'] = 22.0

# check if there are missing buses
for i in range(len(BusNum)-1):
    if abs(BusNum[i] - BusNum[i+1])!=1:
        # print(BusNum[i] - BusNum[i+1],BusNum[i], BusNum[i+1])
        pass

R_line, X_line, LineData_Z_pu = {}, {}, {}
for k, component in ejson_nw['components'].items():
    
    if 'Line' in component:

        # print(k)
        # get the line number from the string line name
        line_num.append(int(k.split("_")[1]))        

        # there is no line_1
        component_dct = component['Line']
        
        if 'cons' in component_dct:
            # print('yo', component_dct["cons"][0]["node"], component_dct["cons"][1]["node"])
            first_node = int(component_dct["cons"][0]["node"].split("_")[1])
            second_node = int(component_dct["cons"][1]["node"].split("_")[1])
            # make sure arcs are ordered
            arcs.append((first_node, second_node))
            # get R_line, X_line and imp
            R, X = component_dct['imped_mod']['ZZ0']['z'][0], component_dct['imped_mod']['ZZ0']['z'][1]
            R_line[(first_node, second_node)] = R
            X_line[(first_node, second_node)] = X
            component_dct['z'] = component_dct['imped_mod']['ZZ0']['z']
            LineData_Z_pu[(first_node, second_node)] = R + X*1j
            
            # if len(component_dct["cons"])!=2:
            #        print(k, len(component_dct["cons"]))
        #     break
        # break

##################################################
# bus_arcs
##################################################

# bus_arcs[0] = {"To":[],"from":[(0,1)]}

bus_arcs = {}
for i in BusNum:
    t = []
    f = []

    for ii in arcs:
        if i == ii[0]:
            f.append(ii)        
        if i == ii[1]:
            t.append(ii)
        # break
    bus_arcs[i] = {"To":t,"from":f}
    # break
# bus_arcs[1]["To"] = [(0,1)]

# check if there is gap between line numbers
for i in range(len(line_num)-1):
    if abs(line_num[i] - line_num[i+1])!=1:
        print(line_num[i] - line_num[i+1],line_num[i], line_num[i+1])
        pass
   

# nw = network_from_ejson("loaded_nw", ejson_nw)

# ##################################################
# # arcs
# ##################################################
# arcs = []
# for i in range(0,busNo-1):
#     arcs.append(bus_arcs[i+1]["To"][0])