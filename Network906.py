import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
from random import seed
import time

###########################################################
#Network characteristics
########################################################### 
busNo = 906+1
ReadLineData = pd.read_excel (r'/home/shub/Documents/phd/distflow/LVNetworks/906/Network_1/Feeder_1/Lines.xlsx')
ReadLineCodeData = pd.read_excel (r'/home/shub/Documents/phd/distflow/LVNetworks/906/Network_1/Feeder_1/LineCode.xlsx')
ReadLoadBuses = pd.read_excel (r'/home/shub/Documents/phd/distflow/LVNetworks/906/Network_1/Feeder_1/Loads.xlsx')
LoadData = pd.read_excel (r'/home/shub/Documents/phd/distflow/LVNetworks/906/Network_1/Feeder_1/LoadShapes.xlsx')
Vbase = .48/np.sqrt(3) #kV
Sbase = 1 #kVA three-phase

BusNum = np.arange(busNo)
Slack_Bus_Num=[0] #slack bus
Zbase = 1000*Vbase**2/Sbase

Zbase_t_old = 1000*Vbase**2/(500/3) # 500 kVA is the three phase rated apparent power of the transformer 
Zbase_t_new = 1000*Vbase**2/Sbase
Z_t_old_pu = [
[(0.0011+0.002j) ,0+0j            ,0+0j           ],
[0+0j            ,(0.0011+0.002j) ,0+0j           ],
[0+0j            ,0+0j            ,(0.0011+0.002j)]
]
Z_t_new_pu = [[element*Zbase_t_old/Zbase_t_new for element in row] for row in Z_t_old_pu]

##################################################
# bus_arcs
##################################################
bus_arcs = {}
bus_arcs[0] = {"To":[],"from":[(0,1)]}

for i in range(1,busNo):
    t = []
    f = []
    for ii in range(0,busNo-2):
        if ReadLineData.iat[ii,2] == i:
            a1 = ReadLineData.iat[ii,1]
            a2 = ReadLineData.iat[ii,2]
            t.append(tuple((a1,a2)))
        if ReadLineData.iat[ii,1] == i:
            a3 = ReadLineData.iat[ii,1]
            a4 = ReadLineData.iat[ii,2]
            f.append(tuple((a3,a4)))
    bus_arcs[i] = {"To":t,"from":f}

bus_arcs[1]["To"] = [(0,1)]

##################################################
# arcs
##################################################
arcs = []
for i in range(0,busNo-1):
    arcs.append(bus_arcs[i+1]["To"][0])

##################################################
# Line Data
##################################################
# 1- Calculation of Zabc
LineData = {}

for i in range(0,len(ReadLineData)):
    Z012 = np.zeros((3,3),dtype=complex)
    a = -0.500000000000000 + 0.866025403784439j
    A = [[1/3,1/3,1/3],[1/3,a/3,a*a/3],[1/3,a*a/3,a/3]]
    invA = [[1,1,1],[1,a*a,a],[1,a,a*a]]

    code = ReadLineData.iat[i,6]
    for ii in range(0,len(ReadLineCodeData)):      
        if ReadLineCodeData.iat[ii,0] == code:
            Z0 = complex(ReadLineCodeData.iat[ii,4],ReadLineCodeData.iat[ii,5])
            Z1 = complex(ReadLineCodeData.iat[ii,2],ReadLineCodeData.iat[ii,3])
            Z2 = Z1
            Z012[0,0] = Z0*ReadLineData.iat[i,4]
            Z012[1,1] = Z1*ReadLineData.iat[i,4]
            Z012[2,2] = Z2*ReadLineData.iat[i,4]
    Zabc = np.dot(invA,np.dot(Z012,A))
    LineData[(ReadLineData.iat[i,1],ReadLineData.iat[i,2])] = [ [Zabc[0,0],Zabc[0,1],Zabc[0,2]] , [Zabc[1,0],Zabc[1,1],Zabc[1,2]] , [Zabc[2,0],Zabc[2,1],Zabc[2,2]]]

LineData[(0,1)] = [[element for element in row] for row in Z_t_new_pu]

###########################################################
#P.U impedance matrix
########################################################### 
LineData_Z_pu_threePhase = {} # three phase impedance
LineData_Z_pu = {} # single phase impedance
for (i,j) in LineData.keys():
    LineData_Z_pu_threePhase[(i,j)] = [[element*0.001/Zbase for element in row] for row in LineData[(i,j)]] # 0.001 is to convert the length of the lines to km
    LineData_Z_pu[(i,j)] = LineData_Z_pu_threePhase[(i,j)][0][0] # Only Phase "A" is considered. 

###########################################################
#Load Data
########################################################### 
P_Load = {}
Q_Load = {}

LoadBuses = [] #Identify load buses from data sheet
phase = 'A' # get loads for a specific phase
LoadBuses = np.asarray(ReadLoadBuses[ReadLoadBuses['phaseNumber'] == phase]['BusNumber'])

seed(10)
SS = [] 
for i in range(0,len(LoadBuses)):
    SS.append(random.randrange(0,54,1))

j = 0
for i in BusNum:
    if i in LoadBuses:
        P_Load[i] = LoadData.iloc[:,SS[j]] # 1440*1 zero vector (1-min resolution load data)
        j = j+1
    else:
        P_Load[i] = 0*LoadData.iloc[:,0] # 1440*1 zero vector
    Q_Load[i] = 0*LoadData.iloc[:,0] # 1440*1 zero vector

scaling_factor = 1
# Let's just consider a specific time like t = 720 (data for 12pm)
for i in BusNum:
    P_Load[i] = P_Load[i][720] * scaling_factor
    Q_Load[i] = Q_Load[i][720] * scaling_factor      
    # print(P_Load, Q_Load)

# added nw info on top of Masoume's code
R_line = {key:val.real for key, val in LineData_Z_pu.items()} # resistance of every line
X_line = {key:val.imag for key, val in LineData_Z_pu.items()} # reactancce of every line

# will have to normalize P and Q before running state estimation
P_Load = {key:val/Sbase for key, val in P_Load.items()} # ground truth
Q_Load = {key:val/Sbase for key, val in Q_Load.items()} # ground truth
