# same as Network37, but modified to get load for real world dataset

import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
import random
from random import seed

import sys
sys.path.append("..") # Adds higher directory to python modules path
###########################################################
#Network characteristics
########################################################### 
busNo = 37 
Vbase = 4.8/np.sqrt(3) #kV Base voltage for normalization
Sbase = 100 #kVA Base apparent power for normalization
Zbase=1000*Vbase**2/Sbase #ohm

BusNum = np.arange(busNo)
Slack_Bus_Num=[0] #slack bus, Note that ideal secondary of the transformer is considered as the slack bus, i.e., V = 1<0 pu

Sbase_old_t = 2500/3 #kVA Transformer rated apparent power. Note that transforer impedance is given in pu and it is based on transformer rated values (S and V) as the base values. We should update this value based on new base values, i.e., Sbase and Vbase 
Zbase_t_old = 1000*Vbase**2/Sbase_old_t # Vbase is the same, however, Sbase_old_t and Sbase_new_t (Sbase) are different.
Zbase_t_new = 1000*Vbase**2/Sbase

###########################################################
#arcs connected to each bus
########################################################### 
bus_arcs = {
0   : {"To": [        ], "from": [(0, 1)]},
1   : {"To": [(0 , 1 )], "from": [(1,2)]},
2   : {"To": [(1 , 2 )], "from": [(2,3)]},
3   : {"To": [(2 , 3 )], "from": [(3,4),(3,24),(3,27)]},
4   : {"To": [(3 , 4 )], "from": [(4,5),(4,20)]},
5   : {"To": [(4 , 5 )], "from": [(5,6)]},
6   : {"To": [(5 , 6 )], "from": [(6,7),(6,19)]},
7   : {"To": [(6 , 7 )], "from": [(7,8),(7,18)]},
8   : {"To": [(7 , 8 )], "from": [(8,9)]},
9   : {"To": [(8 , 9 )], "from": [(9,10),(9,15)]},
10  : {"To": [(9 , 10)], "from": [(10,11)]},
11  : {"To": [(10, 11)], "from": [(11,12)]},
12  : {"To": [(11, 12)], "from": [(12,13),(12,14)]},
13  : {"To": [(12, 13)], "from": []},
14  : {"To": [(12, 14)], "from": []},
15  : {"To": [(9 , 15)], "from": [(15,16),(15,17)]},
16  : {"To": [(15, 16)], "from": []},
17  : {"To": [(15, 17)], "from": []},
18  : {"To": [(7 , 18)], "from": []},
19  : {"To": [(6 , 19)], "from": []},
20  : {"To": [(4 , 20)], "from": [(20,21)]},
21  : {"To": [(20, 21)], "from": [(21,22),(21,23)]},
22  : {"To": [(21, 22)], "from": []},
23  : {"To": [(21, 23)], "from": []},
24  : {"To": [(3 , 24)], "from": [(24,25),(24,26)]},
25  : {"To": [(24, 25)], "from": []},
26  : {"To": [(24, 26)], "from": []},
27  : {"To": [(3 , 27)], "from": [(27,28)]},
28  : {"To": [(27, 28)], "from": [(28,29),(28,35)]},
29  : {"To": [(28, 29)], "from": [(29,30),(29,32)]},
30  : {"To": [(29, 30)], "from": [(30,31)]},
31  : {"To": [(30, 31)], "from": []},
32  : {"To": [(29, 32)], "from": [(32,33),(32,34)]},
33  : {"To": [(32, 33)], "from": []},
34  : {"To": [(32, 34)], "from": []},
35  : {"To": [(28, 35)], "from": [(35,36)]},
36  : {"To": [(35, 36)], "from": []}
}
###########################################################
#Arcs
########################################################### 
arcs = {
(0 , 1 ) ,
(1 , 2 ) ,
(2 , 3 ) ,
(3 , 4 ) ,
(4 , 5 ) ,
(5 , 6 ) ,
(6 , 7 ) ,
(7 , 8 ) ,
(8 , 9 ) ,
(9 , 10) ,
(10, 11) ,
(11, 12) ,
(12, 13) ,
(12, 14) ,
(9 , 15) ,
(15, 16) ,
(15, 17) ,
(7 , 18) ,
(6 , 19) ,
(4 , 20) ,
(20, 21) ,
(21, 22) ,
(21, 23) ,
(3 , 24) ,
(24, 25) ,
(24, 26) ,
(3 , 27) ,
(27, 28) ,
(28, 29) ,
(29, 30) ,
(30, 31) ,
(29, 32) ,
(32, 33) ,
(32, 34) ,
(28, 35) ,
(35, 36) 
}
###########################################################
#Transformers impedances
########################################################### 
Zt_old_pu = 0.02 + 0.08j

Zt_new_pu = Zt_old_pu*Zbase_t_old/Zbase_t_new # transformer impedance in the new per unit system

###########################################################
#Line impedances (R +jX) in ohms per mile 
########################################################### 
Z1 = 0.2926 + 0.1973j  # Z1 to Z4 are chosen from IEEE 37-node datasheet. Note that only info of phase A is being used because here the assumption is that the system is single phase.
Z2 = 0.4751 + 0.2973j
Z3 = 1.2936 + 0.6713j 
Z4 = 2.0952 + 0.7758j


# Line lengths in ft, The original lengths are divided by 5280 to change to mile
length = [1 , 0.3504 , 0.1818 , 0.2500 , 0.1136 , 0.0379 , 0.0606 , 0.0606 ,  0.1061 , 0.1212 , 0.0758 , 0.0758 , 0.0379 , 0.0758 , 0.0985 , 0.0379 , 0.2424 , 0.0606 , 0.1136  ,  0.0455  ,  0.0530  , 0.0379 , 0.0530 , 0.0758 , 0.0455 , 0.0606 , 0.0682 , 0.0985 , 0.1515 , 0.1136 , 0.0530 , 0.1742 , 0.0227 , 0.1439 , 0.0152 , 0.0985]


LineData_Z_pu = {
(0 , 1 ): Zt_new_pu*length[0],
(1 , 2 ): Z1*length[1 ]/Zbase,
(2 , 3 ): Z2*length[2 ]/Zbase,
(3 , 4 ): Z2*length[3 ]/Zbase,
(4 , 5 ): Z3*length[4 ]/Zbase,
(5 , 6 ): Z3*length[5 ]/Zbase,
(6 , 7 ): Z3*length[6 ]/Zbase,
(7 , 8 ): Z3*length[7 ]/Zbase,
(8 , 9 ): Z3*length[8 ]/Zbase,
(9 , 10): Z3*length[9 ]/Zbase,
(10, 11): Z3*length[10]/Zbase,
(11, 12): Z3*length[11]/Zbase,
(12, 13): Z4*length[12]/Zbase,
(12, 14): Z3*length[13]/Zbase,
(9 , 15): Z4*length[14]/Zbase,
(15, 16): Z4*length[15]/Zbase,
(15, 17): Z4*length[16]/Zbase,
(7 , 18): Z4*length[17]/Zbase,
(6 , 19): Z3*length[18]/Zbase,
(4 , 20): Z4*length[19]/Zbase,
(20, 21): Z3*length[20]/Zbase,
(21, 22): Z4*length[21]/Zbase,
(21, 23): Z4*length[22]/Zbase,
(3 , 24): Z4*length[23]/Zbase,
(24, 25): Z4*length[24]/Zbase,
(24, 26): Z4*length[25]/Zbase,
(3 , 27): Z3*length[26]/Zbase,
(27, 28): Z3*length[27]/Zbase,
(28, 29): Z3*length[28]/Zbase,
(29, 30): Z3*length[29]/Zbase,
(30, 31): Z4*length[30]/Zbase,
(29, 32): Z4*length[31]/Zbase,
(32, 33): Z4*length[32]/Zbase,
(32, 34): Z4*length[33]/Zbase,
(28, 35): Z4*length[34]/Zbase,
(35, 36): Z4*length[35]/Zbase
}

##########################################################
# Load in kW and kVAR
##########################################################
# P_Load = {
# 0 : 0   ,
# 1 : 0   ,
# 2 : 140 ,
# 3 : 0   ,
# 4 : 0   ,
# 5 : 0   ,
# 6 : 0   ,
# 7 : 0   ,
# 8 : 85  ,
# 9 : 0   ,
# 10: 140 ,
# 11: 126 ,
# 12: 0   ,
# 13: 0   ,
# 14: 0   ,
# 15: 0   ,
# 16: 0   ,
# 17: 0   ,
# 18: 0   ,
# 19: 0   ,
# 20: 0   ,
# 21: 42  ,
# 22: 42  ,
# 23: 42  ,
# 24: 0   ,
# 25: 0   ,
# 26: 8   ,
# 27: 0   ,
# 28: 0   ,
# 29: 0   ,
# 30: 0   ,
# 31: 0   ,
# 32: 0   ,
# 33: 0   ,
# 34: 0   ,
# 35: 17  ,
# 36: 85
# }

# Q_Load = {
# 0 : 0  ,
# 1 : 0  ,
# 2 : 70 ,
# 3 : 0  ,
# 4 : 0  ,
# 5 : 0  ,
# 6 : 0  ,
# 7 : 0  ,
# 8 : 40 ,
# 9 : 0  ,
# 10: 70 ,
# 11: 62 ,
# 12: 0  ,
# 13: 0  ,
# 14: 0  ,
# 15: 0  ,
# 16: 0  ,
# 17: 0  ,
# 18: 0  ,
# 19: 0  ,
# 20: 0  ,
# 21: 21 ,
# 22: 21 ,
# 23: 21 ,
# 24: 0  ,
# 25: 0  ,
# 26: 4  ,
# 27: 0  ,
# 28: 0  ,
# 29: 0  ,
# 30: 0  ,
# 31: 0  ,
# 32: 0  ,
# 33: 0  ,
# 34: 0  ,
# 35: 8  ,
# 36: 40
# }

# fix this fuunction later
# P_Load, Q_Load = getLoadsFromRealLoads()
pAll, qAll= np.load("../NextGen/p_array.npy"), np.load("../NextGen/q_array.npy")
np.random.seed(0)
t    = random.randint(0, pAll.shape[1] - 1)
flag = 0
# scaling factor to have loading conditins similar to IEEE 37 Simulated nw
scaling_factor = 5*2.5 # 2.5: perfect, 2.85: worked
while flag ==0:
    p, q = pAll[:,t] *scaling_factor, qAll[:,t]*scaling_factor
    # print(sum(p)/Sbase, sum(q)/Sbase)
    if sum(p)/Sbase<8 and sum(q)/Sbase<4.2:
        # print(sum(p)/Sbase, sum(q)/Sbase)
        flag = 1
    scaling_factor-=1 # update scaling facotr
    # you can do more manipulation, I just did soething with trial and error
# sum of p and q to see it
# p,q = np.random.rand(busNo-1), np.random.rand(busNo-1)
P_Load, Q_Load = {}, {}
P_Load[0], Q_Load[0] = 0, 0
loadKeys = np.arange(1,busNo)
for idx, val in enumerate(loadKeys):
    P_Load[val] = p[idx]
    Q_Load[val] = q[idx]

# added nw info on top of Masoume's code
R_line = {key:val.real for key, val in LineData_Z_pu.items()} # resistance of every line
X_line = {key:val.imag for key, val in LineData_Z_pu.items()} # reactancce of every line

# will have to normalize P and Q before running state estimation
P_Load = {key:val/Sbase for key, val in P_Load.items()} # ground truth
Q_Load = {key:val/Sbase for key, val in Q_Load.items()} # ground truth