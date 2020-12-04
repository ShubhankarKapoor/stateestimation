def LinDistFlowBackwardForwardSweep(P_Load,Q_Load, which, V0=None):

    import numpy as np
    import copy

    if which == 37:
        from Network37 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase, R_line, X_line
    else:
        from Network906 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase, R_line, X_line

    I_load = {}
    for i in BusNum:
        I_load[i] = 0+0j

    I_line = {}
    for i in range(len(BusNum)-1,0,-1):
        # print(I_line[bus_arcs[i]["To"][0]])
        I_line[bus_arcs[i]["To"][0]] = 0+0j
        
    P_line, Q_line = {}, {}
    for i in range(len(BusNum)-1,0,-1):
        # print(I_line[bus_arcs[i]["To"][0]])
        P_line[bus_arcs[i]["To"][0]] = 0
        Q_line[bus_arcs[i]["To"][0]] = 0
        
    #Initialization of voltages squared magnitude
    V = {}
    for i in BusNum: #Note that bus 0 here shows the ideal secondary voltages of the transformer
        V[i] = 1 # square magnitude
    V[0] = 1 if V0 is None else V0

    #Initialization of iteration count
    k = 0 # iteration count
    e_max = 1
    tolerance = 0.000000000001

    while e_max > tolerance:
        #Number of iteration
        k = k+1 

        #Save previous iteration's voltage
        V_previous = copy.deepcopy(V)

        # #Calculation of load currents
        # for i in BusNum:
        #     I_load[i] = np.conj(P_Load[i]/Sbase+1j*Q_Load[i]/Sbase)/np.conj(V[i])
        
        #Backward sweep
        for i in range(len(BusNum)-1,0,-1):
            P_line[bus_arcs[i]["To"][0]] = P_Load[i] + sum(P_line[g] for g in bus_arcs[i]["from"] )
            Q_line[bus_arcs[i]["To"][0]] = Q_Load[i] + sum(Q_line[g] for g in bus_arcs[i]["from"] )
            
        #Forward sweep
        for (i,j) in LineData_Z_pu.keys():
            V[j] = V[i] - 2*(R_line[(i,j)]*P_line[(i,j)] + X_line[(i,j)]*Q_line[(i,j)])
        
        #Calculation of error
        e_max = max(abs(V[i] - V_previous[i]) for i in BusNum)
    Vmag = {key:np.sqrt(val) for key, val in V.items()} # sqrt of mag
    #Report Results
    # V_mag = {}
    # V_ang = {}
    # Voltage = {}
    # for i in BusNum:
    #     Voltage[i] = V[i]
    #     V_mag[i] = abs(V[i])
    #     V_ang[i] = np.angle(V[i])*(180/np.pi)

    # S_line = {}
    # for (i,j) in arcs:
    #     S_line[(i,j)] = Voltage[i]*np.conj(I_line[(i,j)]) 
    
    # would want to return current as well in futuret

    return(V, Vmag, P_line, Q_line, e_max,k)
