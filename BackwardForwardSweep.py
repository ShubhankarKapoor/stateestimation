def BackwardForwardSweep(P_Load,Q_Load,which, V0=None, max_iter= None):

    import numpy as np
    import copy

    # giving an insanely high number below so it converges with tol when max_iters are missing
    max_iter = 10e12 if max_iter is None else max_iter # max iterations without considering tolerance  
    
    if which == 37:
        from Network37 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase
    else:
        from Network906 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase

    I_load = {}
    for i in BusNum:
        I_load[i] = 0+0j

    I_line = {}
    for i in range(len(BusNum)-1,0,-1):
        # print(I_line[bus_arcs[i]["To"][0]])
        I_line[bus_arcs[i]["To"][0]] = 0+0j
    
    #Initialization of voltages
    V = {}
    for i in BusNum: #Note that bus 0 here shows the ideal secondary voltages of the transformer
        V[i] = 1+0j
    V[0] = 1 if V0 is None else (V0)**(1/2)+0j
    
    #Initialization of iteration count
    k = 0 # iteration count
    e_max = 1
    tolerance = 0.000000000001 

    while e_max > tolerance and k<max_iter:
        #Number of iteration
        k = k+1 

        #Save previous iteration's voltage
        V_previous = copy.deepcopy(V)

        #Calculation of load currents
        for i in BusNum:
            I_load[i] = np.conj(P_Load[i]+1j*Q_Load[i])/np.conj(V[i])
        
        #Backward sweep
        # for i in range(len(BusNum)-1,0,-1):
        for i in BusNum[:0:-1]:
            I_line[bus_arcs[i]["To"][0]] = I_load[i] + sum(I_line[g] for g in bus_arcs[i]["from"] ) 

        #Forward sweep
        for (i,j) in LineData_Z_pu.keys():
            V[j] = V[i] - LineData_Z_pu[(i,j)] * I_line[(i,j)] 
        
        #Calculation of error
        e_max = max(abs(V[i] - V_previous[i]) for i in BusNum)
    
    #Report Results
    V_mag = {}
    V_ang = {}
    Voltage = {}
    for i in BusNum:
        Voltage[i] = V[i]
        V_mag[i] = abs(V[i])
        V_ang[i] = np.angle(V[i])*(180/np.pi)

    S_line = {}
    for (i,j) in LineData_Z_pu.keys():
        S_line[(i,j)] = Voltage[i]*np.conj(I_line[(i,j)])
        
    # S_line = {}
    # for (i,j) in arcs:
    #     S_line[(i,j)] = Voltage[i]*np.conj(I_line[(i,j)]) 

    return(V_mag,V_ang,Voltage,S_line,I_line,I_load,e_max,k)
