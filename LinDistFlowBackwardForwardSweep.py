def LinDistFlowBackwardForwardSweep(P_Load,Q_Load, which, V0=None, loss=None, pflow = None, max_iter= None):

    import numpy as np
    import copy

    if which == 37:
        from Network37 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase, R_line, X_line
    else:
        from Network906 import BusNum, bus_arcs, LineData_Z_pu, arcs, Sbase, R_line, X_line

    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term
    # giving an insanely high number below so it converges with tol when max_iters are missing
    max_iter = 10e12 if max_iter is None else max_iter # max iterations without considering tolerance  

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

    while e_max > tolerance and k<max_iter: # when both conditions are satisfied
        #Number of iteration
        k = k+1 

        #Save previous iteration's voltage
        V_previous = copy.deepcopy(V)

        # #Calculation of load currents
        # for i in BusNum:
        #     I_load[i] = np.conj(P_Load[i]/Sbase+1j*Q_Load[i]/Sbase)/np.conj(V[i])

        if loss == 0 and pflow == 0: # lindistflow
            #Backward sweep
            # for i in range(len(BusNum)-1,0,-1):
            for i in BusNum[:0:-1]:
                P_line[bus_arcs[i]["To"][0]] = P_Load[i] + sum(P_line[g] for g in bus_arcs[i]["from"] )
                Q_line[bus_arcs[i]["To"][0]] = Q_Load[i] + sum(Q_line[g] for g in bus_arcs[i]["from"] )

            #Forward sweep
            for (i,j) in LineData_Z_pu.keys():
                V[j] = V[i] - 2*(R_line[(i,j)]*P_line[(i,j)] + X_line[(i,j)]*Q_line[(i,j)])
        else: # adding loss in voltage and in p/q atm & provides options

            #Backward sweep
            # for i in range(len(BusNum)-1,0,-1):
            for i in BusNum[:0:-1]:
                # if no loss included in pflow/ qflow
                P_line[bus_arcs[i]["To"][0]] = P_Load[i] + sum(P_line[g] for g in bus_arcs[i]["from"] )
                Q_line[bus_arcs[i]["To"][0]] = Q_Load[i] + sum(Q_line[g] for g in bus_arcs[i]["from"] )

                if pflow == 1: # pflow/qflow loss term
                    current_sq = (P_line[bus_arcs[i]["To"][0]]**2 + Q_line[bus_arcs[i]["To"][0]]**2)* (1/V[i])
                    loss_term_p = current_sq * LineData_Z_pu[bus_arcs[i]["To"][0]].real
                    loss_term_q = current_sq * LineData_Z_pu[bus_arcs[i]["To"][0]].imag
                    P_line[bus_arcs[i]["To"][0]] = P_line[bus_arcs[i]["To"][0]] + loss_term_p
                    Q_line[bus_arcs[i]["To"][0]] = Q_line[bus_arcs[i]["To"][0]] + loss_term_q  
                    # if loss_term_p > 0.13: # loss greater than 25kW
                    #     print('loss is:', loss_term_p/2.60*100, bus_arcs[i],)

            #Forward sweep
            if loss == 1: # voltage loss term
                for (i,j) in LineData_Z_pu.keys():
                    loss_term = ((abs(LineData_Z_pu[(i,j)])**2) * (P_line[(i,j)]**2 + Q_line[(i,j)]**2)) * (1/V[i])
                    V[j] = V[i] - 2*(R_line[(i,j)]*P_line[(i,j)] + X_line[(i,j)]*Q_line[(i,j)]) + loss_term
            else: # no loss term in voltage
                for (i,j) in LineData_Z_pu.keys():
                    V[j] = V[i] - 2*(R_line[(i,j)]*P_line[(i,j)] + X_line[(i,j)]*Q_line[(i,j)])

        #Calculation of error
        e_max = max(abs(V[i] - V_previous[i]) for i in BusNum)
        if k == max_iter:
            print('Maybe reconsider increasing iterations')
    Vmag = {key:np.sqrt(val) for key, val in V.items()} # sqrt of mag

    # reorder the keys
    S_line, P_line_ordered, Q_line_ordered = {}, {}, {}
    for (i,j) in LineData_Z_pu.keys():
        S_line[(i,j)] = P_line[(i,j)] + 1j*Q_line[(i,j)]
        P_line_ordered[(i,j)] = P_line[(i,j)]
        Q_line_ordered[(i,j)] = Q_line[(i,j)]

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
    # print('k is:', k)
    return(V, Vmag, P_line_ordered, Q_line_ordered, S_line, e_max,k)
