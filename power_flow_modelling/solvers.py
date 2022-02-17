import numpy as np
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import spsolve_triangular

# implementation of backwardforwardsweepsolver
# assumes LV network, slack bus is first node, and PQ loads only

def backwardforwardsweep(network, max_iter=100, tolerance=1e-10):
    busNo = network.busNo
    V_slack = network.V_slack
    node_a = network.node_a
    sparse = network.sparse
    # node_b = network.node_b
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    load_powers = network.load_powers

    node_voltages = np.ones((busNo - 1, 1)) * V_slack  # remove slack
    line_currents = np.zeros((len(node_a), 1))

    diff_save = []

    for iter in range(max_iter):
        ## backward sweep (calculate currents for fixed voltages)
        load_currents = np.conj(load_powers[1:]) / np.conj(node_voltages)  # ignore load_current at slack node
        if sparse:
            new_currents = spsolve_triangular(current_graph, load_currents,
                                            unit_diagonal=True,
                                              lower=False)
        else:
            new_currents = solve_triangular(current_graph, load_currents,
                                            unit_diagonal=True,
                                            check_finite=False)
        current_diff = line_currents - new_currents
        line_currents = 1. * new_currents

        ## forward sweep
        # line voltage drops
        line_voltages = np.expand_dims(line_z_pu, 1) * line_currents
        btmp = -1. * line_voltages
        btmp[0] = btmp[0] + V_slack
        if sparse:
            new_voltages = spsolve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True)
        else:
            new_voltages = solve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True, check_finite=False)
        voltage_diff = node_voltages - new_voltages
        node_voltages = 1. * new_voltages
        max_diff = np.maximum(np.max(np.abs(voltage_diff)), np.max(np.abs(current_diff)))
        diff_save.append(max_diff)
        if max_diff < tolerance:
            break

    V_all = np.vstack((np.atleast_2d(V_slack), node_voltages))

    V_mag = np.absolute(V_all)
    V_ang = np.angle(V_all) * (180 / np.pi)
    S_line = V_all[:-1] * np.conj(line_currents)

    return V_all, line_currents, V_mag, V_ang, S_line, max_diff, diff_save

# implementation of Lindistflow
# assumes LV network, slack bus is first node, and PQ loads only

def lindistflowsweep(network, max_iter=100, tolerance=1e-10, loss=None,  pflow = None):
    busNo = network.busNo
    # V_slack = network.V_slack
    V_slack = 1
    node_a = network.node_a
    sparse = network.sparse
    line_z_pu = network.line_z_pu
    current_graph = network.current_graph
    voltage_graph = network.voltage_graph
    load_powers = network.load_powers
    active_load = load_powers.real
    reactive_load = load_powers.imag

    node_voltages = np.ones((busNo - 1, 1)) * V_slack  # remove slack
    line_currents = np.zeros((len(node_a), 1))
    P_line = np.zeros((len(node_a), 1))
    Q_line = np.zeros((len(node_a), 1))
    S_line = np.zeros((len(node_a), 1))
    diff_save = []

    V_all = np.ones((busNo, 1)) * V_slack # square mag of voltages for all nodes

    loss = 0 if loss is None else loss # for voltage loss term
    pflow = 0 if pflow is None else pflow # for pflow/qflow loss term
    
    k = 0
    for iter in range(max_iter):
        k+=1
        ## backward sweep (calculate currents for fixed voltages)
        load_currents = np.conj(load_powers[1:]) / np.conj(node_voltages)  # ignore load_current at slack node
        if sparse:
            new_P_line = spsolve_triangular(current_graph, active_load[1:],
                                            unit_diagonal=True,
                                              lower=False)
            new_Q_line = spsolve_triangular(current_graph, reactive_load[1:],
                                            unit_diagonal=True,
                                              lower=False)
            new_S_line = spsolve_triangular(current_graph, load_powers[1:],
                                            unit_diagonal=True,
                                              lower=False)

        else:
            new_P_line = solve_triangular(current_graph, active_load[1:],
                                            unit_diagonal=True,
                                            check_finite=False)
            new_Q_line = solve_triangular(current_graph, reactive_load[1:],
                                            unit_diagonal=True,
                                            check_finite=False)

            # losses not added in S_line because of laziness
            new_S_line = solve_triangular(current_graph, load_powers[1:],
                                            unit_diagonal=True,
                                            check_finite=False)
        if pflow == 1:
            # calc loss term and add it to p_lines
            current_sq = (new_P_line**2 + new_Q_line**2) * (1/V_all[node_a])
            # current_sq = (P_line[bus_arcs[i]["To"][0]]**2 + Q_line[bus_arcs[i]["To"][0]]**2)* (1/V[i])
            loss_term_p = current_sq * np.expand_dims(line_z_pu, 1).real
            loss_term_q = current_sq * np.expand_dims(line_z_pu, 1).imag
            
            new_P_line += loss_term_p
            new_Q_line += loss_term_q

        S_line_diff = S_line - new_S_line
        S_line = 1. * new_S_line
        P_line = 1. * new_P_line
        Q_line = 1. * new_Q_line

        ## forward sweep
        # line voltage drops
        # line_voltages = np.expand_dims(line_z_pu, 1) * line_currents
        # line_voltages = 2*(R_line[(i,j)]*P_line[(i,j)] + X_line[(i,j)]*Q_line[(i,j)])
        line_voltages = 2 * (np.expand_dims(line_z_pu, 1).real * new_P_line  \
                        + np.expand_dims(line_z_pu, 1).imag * new_Q_line)
        btmp = -1. * line_voltages
        btmp[0] = btmp[0] + V_slack.real
        if sparse:
            new_voltages = spsolve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True)
        else:
            new_voltages = solve_triangular(voltage_graph, btmp, lower=True, unit_diagonal=True, check_finite=False)
        voltage_diff = node_voltages - new_voltages
        node_voltages = 1. * new_voltages
        # max_diff = np.maximum(np.max(np.abs(voltage_diff)), np.max(np.abs(S_line_diff)))
        max_diff = np.max(np.abs(voltage_diff))
        diff_save.append(max_diff)
        
        # update node voltages
        V_all = np.vstack((np.atleast_2d(V_slack), node_voltages))
        V_mag = 1. * np.sqrt(V_all)

        if max_diff < tolerance:
            break

    # V_ang = np.angle(V_all) * (180 / np.pi)
    # S_line = V_all[:-1] * np.conj(line_currents)

    return V_all, V_mag, P_line, Q_line, S_line, max_diff, diff_save, k