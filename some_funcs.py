import numpy as np
import pandas as pd
import random
from LinDistFlowBackwardForwardSweep import LinDistFlowBackwardForwardSweep
from BackwardForwardSweep import BackwardForwardSweep
# from solvers import cost
import solvers
import matplotlib.pyplot as plt
# error function
def error_calc(ground_truth, estimated):
    ''' function to calculate different types of errors '''
    # when gtruth is 0 it gives a warning in divide
    # this is to avoid the warning    
    with np.errstate(divide='ignore', invalid='ignore'):
        perc_err = abs((estimated-ground_truth)/ground_truth * 100)
    perc_err[np.isnan(perc_err)] = 0

    # Percentage error
    mean_perc_error = np.mean(perc_err)
    max_perc_error = np.max(perc_err)

    # absolute error
    abs_error = abs(estimated-ground_truth)
    max_abs_error = max(abs_error)
    mean_abs_error = np.mean(abs_error)

    # index of max absolute error
    max_index = np.where(abs_error == abs_error.max())[0]

    return perc_err, mean_perc_error, max_perc_error, abs_error, mean_abs_error, max_abs_error, max_index

def refactor_estimates(num_all_state_vars, x_estn, non_zib_index, num_buses):
    ''' takes the estimated states, state variables (non zib here): could be extended
        returns full estimate array, separates P_Load/ Q_Load estimates
    '''
    full_x_est = np.zeros((num_all_state_vars))
    full_x_est[non_zib_index] = x_estn[0:len(non_zib_index)] # insert p vals
    full_x_est[num_buses+np.asarray(non_zib_index)] = x_estn[len(non_zib_index):2*len(non_zib_index)] # insert q vals
    full_x_est[-1] = x_estn[-1] # slack bus square voltage

    keys = list(range(num_buses))
    array = full_x_est[0:num_buses]
    P_Load_est = dict(zip(keys, array))
    Q_Load_est = dict(zip(keys, full_x_est[num_buses:num_buses*2]))

    return full_x_est, P_Load_est, Q_Load_est

def error_calc_refactor(x, x_estn, non_zib_index, num_buses, est_lin, est_full_ac, 
                        which, V, V_mag, loss = None, pflow = None, state_err= None, V_err = None):
    ''' Takes the non zib estiates states and returns the complete error
        x: true state values
        x_estn: estimated state values
    '''
    state_err = state_err if state_err is not None else True
    V_err = V_err if V_err is not None else True
    loss = 0 if loss is None else loss # term to reconstruct v using the loss term
    pflow = 0 if pflow is None else pflow # term to reconstruct v using the loss term
    print('loss', loss, 'pflow', pflow)

    if state_err == True: # error between states
        # get the following for compatibility for error and power flow calc
        full_x_est, P_Load_est, Q_Load_est = refactor_estimates(len(x), x_estn,
                                                                non_zib_index, num_buses)

        # print(x[not_considered], full_x_est[not_considered])
        # print(x[not_considered+37], full_x_est[not_considered+37])

        # calculate error between state vectors
        errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p = error_calc(x[0:num_buses], full_x_est[0:num_buses])
        errperc_vectorq, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, _ = error_calc(x[num_buses:2*num_buses], full_x_est[num_buses:2*num_buses])

        # print some results
        # print('mean_perc_error, max_perc_error, mean_abs_error, max_abs_error')
        print('p bus err:', mean_error_st_p, max_error_st_p, mean_error_st_abs_p, max_error_st_abs_p) 
        print('q bus err:', mean_error_st_q, max_error_st_q, mean_error_st_abs_q, max_error_st_abs_q)

    if V_err == True: # error between measurements
        # Regenerated measurements using the estimated states
        if est_lin == 1: # reconstruction using lindist or distflow depending on loss and pflow vals
            [V_con, V_mag_con ,P_line_con, Q_line_con, _, e_max_con, k_con] = LinDistFlowBackwardForwardSweep(
                P_Load_est, Q_Load_est, which, full_x_est[-1], loss, pflow, max_iter=1) # using lindistflow
 
        # using Full AC Network
        if est_full_ac == 1:
            [V_mag_con,_,_,S_line_con,_,_,e_max,k] = BackwardForwardSweep(P_Load_est,
                    Q_Load_est, which, full_x_est[-1], max_iter=1)
            Vsq_con =  {key:val**2 for key, val in V_mag_con.items()} # square of V_mag
            V_con = Vsq_con

            # when running full network
            P_line_con = {key:val.real for key, val in S_line_con.items()} # resistance of every line
            Q_line_con = {key:val.imag for key, val in S_line_con.items()} # reactancce of every line

        # error calc between measurements
        # V_mag^2 and V_mag error
        errperc_vector_vsq, mean_vsq_err, max_vsq_err, _, mean_abs_vsq_err, max_abs_vsq_err, _ = error_calc(np.array(list(V.values())), np.array(list(V_con.values())))
        errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err, _ = error_calc(np.array(list(V_mag.values())), np.array(list(V_mag_con.values())))
        print('vmag bus err:', mean_vmag_err, max_vmag_err, mean_abs_vmag_err, max_abs_vmag_err)

        # pflow and qflow error
        # _, mean_pflow_err, max_pflow_err, mean_abs_pflow_err, max_abs_pflow_err, _ = error_calc(np.array(list(P_line.values())), np.array(list(P_line_con.values())))
        # _, mean_qflow_err, max_qflow_err, mean_abs_qflow_err, max_abs_qflow_err, _ = error_calc(np.array(list(Q_line.values())), np.array(list(Q_line_con.values())))
    #     return errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
    #            errperc_vectorq, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \
    #            errperc_vector_vmag, mean_vmag_err, max_vmag_err, _, mean_abs_vmag_err, max_abs_vmag_err
    # else:
    #     return errperc_vectorp, mean_error_st_p, max_error_st_p, st_err_p, mean_error_st_abs_p, max_error_st_abs_p, max_index_p, \
    #            errperc_vectorp, mean_error_st_q, max_error_st_q, st_err_q, mean_error_st_abs_q, max_error_st_abs_q, \

def noise_addition(z, sd, mu = None):

    mu = mu if mu is not None else 0
    # noise addition to jsut non zero values only
    noise = np.random.normal(mu, sd, len(np.where(z!=0)[0]))
    # noise = 0
    z[np.where(z!=0)] = z[np.where(z!=0)] + noise # noisy data

    return z

def create_mes_set(filename_bus, filename_branch):
    ''' get measurement vectors from csv files '''
    # for testing
    # filename_bus = 'data/r1_bus_meas.csv'
    # filename_branch = 'data/r1_branch_meas.csv'

    f1 = pd.read_csv(filename_branch)
    f1 = f1.sort_values('to_bus')
    f1['arcs'] = list(zip(f1.from_bus, f1.to_bus))

    f1['value'] = f1['value']*10
    # meas_Q_line = f1['q_from_mvar']*10

    p_line_meas = dict(zip(f1[f1['measurement_type'] == 'p'].arcs, f1[f1['measurement_type'] == 'p'].value))
    q_line_meas = dict(zip(f1[f1['measurement_type'] == 'q'].arcs, f1[f1['measurement_type'] == 'q'].value))

    f2 = pd.read_csv(filename_bus)
    f2.sort_values('name_bus')
    # f2['value'] = f2['value']*10
    f2.loc[f2['measurement_type'] == 'p', 'value'] = f2[f2['measurement_type'] == 'p']['value'] * 10
    f2.loc[f2['measurement_type'] == 'q', 'value'] = f2[f2['measurement_type'] == 'q']['value'] * 10
    f2.loc[f2['measurement_type'] == 'v', 'value'] = f2[f2['measurement_type'] == 'v']['value'] ** 2
    # f2[f2['measurement_type'] == 'p']['value'] = f2[f2['measurement_type'] == 'p']['value'] * 10

    meas_P_load = dict(zip(f2[f2['measurement_type'] == 'p'].name_bus, f2[f2['measurement_type'] == 'p'].value))
    meas_Q_load = dict(zip(f2[f2['measurement_type'] == 'q'].name_bus, f2[f2['measurement_type'] == 'q'].value))
    meas_V = dict(zip(f2[f2['measurement_type'] == 'v'].name_bus, f2[f2['measurement_type'] == 'v'].value))

    # z = np.concatenate((meas_P_line, meas_Q_line, meas_P_load, meas_Q_load, meas_V)) # ground truth for meas

    return p_line_meas, q_line_meas, meas_P_load, meas_Q_load, meas_V

def subset_of_measurements(num_plow_meas, arcs, P_line, Q_line, V):
    ''' function for subset of measurements of pflow, qflow'''

    # keys list p_line and q_line
    key_list = list(P_line.keys())
    if num_plow_meas == 0: # return empty dict for flows
        return {}, {}
    if num_plow_meas == 1:
        meas_P_line = {key_list[0]: P_line[key_list[0]]}
        meas_Q_line = {key_list[0]: Q_line[key_list[0]]}
    else:
        # randomly chose keys for powerflows
        p_line_index = np.sort(random.sample(range(1,len(P_line)), num_plow_meas-1))
        p_line_index = np.insert(p_line_index, 0, 0) # so that the pflow in first line is always considered
        meas_P_line = {key_list[k]: P_line[key_list[k]] for k in p_line_index}
        meas_Q_line = {key_list[k]: Q_line[key_list[k]] for k in p_line_index}

    return meas_P_line, meas_Q_line

def bus_measurements_equal_distribution(P_Load, Q_Load, V, primary_branch_flow_p, 
                     primary_branch_flow_q, non_zib_index, zib_index, 
                     num_known_meas=None, indices = None):
    ''' function for pseudo and known p, q, v bus measurements
    indices: array of index of known measurements in non_zib_index
    '''
    # indices are for index in non zib array for now
    # will have to fix it in future if required
    if indices is None:
        if num_known_meas is None:
            raise ValueError("Needs either indices ot num_known_meas")
        shuffled_vals = np.random.permutation(len(non_zib_index))
        known_meas_idx = (shuffled_vals[0:num_known_meas])
        unknown_meas_idx = shuffled_vals[num_known_meas:]
    else:
        if num_known_meas is not None and num_known_meas!=len(indices):
            raise ValueError("NUmber of indices not equal to length of known measurements")
        else:
            known_meas_idx = indices
            # missing indices from known_meas_idx
            unknown_meas_idx = np.setdiff1d(np.arange(0,len(non_zib_index)), known_meas_idx) 
            # should fix the above mentioned issue here

    known_meas1 = {non_zib_index[k]: P_Load[non_zib_index[k]] for k in known_meas_idx}
    known_meas2 = {k:P_Load[k] for k in zib_index} # no load measurements
    known_meas = {**known_meas1, **known_meas2}

    # known measurements
    P_known_meas = dict(sorted(known_meas.items()))
    Q_known_meas = {k:Q_Load[k] for k in P_known_meas.keys()}
    V_known_meas = {k:V[k] for k in known_meas1.keys()} # get voltage vals for known measurements
    
    # add some additional voltages
    # V_known_meas[36] = V[36]
    # V_known_meas[35] = V[35]
    # V_known_meas[26] = V[26]
    # V_known_meas[23] = V[23]
    # V_known_meas[22] = V[22]
    # V_known_meas[21] = V[21]
    # V_known_meas[11] = V[11]
    # V_known_meas[10] = V[10]
    # V_known_meas[8] = V[8]
    # V_known_meas[2] = V[2]
    # V_known_meas[817] = V[817]
    # V_known_meas[860] = V[860]
    # V_known_meas[861] = V[861]
    # V_known_meas[896] = V[896]
    # V_known_meas[906] = V[906]
    # V_known_meas[898] = V[898]

    # include slack bus voltage at all times

    # V_known_meas = {k: V[k] for k in P_Load.keys()}
    if 0 in V_known_meas.keys():
        V_known_meas = dict(sorted(V_known_meas.items()))
    else:
        V_known_meas[0] = V[0] # add slack bus voltage
        V_known_meas = dict(sorted(V_known_meas.items()))

    # distribute the load equally between unknown loads
    if len(unknown_meas_idx) != 0:
        dist_load_p = (primary_branch_flow_p - sum(P_known_meas.values()))/ len(unknown_meas_idx)
        # dist_load = (sum(Q_Load.values()) - sum(P_known_meas.values()))/ len(unknown_meas_idx)
        dist_load_q = (primary_branch_flow_q - sum(Q_known_meas.values()))/ len(unknown_meas_idx)

        # assign initial vals to pseudo measurements
        pseudo_meas = {non_zib_index[k]: 0 for k in unknown_meas_idx}
        P_pseudo_meas = dict(sorted(pseudo_meas.items()))
        Q_pseudo_meas = {k: 0 for k in P_pseudo_meas.keys()}
    else:
        P_pseudo_meas, Q_pseudo_meas = {}, {}

    return P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas, V_known_meas

def bus_measurements_with_noise(P_Load, Q_Load, primary_branch_flow_p, 
                     primary_branch_flow_q, non_zib_index, zib_index, 
                     num_known_meas=None, indices = None):
    ''' function for pseudo and known p,q bus 
    indices: array of index of known measurements in non_zib_index
    '''
    # indices are for index in non zib array for now
    # will have to fix it in future if required
    if indices is None:
        if num_known_meas is None:
            raise ValueError("Needs either indices ot num_known_meas")
        shuffled_vals = np.random.permutation(len(non_zib_index))
        known_meas_idx = (shuffled_vals[0:num_known_meas])
        unknown_meas_idx = shuffled_vals[num_known_meas:]
    else:
        if num_known_meas is not None and num_known_meas!=len(indices):
            raise ValueError("NUmber of indices not equal to length of known measurements")
        else:
            known_meas_idx = indices
            # missing indices from known_meas_idx
            unknown_meas_idx = np.setdiff1d(np.arange(0,len(non_zib_index)), known_meas_idx) 
            # should fix the above mentioned issue here

    known_meas1 = {non_zib_index[k]: P_Load[non_zib_index[k]] for k in known_meas_idx}
    known_meas2 = {k:P_Load[k] for k in zib_index} # no load measurements
    known_meas = {**known_meas1, **known_meas2}

    # known measurements
    P_known_meas = dict(sorted(known_meas.items()))
    Q_known_meas = {k:Q_Load[k] for k in P_known_meas.keys()}

    # distribute the load equally between unknown loads
    if len(unknown_meas_idx) != 0:

        # pseudo measurements
        pseudo_meas_p = {non_zib_index[k]:P_Load[non_zib_index[k]] for k in unknown_meas_idx}
        pseudo_meas_q = {k: Q_Load[k] for k in pseudo_meas_p.keys()}
        z_pseudo = np.asarray(list(pseudo_meas_p.values()) + list(pseudo_meas_q.values()))
        sd = 0 # 0.01: 1% error
        z_noise = noise_addition(z_pseudo, sd)
        P_pseudo_meas = dict(zip(pseudo_meas_p.keys(),z_noise[0:len(pseudo_meas_p)]))
        P_pseudo_meas = dict(sorted(P_pseudo_meas.items()))
        Q_pseudo_meas = dict(zip(pseudo_meas_p.keys(),z_noise[len(pseudo_meas_p):]))
        Q_pseudo_meas = dict(sorted(Q_pseudo_meas.items()))
    else:
        P_pseudo_meas, Q_pseudo_meas = {}, {}
    
    return P_known_meas, P_pseudo_meas, Q_known_meas, Q_pseudo_meas

def countour_plot(w1, w2, theta, thetas, y, W, jacobian_matrix):
    '''
    Parameters
    ----------
    w1 : state variable index number
    w2 : state variable index number
    theta : final estimated state vars
    thetas : estimated state vars at each iteration
    y : measurements
    W : weight matrix
    jacobian_matrix :

    Returns
    -------
    None.

    '''
    levels = [0.0, 1.0, 2.0, 4.0, 8.0, 12.0, 14.0]
    levels = [0.0, 0.0005, 0.05, 0.2, 0.4, 2.0, 3.0, 4.0, 5.0]
    # different values of the state vars used for contour plot
    w1_vec = np.linspace(-10, 10, 100)
    w2_vec = np.linspace(-10, 10, 100)
    # w1_vec = np.linspace(-np.max(x_true)*2, np.max(x_true)*2, 1000)
    # w2_vec = np.linspace(-np.max(x_true)*2, np.max(x_true)*2, 1000)
    # w1_vec = thetas[:,w1]
    # w1_vec = w1_vec[::10]
    # w2_vec = thetas[:,w2]
    # w2_vec = w2_vec[::10]
    weight_copy = np.zeros_like(theta)
    weight_copy[:] = theta[:]
    cost_func = np.zeros((len(w1_vec),len(w2_vec)))
    for i, value1 in enumerate(w1_vec):
        print(i)
        for j, value2 in enumerate(w2_vec[i:]): # to avoid repitition
            
            weight_copy[w1] = value1
            weight_copy[w2] = value2
            cost_val = cost(weight_copy, jacobian_matrix, y, W)
            cost_func[i, j+i] = cost_val
            cost_func[j+i,i] = cost_val
    plt.figure()
    CS = plt.contour(w1_vec, w1_vec, cost_func, levels, colors='black', linestyles='dashed', linewidths=1)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.contourf(w1_vec, w2_vec, cost_func, levels)

def weight_vals(meas_P_line, c, abs_error):
    weight = np.mean(np.asarray(list(meas_P_line.values()))) * c + abs_error
    return weight
