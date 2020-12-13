import numpy as np
import pandas as pd
import random

# error function
def error_calc(ground_truth, estimated):
    with np.errstate(divide='ignore'):
        err = abs((estimated-ground_truth)/ground_truth * 100)
    err[np.isnan(err)] = 0
    mean_perc_error = np.mean(err)
    max_perc_error = np.max(err)

    return err, mean_perc_error, max_perc_error, max(abs(estimated-ground_truth))

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

def subset_of_measurements(num_plow_meas, num_voltage_meas, arcs, P_line, Q_line, V):
    # keys for voltage measurements
    if num_voltage_meas == 0: # return empty dict for flows
        meas_V = {}
    elif num_voltage_meas == 1:
        meas_V = {0:V[0]}
    else: 
        v_meas_keys = np.sort(random.sample(range(1,len(V)),num_voltage_meas-1))
        # add slack voltage key for measurement
        v_meas_keys = np.insert(v_meas_keys, 0, 0) # so that the voltage at slack bus is always considered
        meas_V = {k:V[k] for k in v_meas_keys} # create the meas vector for V
    
    # keys list p_line and q_line
    key_list = list(P_line.keys())
    if num_plow_meas == 0: # return empty dict for flows
        return {}, {}, meas_V
    if num_plow_meas == 1:
        meas_P_line = {key_list[0]: P_line[key_list[0]]}
        meas_Q_line = {key_list[0]: Q_line[key_list[0]]}
    else:
        # chose keys for powerflows
        p_line_index = np.sort(random.sample(range(1,len(P_line)), num_plow_meas-1))
        p_line_index = np.insert(p_line_index, 0, 0) # so that the pflow in first line is always considered
        meas_P_line = {key_list[k]: P_line[key_list[k]] for k in p_line_index}
        meas_Q_line = {key_list[k]: Q_line[key_list[k]] for k in p_line_index}
    
    return meas_P_line, meas_Q_line, meas_V

def weight_vals(meas_P_line, c, abs_error):
    weight = np.mean(np.asarray(list(meas_P_line.values()))) * c + abs_error

    return weight

def noise_addition(z, sd, mu = None):

    mu = mu if mu is not None else 0
    # noise addition to jsut non zero values only
    noise = np.random.normal(mu, sd, len(np.where(z!=0)[0]))
    # noise = 0
    z[np.where(z!=0)] = z[np.where(z!=0)] + noise # noisy data
    
    return z
