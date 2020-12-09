import numpy as np
import pandas as pd

# error function
def error_calc(ground_truth, estimated):
    with np.errstate(divide='ignore'):
        err = abs((estimated-ground_truth)/ground_truth * 100)
    err[np.isnan(err)] = 0 
    mean_perc_error = np.mean(err)
    max_perc_error = np.max(err)
    
    return err, mean_perc_error, max_perc_error, max(abs(estimated-ground_truth))

def create_mes_set(filename_bus, filename_branch):
    
    
    # get measurement vectors from csv files
    f1 = pd.read_csv(filename_branch)
    f1 = f1.sort_values('to_bus')
    f1['arcs'] = list(zip(f1.from_bus, f1.to_bus))
    
    f1['value'] = f1['value']*10
    meas_Q_line = f1['q_from_mvar']*10
    
    p_line_meas = dict(zip(f1[f1['measurement_type'] == 'p'].arcs, f1[f1['measurement_type'] == 'p'].value))
    q_line_meas = dict(zip(f1[f1['measurement_type'] == 'q'].arcs, f1[f1['measurement_type'] == 'q'].value))
    


    
    f2 = pd.read_csv(filename_bus)
    meas_P_load = f2['p_mw']*10
    meas_Q_load = f2['q_mvar']*10
    
    f3 = pd.read_csv('data/mm_bus_v_noisy.csv')
    meas_V = f3['vm_pu']**2
    z = np.concatenate((meas_P_line, meas_Q_line, meas_P_load, meas_Q_load, meas_V)) # ground truth for meas
    
    # return separatelyand together the measurements
        