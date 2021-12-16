import numpy as np
# measurement set for approx distflow

def meas_from_approx_distflow(x_est, P_Load_state, comb_idx1, comb_idx2, 
                sum_r, sum_x, R_mat_req, X_mat_req, v_RX_Z_comb_req, V0, hx):

    # input: 
        # state variable vector (x)
        # pre computed matrices
    # output: measurement vector estimate (y = h(x))

    # P_ij/Q_ij
    # use slightly modified term from grad_pline_with_vnode_loss_ass_updated_fast
    # P_ij = p_downstream_term(should be straight forward) + modified term
    p_term = x_est[comb_idx1]*x_est[comb_idx2]
    q_term = x_est[comb_idx1 + len(P_Load_state)]*x_est[comb_idx2 + len(P_Load_state)]
    pq_term = p_term + q_term

    # hacky way for line 01, wont work for 
    p_01 = sum(x_est[0:len(P_Load_state)]) + (1/V0)*(sum(pq_term * sum_r))
    q_01 = sum(x_est[len(P_Load_state):2*len(P_Load_state)]) + (1/V0)*(sum(pq_term * sum_x))
    hx[0] = p_01
    hx[1] = q_01

    # v_i
    # calculate v_i again
    # looks incorrect
    # maybe check part by part
    
    # lin part looks correct -- yep! checked, it is correct.
    lin_part = np.matmul(R_mat_req, x_est[0:len(P_Load_state)]) + np.matmul(X_mat_req, x_est[len(P_Load_state):2*len(P_Load_state)])
    
    # check the additional part
    addn_part = np.matmul(v_RX_Z_comb_req, pq_term)
    
    # combine the terms
    v_i = V0 - 2 * lin_part - (1/V0*(addn_part))
    
    # # another way to calc additional term
    # addn_part1 = np.matmul(z_common_path[meas_V_idx, :], pq_term)
    # # start for the second addn part
    # addn_part2 = np.matmul(v_node_RX_comb[meas_V_idx, :], pq_term)
    # # combine the terms
    # v_i2 = x_est[-1] - (2 * lin_part) - (1/x_est[-1] * addn_part1) - (2/x_est[-1] * addn_part2)        
    
    hx[-len(v_i):] = v_i
    # print('b', hx[-len(v_i):])
    return hx
