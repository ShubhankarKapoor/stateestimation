import numpy as np

# measurement set for approx distflow

def meas_from_approx_distflow(x_est, meas_P_line, P_Load_state, comb_idx1, comb_idx2, 
                sum_r, sum_x, R_mat_req, X_mat_req, v_RX_Z_comb_req, mat_r, mat_x, 
                downstream_matrix, path_to_all_nodes, V0, hx):

    # input: 
        # state variable vector (x)
        # pre computed matrices
    # output: measurement vector estimate (y = h(x))

    # P_ij/Q_ij
    # use slightly modified term from grad_pline_with_vnode_loss_ass_updated_fast
    # P_ij = p_downstream_term(should be straight forward) + modified term
    p_term = x_est[comb_idx1]*x_est[comb_idx2] # 2 in the coupled terms have been included in r/x terms
    q_term = x_est[comb_idx1 + len(P_Load_state)]*x_est[comb_idx2 + len(P_Load_state)]
    pq_term = p_term + q_term

    # hacky way for line 01, wont work for other cases
    # it think it should work now, depends on mat_r/mat_x size - haven't tested it
    if meas_P_line:    
        # p_01 = sum(x_est[0:len(P_Load_state)]) + (1/V0)*(sum(pq_term * sum_r))
        # q_01 = sum(x_est[len(P_Load_state):2*len(P_Load_state)]) + (1/V0)*(sum(pq_term * sum_x))
        # hx[0] = p_01
        # hx[1] = q_01
        # matrix multiplication for linear terms
        lin_term_p = np.matmul(downstream_matrix, x_est[0: len(P_Load_state)])
        lin_term_q = np.matmul(downstream_matrix, x_est[len(P_Load_state):2*len(P_Load_state)])
        loss_term_p = (1/V0) * (np.matmul(mat_r, p_term) + np.matmul(mat_r, q_term))
        loss_term_q = (1/V0) * (np.matmul(mat_x, p_term) + np.matmul(mat_x, q_term))
        p_01 = lin_term_p + loss_term_p
        q_01 = lin_term_q + loss_term_q
        hx[0:len(meas_P_line)] = p_01
        hx[len(meas_P_line):2*len(meas_P_line)] = q_01
    # v_i
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
    return hx
