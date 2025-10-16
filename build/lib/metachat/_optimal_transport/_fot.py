import numpy as np
from scipy import sparse

from ._unot import unot 


def fot_combine_sparse(S, met_order, D, A, M, LRC, LRC_type, cutoff,
                       eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, weights=(1.0,0.0,0.0,0.0), nitermax=1e4, stopthr=1e-8, verbose=False):
    if isinstance(eps_p, tuple):
        eps_p_fot, eps_p_row, eps_p_col, eps_p_blk = eps_p
    else:
        eps_p_fot = eps_p_row = eps_p_col = eps_p_blk = eps_p
    if isinstance(rho, tuple):
        rho_fot, rho_row, rho_col, rho_blk = rho
    else:
        rho_fot = rho_row = rho_col = rho_blk = rho
    if eps_mu is None:
        eps_mu_fot = eps_p_fot; eps_mu_row = eps_p_row
        eps_mu_col = eps_p_col; eps_mu_blk = eps_p_blk
    elif isinstance(eps_mu, tuple):
        eps_mu_fot, eps_mu_row, eps_mu_col, eps_mu_blk = eps_mu
    else:
        eps_mu_fot = eps_mu_row = eps_mu_col = eps_mu_blk = eps_mu
    if eps_nu is None:
        eps_nu_fot = eps_p_fot; eps_nu_row = eps_p_row
        eps_nu_col = eps_p_col; eps_nu_blk = eps_p_blk
    elif isinstance(eps_nu, tuple):
        eps_nu_fot, eps_nu_row, eps_nu_col, eps_nu_blk = eps_nu
    else:
        eps_nu_fot = eps_nu_row = eps_nu_col = eps_nu_blk = eps_nu
    
    S_copy = S.copy()
    D_copy = D.copy()
    P_fot_sender, P_fot_receiver = fot_sparse(S_copy, met_order, D_copy, A, M, LRC, LRC_type, cutoff, \
        eps_p=eps_p_fot, eps_mu=eps_mu_fot, eps_nu=eps_nu_fot, rho=rho_fot, \
        nitermax=nitermax, stopthr=stopthr, verbose=False)
    
    S_copy = S.copy()
    D_copy = D.copy()
    if weights[1] > 0:
        P_row_sender, P_row_receiver = fot_row_sparse(S_copy, met_order, D_copy, A, M, LRC, LRC_type, cutoff, \
            eps_p=eps_p_row, eps_mu=eps_mu_row, eps_nu=eps_nu_row, rho=rho_row, \
            nitermax=nitermax, stopthr=stopthr, verbose=False)
    else:
        P_row_sender = {key: 0 for key in P_fot_sender}
        P_row_receiver = {key: 0 for key in P_fot_receiver}
    
    S_copy = S.copy()
    D_copy = D.copy()
    if weights[2] > 0:
        P_col_sender, P_col_receiver = fot_col_sparse(S_copy, met_order, D_copy, A, M, LRC, LRC_type, cutoff, \
            eps_p=eps_p_col, eps_mu=eps_mu_col, eps_nu=eps_nu_col, rho=rho_col, \
            nitermax=nitermax, stopthr=stopthr, verbose=False)
    else:
        P_col_sender = {key: 0 for key in P_fot_sender}
        P_col_receiver = {key: 0 for key in P_fot_receiver}
    
    S_copy = S.copy()
    D_copy = D.copy()
    if weights[3] > 0:
        P_blk_sender, P_blk_receiver = fot_blk_sparse(S_copy, met_order, D_copy, A, M, LRC, LRC_type, cutoff, \
            eps_p=eps_p_blk, eps_mu=eps_mu_blk, eps_nu=eps_nu_blk, rho=rho_blk, \
            nitermax=nitermax, stopthr=stopthr, verbose=False)
    else:
        P_blk_sender = {key: 0 for key in P_fot_sender}
        P_blk_receiver = {key: 0 for key in P_fot_receiver}

    P_sender = {}
    P_receiver = {}
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if not np.isinf(A[i,j]):
                P_sender[(i,j)] = float(weights[0]) * P_fot_sender[(i,j)] + float(weights[1]) * P_row_sender[(i,j)] \
                    + float(weights[2]) * P_col_sender[(i,j)] + float(weights[3]) * P_blk_sender[(i,j)]
                P_receiver[(i,j)] = float(weights[0]) * P_fot_receiver[(i,j)] + float(weights[1]) * P_row_receiver[(i,j)] \
                    + float(weights[2]) * P_col_receiver[(i,j)] + float(weights[3]) * P_blk_receiver[(i,j)]
    return P_sender, P_receiver

def fot_sparse(S, met_order, D, A, M, LRC, LRC_type, cutoff, \
               eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """ Solve the collective optimal transport problem with distance limits in sparse format.
    
    Parameters
    ----------
    S : (n_pos_s,ns_s) numpy.ndarray
        Source distributions over `n_pos_s` positions of `ns_s` source species.
    D : (n_pos_d,ns_d) numpy.ndarray
        Destination distributions over `n_pos_d` positions of `ns_d` destination species.
    A : (ns_s,ns_d) numpy.ndarray
        The cost coefficients for source-destination species pairs. An infinity value indicates that the two species cannot be coupled.
    M : (n_pos_s,n_pos_d) numpy.ndarray
        The distance (cost) matrix among the positions.
    cutoff : (ns_s,ns_d) numpy.ndarray
        The distance (cost) cutoff between each source-destination species pair. All transports are restricted by the cutoffs.
    eps_p : float, defaults to 1e-1
        The coefficient for entropy regularization of P.
    eps_mu : float, defaults to eps_p
        The coefficient for entropy regularization of unmatched source mass.
    eps_nu : float, defaults to eps_p
        The coefficient for entriopy regularization of unmatched target mass.
    rho : float, defaults to 1e2
        The coefficient for penalizing unmatched mass.
    nitermax : int, optional
        The maximum number of iterations in the unormalized OT problem. Defaults to 1e4.
    stopthr : float, optional
        The relatitive error threshold for terminating the iteration. Defaults to 1e-8.
    
    Returns
    -------
    A dictionary of scipy.sparse.coo_matrix
        The transport plan in coo sparse format for source species i and destinaton species j can be retrieved with the key (i,j).
    """
    np.set_printoptions(precision=2)
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    S_sum_amount = S.max()
    D_sum_amount = D.max()
    S = S / S_sum_amount
    D = D / D_sum_amount

    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    # Set up the large collective OT problem
    a = S.flatten('F')
    b = D.flatten('F')
    
    C_data, C_row, C_col = [], [], []

    max_cutoff = cutoff.max()
    cost_scales = []

    for i in range(ns_s):  
        # For each metabolite, compute the minimum distance among all possible long-range channel.
        LRC_name = LRC[met_order[i]]
        if isinstance(LRC_name, str):
            possible_LRC = ['base'] + [element for element in LRC_type if element in LRC_name]       
        else:
            possible_LRC = ['base']       
        stacked_LRC = np.stack([M[key] for key in possible_LRC])
        M_LRC_min = np.min(stacked_LRC, axis=0)

        # reserve the distance between spots that are lower than the cutoff value.
        M_row, M_col = np.where(M_LRC_min <= max_cutoff)
        M_max_sp = sparse.coo_matrix((M_LRC_min[M_row,M_col], (M_row, M_col)), shape = M_LRC_min.shape)

        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])])*A[i,j] )
                C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,j] )
                C_row.append( tmp_row+i*n_pos_s )
                C_col.append( tmp_col+j*n_pos_d )  

    cost_scale = np.max(cost_scales)
    C_data = np.concatenate(C_data, axis=0)
    C_row = np.concatenate(C_row, axis=0)
    C_col = np.concatenate(C_col, axis=0)
    C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a),len(b)))

    # Solve the problem on nonzero mass
    nzind_a = np.where(a > 0)[0]
    nzind_b = np.where(b > 0)[0]
    C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

    if verbose:
        print('Number of non-infinity entries in transport cost:', len(C.data))

    del C_data, C_row, C_col, C

    tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
        eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)
    
    del C_nz

    P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
    P = P.tocsr()

    # Output a dictionary of transport plans
    P_sender_expand = {}
    P_receiver_expand = {}
    for i in range(ns_s):
        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                tmp_P = P[i*n_pos_s:(i+1)*n_pos_s, j*n_pos_d:(j+1)*n_pos_d]
                # P_expand[(i,j)] = tmp_P.tocoo() * max_amount
                P_sender_expand[(i,j)] = tmp_P.tocoo() * S_sum_amount
                P_receiver_expand[(i,j)] = tmp_P.tocoo() * D_sum_amount

    return P_sender_expand, P_receiver_expand 

def fot_row_sparse(S, met_order, D, A, M, LRC, LRC_type, cutoff, \
                   eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each sender species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # S_sum_amount = S.sum()
    # D_sum_amount = D.sum()
    # S = S / S_sum_amount
    # D = D / D_sum_amount

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    max_cutoff = cutoff.max()        
    P_sender_expand = {}
    P_receiver_expand = {}

    for i in range(ns_s):

        # For each metabolite, compute the minimum distance among all possible long-range channel.
        LRC_name = LRC[met_order[i]]
        if isinstance(LRC_name, str):
            possible_LRC = ['base'] + [element for element in LRC_type if element in LRC_name]       
        else:
            possible_LRC = ['base']
        stacked_LRC = np.stack([M[key] for key in possible_LRC])
        M_LRC_min = np.min(stacked_LRC, axis=0)

        # reserve the distance between spots that are lower than the cutoff value.
        M_row, M_col = np.where(M_LRC_min <= max_cutoff)
        M_max_sp = sparse.coo_matrix((M_LRC_min[M_row,M_col], (M_row, M_col)), shape = M_LRC_min.shape)

        a = S[:,i].copy()
        D_ind = np.where(~np.isinf(A[i,:]))[0]
        b = D[:,D_ind].flatten('F').copy()
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for j in range(len(D_ind)):
                P_sender_expand[(i,D_ind[j])] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                P_receiver_expand[(i,D_ind[j])] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
            continue

        a_sum_amount = a.max()
        b_sum_amount = b.max()
        a = a / a_sum_amount
        b = b / b_sum_amount

        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for j in range(len(D_ind)):
            D_j = D_ind[j]
            tmp_nzind_s = np.where(S[:,i] > 0)[0]
            tmp_nzind_d = np.where(D[:,D_j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,D_j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[i,D_j] )
            C_row.append( tmp_row )
            C_col.append( tmp_col+j*n_pos_d )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,D_j])])*A[i,D_j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for j in range(len(D_ind)):
            tmp_P = P[:,j*n_pos_d:(j+1)*n_pos_d]
            P_sender_expand[(i,D_ind[j])] = tmp_P.tocoo() * a_sum_amount
            P_receiver_expand[(i,D_ind[j])] = tmp_P.tocoo()* b_sum_amount
            
        del P

    return P_sender_expand, P_receiver_expand

def fot_col_sparse(S, met_order, D, A, M, LRC, LRC_type, cutoff, \
                   eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    """Solve for each destination species separately.
    """
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu),abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"

    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # S_sum_amount = S.sum()
    # D_sum_amount = D.sum()
    # S = S / S_sum_amount
    # D = D / D_sum_amount
    
    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    max_cutoff = cutoff.max()
        
    # P_expand = {}
    P_sender_expand = {}
    P_receiver_expand = {}
    for j in range(ns_d):
        S_ind = np.where(~np.isinf(A[:,j]))[0]
        met_order_new = [met_order[i] for i in S_ind]
        a = S[:,S_ind].flatten('F').copy()
        b = D[:,j].copy()
        nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
        if len(nzind_a)==0 or len(nzind_b)==0:
            for i in range(len(S_ind)):
                P_sender_expand[(S_ind[i],j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s,n_pos_d), dtype=float)
                P_receiver_expand[(S_ind[i],j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s,n_pos_d), dtype=float)
            continue

        a_sum_amount = a.max()
        b_sum_amount = b.max()
        a = a / a_sum_amount
        b = b / b_sum_amount

        C_data, C_row, C_col = [], [], []
        cost_scales = []
        for i in range(len(S_ind)):
            S_i = S_ind[i]

            # For each metabolite, compute the minimum distance among all possible long-range channel.
            LRC_name = LRC[met_order_new[i]]
            if isinstance(LRC_name, str):
                possible_LRC = ['base'] + [element for element in LRC_type if element in LRC_name]       
            else:
                possible_LRC = ['base']       
            stacked_LRC = np.stack([M[key] for key in possible_LRC])
            M_LRC_min = np.min(stacked_LRC, axis=0)

            # reserve the distance between spots that are lower than the cutoff value.
            M_row, M_col = np.where(M_LRC_min <= max_cutoff)
            M_max_sp = sparse.coo_matrix((M_LRC_min[M_row,M_col], (M_row, M_col)), shape = M_LRC_min.shape)

            tmp_nzind_s = np.where(S[:,S_i] > 0)[0]
            tmp_nzind_d = np.where(D[:,j] > 0)[0]
            tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
            tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[S_i,j])[0]
            tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
            tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
            C_data.append( tmp_M_max_sp.data[tmp_ind]*A[S_i,j] )
            C_row.append( tmp_row+i*n_pos_s )
            C_col.append( tmp_col )
            cost_scales.append( np.max(M_max_sp.data[np.where(M_max_sp.data <= cutoff[S_i,j])])*A[S_i,j] )
        cost_scale = np.max(cost_scales)
        C_data = np.concatenate(C_data, axis=0)
        C_row = np.concatenate(C_row, axis=0)
        C_col = np.concatenate(C_col, axis=0)
        C = sparse.coo_matrix((C_data/cost_scale, (C_row, C_col)), shape=(len(a), len(b)))    

        nzind_a = np.where(a > 0)[0]
        nzind_b = np.where(b > 0)[0]
        C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

        del C_data, C_row, C_col, C

        tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
            eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

        del C_nz

        P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))
        P = P.tocsr()

        for i in range(len(S_ind)):
            tmp_P = P[i*n_pos_s:(i+1)*n_pos_s,:]
            # P_expand[(S_ind[i],j)] = tmp_P.tocoo() * max_amount
            P_sender_expand[(S_ind[i],j)] = tmp_P.tocoo() * a_sum_amount
            P_receiver_expand[(S_ind[i],j)] = tmp_P.tocoo() * b_sum_amount

        del P

    return P_sender_expand, P_receiver_expand

def fot_blk_sparse(S, met_order, D, A, M, LRC, LRC_type, cutoff, \
                   eps_p=1e-1, eps_mu=None, eps_nu=None, rho=1e1, nitermax=1e4, stopthr=1e-8, verbose=False):
    if eps_mu is None: eps_mu = eps_p
    if eps_nu is None: eps_nu = eps_p
    if max(abs(eps_p-eps_mu), abs(eps_p-eps_nu)) > 1e-8:
        unot_solver = "momentum"
    else:
        unot_solver = "sinkhorn"
    
    n_pos_s, ns_s = S.shape
    n_pos_d, ns_d = D.shape

    # S_sum_amount = S.sum()
    # D_sum_amount = D.sum()
    # S = S / S_sum_amount
    # D = D / D_sum_amount

    # max_cutoff = cutoff.max()
    # M_row, M_col = np.where(M <= max_cutoff)
    # M_max_sp = sparse.coo_matrix((M[M_row,M_col], (M_row,M_col)), shape=M.shape)
    
    max_cutoff = cutoff.max()

    # P_expand = {}
    P_sender_expand = {}
    P_receiver_expand = {}
    for i in range(ns_s):
        # For each metabolite, compute the minimum distance among all possible long-range channel.
        LRC_name = LRC[met_order[i]]
        if isinstance(LRC_name, str):
            possible_LRC = ['base'] + [element for element in LRC_type if element in LRC_name]       
        else:
            possible_LRC = ['base']       
        stacked_LRC = np.stack([M[key] for key in possible_LRC])
        M_LRC_min = np.min(stacked_LRC, axis=0)

        # reserve the distance between spots that are lower than the cutoff value.
        M_row, M_col = np.where(M_LRC_min <= max_cutoff)
        M_max_sp = sparse.coo_matrix((M_LRC_min[M_row,M_col], (M_row, M_col)), shape = M_LRC_min.shape)

        for j in range(ns_d):
            if not np.isinf(A[i,j]):
                a = S[:,i].copy(); b = D[:,j].copy()
                nzind_a = np.where(a > 0)[0]; nzind_b = np.where(b > 0)[0]
                if len(nzind_a)==0 or len(nzind_b)==0:
                    P_sender_expand[(i,j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                    P_receiver_expand[(i,j)] = sparse.coo_matrix(([],([],[])), shape=(n_pos_s, n_pos_d), dtype=float)
                    continue
                
                # max_amount = max(a.sum(), b.sum())
                # a_max_amount = a.sum()
                # b_max_amount = b.sum()
                # a = a / a_max_amount; b = b / b_max_amount

                a_sum_amount = a.max()
                b_sum_amount = b.max()
                a = a / a_sum_amount
                b = b / b_sum_amount

                tmp_nzind_s = np.where(S[:,i] > 0)[0]
                tmp_nzind_d = np.where(D[:,j] > 0)[0]
                tmp_M_max_sp = coo_submatrix_pull(M_max_sp, tmp_nzind_s, tmp_nzind_d)
                tmp_ind = np.where(tmp_M_max_sp.data <= cutoff[i,j])[0]
                tmp_row = tmp_nzind_s[tmp_M_max_sp.row[tmp_ind]]
                tmp_col = tmp_nzind_d[tmp_M_max_sp.col[tmp_ind]]
                C_data = tmp_M_max_sp.data[tmp_ind] * A[i,j]
                cost_scale = np.max( M_max_sp.data[np.where(M_max_sp.data <= cutoff[i,j])] )*A[i,j]
                C = sparse.coo_matrix((C_data/cost_scale, (tmp_row, tmp_col)), shape=(len(a), len(b)))

                nzind_a = np.where(a > 0)[0]
                nzind_b = np.where(b > 0)[0]
                C_nz = coo_submatrix_pull(C, nzind_a, nzind_b)

                del C_data, C

                tmp_P = unot(a[nzind_a], b[nzind_b], C_nz, eps_p, rho, \
                    eps_mu=eps_mu, eps_nu=eps_nu, sparse_mtx=True, solver=unot_solver, nitermax=nitermax, stopthr=stopthr)

                del C_nz

                P = sparse.coo_matrix((tmp_P.data, (nzind_a[tmp_P.row], nzind_b[tmp_P.col])), shape=(len(a),len(b)))

                # P_expand[(i,j)] = P * max_amount
                P_sender_expand[(i,j)] = P * a_sum_amount
                P_receiver_expand[(i,j)] = P * b_sum_amount
    
    return P_sender_expand, P_receiver_expand

def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix. 
    """
    if type(matr) != sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    
    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])
    
    lr = len(rows)
    lc = len(cols)
    
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows],
        gc[newcols]])),(lr, lc))