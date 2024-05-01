from utils import *
import cvxpy as cp
import mosek

# ---------------------

def GSR_reweighted(C,
                   alpha=.1,
                   mu=None,
                   eps=None,
                   delta=1e-3,
                   max_iters:int=1000,
                   eps_thresh=1e-2,
                   verbose:bool=True):
    assert (mu is None or eps is None), 'Invalid parameters. Provide mu or eps but not both.'
    assert (mu is None) or (mu>=0), 'Invalid choice of penalty parameter mu.'
    assert (eps is None) or (eps>=0), 'Invalid choice of upper bound parameter eps.'
    assert C.shape[0]==C.shape[1], 'Invalid covariance matrix.'

    # ---------------------
    if (mu is None) and (eps is None):
        eps = 1e-2
    (N,N) = C.shape
    # ---------------------


    # ---------------------
    A_prev = np.zeros((N,N))
    obj_prev = np.inf
    for itr in range(max_iters):
        W = 2*alpha/(A_prev + delta)

        # ---------------------
        A = cp.Variable((N,N), symmetric=True)
        obj = 0
        constr = []

        obj = obj + W.flatten('F')@cp.vec(A)
        constr += [ cp.abs(cp.sum(A[0])-1) <= 1e-9 ]
        constr += [ A >= 0 ]
        constr += [ cp.diag(A)==0 ]
        if mu is not None:
            assert eps is None
            obj = obj + mu*cp.norm(C@A - A@C,'fro')**2
        else:
            assert eps is not None
            constr += [ cp.sum_squares(A@C-C@A) <= eps**2 ]

        prob = cp.Problem(cp.Minimize(obj),constr)
        try:
            obj = prob.solve(solver='MOSEK', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='CVXOPT', verbose=False)
            except cp.SolverError:
                try:
                    obj = prob.solve(solver='ECOS', verbose=False)
                except cp.SolverError:
                    print('Solver error. Proceed with caution.')
                    return None
        # ---------------------
        
        A_est = A.value
        if A_est is None:
            return None

        # ---------------------
        norm_A_prev = np.sum(A_prev**2)
        A_diff = np.sum((A_est - A_prev)**2)/norm_A_prev if norm_A_prev>0 else np.sum((A_est - A_prev)**2)
        obj_diff = np.abs(obj - obj_prev)
        A_prev = A_est.copy()

        if verbose:
            print(f"Iter. {itr} | Obj. {obj:.3f} | Status: {prob.status} | Obj. diff.: {obj_diff:.3f} | A diff: {A_diff:.3f}")
        
        if obj_diff < eps_thresh:
            if verbose:
                print("Convergence achieved!")
            break
        # ---------------------

        obj_prev = obj
    # ---------------------

    A_est = A.value
    return A_est
    # ---------------------

def GSR(C,
        alpha=.1,
        mu=None,
        eps=None):
    assert (mu is None or eps is None), 'Invalid parameters. Provide mu or eps but not both.'
    assert (mu is None) or (mu>=0), 'Invalid choice of penalty parameter mu.'
    assert (eps is None) or (eps>=0), 'Invalid choice of upper bound parameter eps.'
    assert C.shape[0]==C.shape[1], 'Invalid covariance matrix.'

    # ---------------------
    if (mu is None) and (eps is None):
        eps = 1e-2
    (N,N) = C.shape
    # ---------------------


    # ---------------------
    A = cp.Variable((N,N),symmetric=True)
    obj = 0
    constr = []

    obj = obj + alpha*cp.norm(A.flatten(),1)
    constr += [ cp.abs(cp.sum(A[0])-1) <= 1e-9 ]
    constr += [ A >= 0 ]
    constr += [ cp.diag(A)==0 ]
    if mu is not None:
        assert eps is None
        obj = obj + mu*cp.norm(C@A - A@C,'fro')**2
    else:
        assert eps is not None
        constr += [ cp.sum_squares(A@C-C@A) <= eps**2 ]
    
    prob = cp.Problem(cp.Minimize(obj),constr)
    try:
        obj = prob.solve(solver='MOSEK', verbose=False)
    except cp.SolverError:
        try:
            obj = prob.solve(solver='CVXOPT', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='ECOS', verbose=False)
            except cp.SolverError:
                print('Solver error. Proceed with caution.')
                return None
    # ---------------------

    A_est = A.value
    return A_est
    # ---------------------

def FGSR_reweighted(C,Z,
                    alpha=.1,
                    beta=1,
                    mu=None,
                    eps=None,
                    delta=1e-3,
                    max_iters:int=1000,
                    eps_thresh=1e-2,
                    bias_type:str='tot_corr',
                    verbose:bool=True):
    assert (mu is None or eps is None), 'Invalid parameters. Provide mu or eps but not both.'
    assert (mu is None) or (mu>=0), 'Invalid choice of penalty parameter mu.'
    assert (eps is None) or (eps>=0), 'Invalid choice of upper bound parameter eps.'
    assert C.shape[0]==C.shape[1], 'Invalid covariance matrix.'
    assert Z.shape[1]==C.shape[0], 'Inconsistent number of nodes.'

    # ---------------------
    if (mu is None) and (eps is None):
        eps = 1e-2
    (N,N) = C.shape
    G = Z.shape[0]
    Ng = np.sum(Z,axis=1).astype(int)
    # ---------------------


    # ---------------------
    if bias_type=='dp':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='global':
        bias_penalty = lambda A: (1/(N*(N-1))) * cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) - cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) for g1 in range(G)] ) )
    elif bias_type=='groupwise':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] )
    elif bias_type=='tot_corr':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1]) - cp.sum(A[i,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    elif bias_type=='nodewise':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1])/np.maximum(Ng[g1],1) - cp.sum(A[i,Z[g2]==1] )/np.maximum(Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    else:
        print('Invalid bias type.')
    # ---------------------


    # ---------------------
    A_prev = np.zeros((N,N))
    obj_prev = np.inf
    for itr in range(max_iters):
        W = 2*alpha/(A_prev + delta)

        # ---------------------
        A = cp.Variable((N,N), symmetric=True)
        obj = 0
        constr = []

        obj = obj + W.flatten('F')@cp.vec(A)
        obj = obj + beta*bias_penalty(A)
        constr += [ cp.abs(cp.sum(A[0])-1) <= 1e-9 ]
        constr += [ A >= 0 ]
        constr += [ cp.diag(A)==0 ]
        if mu is not None:
            assert eps is None
            obj = obj + mu*cp.norm(C@A - A@C,'fro')**2
        else:
            assert eps is not None
            constr += [ cp.sum_squares(A@C-C@A) <= eps**2 ]

        prob = cp.Problem(cp.Minimize(obj),constr)
        try:
            obj = prob.solve(solver='MOSEK', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='CVXOPT', verbose=False)
            except cp.SolverError:
                try:
                    obj = prob.solve(solver='ECOS', verbose=False)
                except cp.SolverError:
                    print('Solver error. Proceed with caution.')
                    return None
        # ---------------------
        
        A_est = A.value
        if A_est is None:
            return None

        # ---------------------
        norm_A_prev = np.sum(A_prev**2)
        A_diff = np.sum((A_est - A_prev)**2)/norm_A_prev if norm_A_prev>0 else np.sum((A_est - A_prev)**2)
        obj_diff = np.abs(obj - obj_prev)
        A_prev = A_est.copy()

        if verbose:
            print(f"Iter. {itr} | Obj. {obj:.3f} | Status: {prob.status} | Obj. diff.: {obj_diff:.3f} | A diff: {A_diff:.3f}")
        
        if obj_diff < eps_thresh:
            if verbose:
                print("Convergence achieved!")
            break
        # ---------------------

        obj_prev = obj
    # ---------------------

    A_est = A.value
    return A_est
    # ---------------------

def FGSR(C,Z,
         alpha=.1,
         beta=1,
         mu=None,
         eps=None,
         bias_type:str='dp'):
    assert (mu is None or eps is None), 'Invalid parameters. Provide mu or eps but not both.'
    assert (mu is None) or (mu>=0), 'Invalid choice of penalty parameter mu.'
    assert (eps is None) or (eps>=0), 'Invalid choice of upper bound parameter eps.'
    assert C.shape[0]==C.shape[1], 'Invalid covariance matrix.'
    assert Z.shape[1]==C.shape[0], 'Inconsistent number of nodes.'

    # ---------------------
    if (mu is None) and (eps is None):
        eps = 1e-2
    (N,N) = C.shape
    G = Z.shape[0]
    Ng = np.sum(Z,axis=1).astype(int)
    # ---------------------


    # ---------------------
    if bias_type=='dp':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='global':
        bias_penalty = lambda A: (1/(N*(N-1))) * cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) - cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) for g1 in range(G)] ) )
    elif bias_type=='groupwise':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] )
    elif bias_type=='tot_corr':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1]) - cp.sum(A[i,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    elif bias_type=='nodewise':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1])/np.maximum(Ng[g1],1) - cp.sum(A[i,Z[g2]==1] )/np.maximum(Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    else:
        print('Invalid bias type.')
    # ---------------------


    # ---------------------
    A = cp.Variable((N,N),symmetric=True)
    obj = 0
    constr = []

    obj = obj + alpha*cp.norm(A.flatten(),1)
    obj = obj + beta*bias_penalty(A)
    constr += [ cp.abs(cp.sum(A[0])-1) <= 1e-9 ]
    constr += [ A >= 0 ]
    constr += [ cp.diag(A)==0 ]
    if mu is not None:
        assert eps is None
        obj = obj + mu*cp.norm(C@A - A@C,'fro')**2
    else:
        assert eps is not None
        constr += [ cp.sum_squares(A@C-C@A) <= eps**2 ]
    
    prob = cp.Problem(cp.Minimize(obj),constr)
    try:
        obj = prob.solve(solver='MOSEK', verbose=False)
    except cp.SolverError:
        try:
            obj = prob.solve(solver='CVXOPT', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='ECOS', verbose=False)
            except cp.SolverError:
                print('Solver error. Proceed with caution.')
                return None
    # ---------------------

    A_est = A.value
    return A_est
    # ---------------------

# ---------------------

def FGLASSO(C,Z,
         alpha=.1,
         beta=1,
         mu=None,
         eps=None,
         bias_type:str='dp'):
    assert (mu is None or eps is None), 'Invalid parameters. Provide mu or eps but not both.'
    assert (mu is None) or (mu>=0), 'Invalid choice of penalty parameter mu.'
    assert (eps is None) or (eps>=0), 'Invalid choice of upper bound parameter eps.'
    assert C.shape[0]==C.shape[1], 'Invalid covariance matrix.'
    assert Z.shape[1]==C.shape[0], 'Inconsistent number of nodes.'

    # ---------------------
    if (mu is None) and (eps is None):
        eps = 1e-2
    (N,N) = C.shape
    G = Z.shape[0]
    Ng = np.sum(Z,axis=1).astype(int)
    # ---------------------


    # ---------------------
    if bias_type=='dp':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='global':
        bias_penalty = lambda A: (1/(N*(N-1))) * cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) - cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) for g1 in range(G)] ) )
    elif bias_type=='groupwise':
        bias_penalty = lambda A: (1/(G*(G-1))) * cp.sum( [cp.abs( cp.sum( [cp.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - cp.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] )
    elif bias_type=='tot_corr':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1]) - cp.sum(A[i,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    elif bias_type=='nodewise':
        bias_penalty = lambda A: (1/(N*G*(G-1))) * cp.sum( [cp.sum( [cp.abs( cp.sum( [cp.sum( A[i,Z[g1]==1])/np.maximum(Ng[g1],1) - cp.sum(A[i,Z[g2]==1] )/np.maximum(Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    else:
        print('Invalid bias type.')
    # ---------------------


    # ---------------------
    A = cp.Variable((N,N),PSD=True)
    obj = 0
    constr = []
    non_diag = ~np.eye(N, dtype=bool)
    
    obj = cp.trace(A@C) - cp.log_det(A) # trace and log det
    obj = obj + alpha*cp.norm(A[non_diag],1) #L1 norm penalty
    obj = obj + beta*bias_penalty(A) # bias penalty
    constr += [ cp.abs(cp.sum(A[0])-1) <= 1e-9] # avoiding zero solution
    #constr += [ A >= 0 ] # Positive values
    #constr += [ cp.diag(A)==0 ] #zero diagonal
    #if mu is not None:
    #    assert eps is None
    #    obj = obj + mu*cp.norm(C@A - A@C,'fro')**2
    #else:
    #    assert eps is not None
    #    constr += [ cp.sum_squares(A@C-C@A) <= eps**2 ]
    
    prob = cp.Problem(cp.Minimize(obj),constr)
    try:
        obj = prob.solve(solver='MOSEK', verbose=False)
    except cp.SolverError:
        try:
            obj = prob.solve(solver='CVXOPT', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='ECOS', verbose=False)
            except cp.SolverError:
                print('Solver error. Proceed with caution.')
                return None
    # ---------------------

    A_est = A.value
    np.fill_diagonal(A_est,0)
    return A_est
    # ---------------------

# ---------------------