from utils import *
import cvxpy as cp
import mosek

from numpy import linalg as la

# ---------------------
def fairness_penalty(Theta, Z, bias_type)
    G,N = Z.shape
    if bias_type=='dp':
        B = np.zeros((N,N))
        for g in range(G):
            for h in range(G):
                if h != g:
                    zg = Z[g,:]==1
                    zh = Z[h,:]==1
                    Ng = np.sum(zg)
                    Nh = np.sum(zh)
                    cg = Ng**2-Ng
                    cgh = Ng*Nh
                    Cgh = cgh*np.outer(zg,zg.T) - cg*np.outer(zh,zg.T)
                    B += 1/(cg*cgh)*np.trace(Theta@Cgh)*Cgh.T 
        bias_penalty = B
    elif bias_type=='nodewise':
        B = np.zeros((G,N))
        for g in range(G):
            v1 = Z[g,:]==1
            v2 = Z[g,:]==0
            Ng = np.sum(v1)
            Nh = np.sum(v2)
            B[g,v1] = (G-1)/Ng
            B[g,v2] = -1/Nh
            
        bias_penalty = (B.T @ B) @ Theta
           
    return bias_penalty
        

def prox_grad_step_(C_hat, Theta, Z, beta, lamb, eta, epsilon, bias_type):
    Soft_thresh = lambda R, alpha: np.maximum(np.abs(R) - alpha, 0)*np.sign(R)
    N = C_hat.shape[0]

    fairness_term = fairness_penalty(Theta, Z, bias_type)
    # Gradient step + soft-thresholding
    Gradient = C_hat - la.inv(Theta + epsilon*np.eye(N)) + beta*fairness_term
    Theta_aux = Soft_thresh(Theta - eta*Gradient, eta*lamb)
    Theta_aux = (Theta_aux + Theta_aux.T)/2
    
    # Projection 
    eigenvals, eigenvecs = la.eigh(Theta_aux)
    eigenvals[eigenvals < 0] = 0
    Theta_next = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    return Theta_next

def node_FGL_ppgd(C_hat, lamb, eta, beta, B, epsilon=.1, iters=1000, A_true=None):
    """
    Solve a graphical lasso problem with node fairness regularization using Projected Proximal Gradient Descent (PPGD).

    Parameters:
    -----------
    C_hat : numpy.ndarray
        Sample covariance matrix.
    lamb : float
        Weight for the l1 norm.
    eta : float
        Step size.
    beta : float
        Weight for the fairness penalty.
    B : numpy.ndarray
        Matrix of sensitive attributes for the fairness penalty.
    epsilon : float, optional
        Small constant to load the diagonal of the estimated Theta to ensure strict positivity (default is 0.1).
    iters : int, optional
        Number of iterations (default is 1000).
    A_true : numpy.ndarray or None, optional
        True precision matrix to keep track of the error (default is None).

    Returns:
    --------
    Theta_next : numpy.ndarray
        Estimated precision matrix.
    errs_A : numpy.ndarray
        Array of errors in precision matrix estimation over iterations (if A_true is provided).

    Notes:
    ------
    Projected Proximal Gradient Descent (PPGD) implementation with the second demographic parity penalty.
    """
    N = C_hat.shape[0]
    # Initialize Theta_k to an invertible matrix
    Theta_k = np.eye(N)

    # Precompute B^T*B for efficiency
    B_TB = beta * B.T @ B

    errs_A = np.zeros(iters)
    if A_true is not None:
        Theta_non_diag = A_true[~np.eye(N, dtype=bool)]
        norm_Theta_true = la.norm(Theta_non_diag)

    for i in range(iters):
        Theta_next = prox_grad_step_(C_hat, Theta_k, B_TB, lamb, eta, epsilon)
        Theta_k = Theta_next

        if A_true is not None:
            errs_A[i] = (la.norm(Theta_non_diag - Theta_k[~np.eye(N, dtype=bool)])/norm_Theta_true)**2

    return Theta_next, errs_A


def node_FGL_fista(C_hat, lamb, eta, beta, Z, bias_type, epsilon=.1, iters=1000, A_true=None):
    """
    Solve a graphical lasso problem with node fairness regularization using the FISTA algorithm.

    Parameters:
    -----------
    C_hat : numpy.ndarray
        Sample covariance matrix.
    lamb : float
        Weight for the l1 norm.
    eta : float
        Step size.
    beta : float
        Weight for the fairness penalty.
    B : numpy.ndarray
        Matrix of sensitive attributes for the fairness penalty.
    epsilon : float, optional
        Small constant to load the diagonal of the estimated Theta to ensure strict positivity (default is 0.1).
    iters : int, optional
        Number of iterations (default is 1000).
    A_true : numpy.ndarray or None, optional
        True precision matrix to keep track of the error (default is None).

    Returns:
    --------
    Theta_k : numpy.ndarray
        Estimated precision matrix.
    errs_A : numpy.ndarray
        Array of errors in precision matrix estimation over iterations (if A_true is provided).

    Notes:
    ------
    FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) implementation with the second demographic parity penalty.
    """
    N = C_hat.shape[0]
    # Ensure Theta_current is initialized to an invertible matrix
    Theta_prev = np.eye(N)
    Theta_fista = np.eye(N)
    t_k = 1

    # Initialize array to store errors in precision matrix estimation
    errs_A = np.zeros(iters)

    # Compute the norm of non-diagonal elements of A_true for error calculation
    if A_true is not None:
        Theta_non_diag = A_true[~np.eye(N, dtype=bool)]
        norm_Theta_true = la.norm(Theta_non_diag)

    for i in range(iters):
        Theta_k = prox_grad_step_(C_hat, Theta_fista, Z, beta, lamb, eta, epsilon, bias_type)
        t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
        Theta_fista = Theta_k + (t_k - 1)/t_next*(Theta_k - Theta_prev)

        # Update values for next iteration
        Theta_prev = Theta_k
        t_k = t_next

        # Calculate error in precision matrix estimation if A_true is provided
        if A_true is not None:
            errs_A[i] = (la.norm(Theta_non_diag - Theta_k[~np.eye(N, dtype=bool)])/norm_Theta_true)**2

    np.fill_diagonal(Theta_k,0)
    Theta_k = np.abs(Theta_k)
    return Theta_k #, errs_A


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