from utils import *
import cvxpy as cp
import mosek

from numpy import linalg as la


### ERRORS ###
INVALID_BIAS = 'Invalid bias type'


def FairGLASSO_fista(Sigma, mu1, eta, mu2, Z, bias_type, epsilon=.1, iters=1000,
                     prec_type=None, tol=1e-3, EARLY_STOP=False, RETURN_ITERS=False):
    """
    Solve a graphical lasso problem with fairness regularization using the FISTA algorithm.

    Parameters:
    -----------
    Sigma : numpy.ndarray
        Sample covariance matrix.
    mu1 : float
        Weight for the l1 norm.
    eta : float
        Step size.
    mu2 : float
        Weight for the fairness penalty.
    Z : numpy.ndarray
        Matrix of sensitive attributes for the fairness penalty.
    epsilon : float, optional
        Small constant to load the diagonal of the estimated Theta to ensure strict positivity (default is 0.1).
    iters : int, optional
        Number of iterations (default is 1000).
    EARLY_STOP: bool, optional
        If True, end iterations when difference small enough.
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

    _,p = Z.shape
    # Ensure Theta_current is initialized to an invertible matrix
    Theta_prev = np.eye(p)
    Theta_fista = np.eye(p)
    t_k = 1

    # Precompute fairness penalty matrix
    B = compute_bias_matrix_(Z, bias_type)

    # Initialize array to store iterations to check convergence
    if RETURN_ITERS:
        norm_iters = []

    for _ in range(iters):
        Theta_k = prox_grad_step_( Sigma, Theta_fista, B, mu2, mu1, eta, epsilon, bias_type, prec_type )
        t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
        Theta_fista = Theta_k + (t_k - 1)/t_next*(Theta_k - Theta_prev)
        
        if EARLY_STOP and np.linalg.norm(Theta_prev-Theta_k,'fro') < tol:
            break

        # Update values for next iteration
        if RETURN_ITERS:
            norm_iters.append( np.linalg.norm(Theta_prev-Theta_k,'fro') )

        Theta_prev = Theta_k
        t_k = t_next

    if RETURN_ITERS:
        return Theta_k, norm_iters
    
    return Theta_k


def compute_bias_matrix_(Z, bias_type):
    g,p = Z.shape
    p_grp = Z.sum(axis=1)
    if bias_type=='dp':
        B = [[ ((Z[a][:,None]*Z[a][None]) * (1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
               ((Z[a][:,None]*Z[b][None]) * (1-np.eye(p)))/(p_grp[a]*p_grp[b])
               if a!=b else np.zeros((p,p))
               for b in np.arange(g)] for a in range(g)]

    elif bias_type=='nodewise':
        Zab = [np.sum([Z[a]/p_grp[a]-Z[b]/p_grp[b] for b in np.delete(np.arange(g),a)],axis=0) for a in range(g)]
        B = [Zab[a][:,None]*Zab[a][None] for a in range(g)]

    else:
        raise ValueError(INVALID_BIAS)

    return B

def grad_fairness_penalty_(Theta, B, bias_type):
    p = Theta.shape[0]
    g = len(B)
    if bias_type == 'dp':
        return (2/(g*(g-1))) * np.sum([ np.sum( B[a][b] * Theta ) * B[a][b].T for a in range(g) for b in np.delete(np.arange(g),a) ], axis=0)
    if bias_type == 'nodewise':
        return (2/(p*g*(g-1)**2))*(np.sum(B,axis=0)@(Theta*(1-np.eye(p))))*(1-np.eye(p))


def prox_grad_step_(Sigma, Theta, Z, mu2, mu1, eta, epsilon, bias_type, prec_type):
    Soft_thresh = lambda R, alpha: np.maximum( np.abs(R)-alpha, 0 ) * np.sign(R)
    p = Sigma.shape[0]

    # Gradient step + soft-thresholding
    fairness_term = grad_fairness_penalty_(Theta, Z, bias_type) if mu2 != 0 else 0
    Gradient = Sigma - la.inv( Theta + epsilon*np.eye(p) ) + mu2*fairness_term
    Theta_aux = Theta - eta*Gradient
    Theta_aux[np.eye(p)==0] = Soft_thresh( Theta_aux[np.eye(p)==0], eta*mu1 )
    Theta_aux = (Theta_aux + Theta_aux.T)/2

    # Projection
    if prec_type == 'non-negative':
        # Projection onto non-negative matrices
        Theta_aux[(Theta_aux <= 0)*(np.eye(p) == 0)] = 0
    elif prec_type == 'non-positive':
        # Projection onto non-positive matrices
        Theta_aux[(Theta_aux >= 0)*(np.eye(p)==0)] = 0

    # Second projection onto PSD set
    eigenvals, eigenvecs = np.linalg.eigh( Theta_aux )
    eigenvals[eigenvals < 0] = 0
    Theta_next = eigenvecs @ np.diag( eigenvals ) @ eigenvecs.T

    return Theta_next
# ----------------------------------------------------

def FairGSR_fista(Sigma, mu, eta, alpha, Z, bias_type, iters=1000,
                     prec_type=None, tol=1e-3, EARLY_STOP=False, RETURN_ITERS=False):
    """
    Solve a graph stationary recovery problem with fairness regularization using the FISTA algorithm.
    """
    _, p = Z.shape
    # Ensure Theta_current is initialized to a valid adjacency
    Theta_prev = np.pinv(Sigma_sq)
    np.fill_diagonal(Theta_prev, 0)
    Theta_prev[Theta_prev < 0] = 0
    Theta_fista = np.copy(Theta_prev)
    t_k = 1

    # Precompute fairness penalty matrix
    B = compute_bias_matrix_(Z, bias_type)
    Sigma_sq = Sigma @ Sigma

    # Initialize array to store iterations to check convergence
    if RETURN_ITERS:
        norm_iters = []

    # NOTE: The function is exactly the same than that of FairGLASSO_fista, only changing on the prox_grad_step_
    # function. We should define a class and use inheritance.
    for _ in range(iters):
        Theta_k = prox_grad_step_stationary_( Sigma, Sigma_sq, Theta_fista, B, alpha, mu, eta, bias_type, prec_type )
        t_next = (1 + np.sqrt(1 + 4*t_k**2))/2
        Theta_fista = Theta_k + (t_k - 1)/t_next*(Theta_k - Theta_prev)
        
        if EARLY_STOP and np.linalg.norm(Theta_prev-Theta_k,'fro') < tol:
            break

        # Update values for next iteration
        if RETURN_ITERS:
            norm_iters.append( np.linalg.norm(Theta_prev-Theta_k,'fro') )

        Theta_prev = Theta_k
        t_k = t_next

    if RETURN_ITERS:
        return Theta_k, norm_iters
    
    return Theta_k

def prox_grad_step_stationary_(Sigma, Sigma_sq, Theta, Z, alpha, mu, eta, bias_type, prec_type):
    Soft_thresh = lambda R, alpha: np.maximum( np.abs(R)-alpha, 0 ) * np.sign(R)
    p = Sigma.shape[0]

    # Gradient step + soft-thresholding
    fairness_term = grad_fairness_penalty_(Theta, Z, bias_type) if mu != 0 else 0
    Gradient = alpha*(Sigma_sq @ Theta - Sigma @ Theta @ Sigma) + mu*fairness_term
    Theta_aux = Theta - eta*Gradient
    Theta_aux[np.eye(p)==0] = Soft_thresh( Theta_aux[np.eye(p)==0], eta )
    Theta_aux = (Theta_aux + Theta_aux.T)/2

    # Scale first column to 1 to avois all-zero trivial solution
    Theta_aux = Theta_aux / np.abs(Theta_aux).sum(axis=0)[0]

    # Projection
    if prec_type == 'non-negative':
        # Projection onto Adjacency matrices
        Theta_aux[(Theta_aux <= 0)*(np.eye(p) == 0)] = 0
        np.fill_diagonal(Theta_aux, 0)
    elif prec_type == 'non-positive':
        # Projection onto non-positive matrices
        Theta_aux[(Theta_aux >= 0)*(np.eye(p)==0)] = 0

    return Theta_aux

def FairGLASSO_cvx(Sigma, Z, mu1=.1, mu2=1, epsilon=.1, bias_type='dp'):
    g,p = Z.shape
    p_grp = Z.sum(axis=1)

    if bias_type =='dp':
        Z_til = lambda a,b: ( ((Z[a][:,None]*Z[a][None]) *(1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
                              ((Z[a][:,None]*Z[b][None]) *(1-np.eye(p)))/(p_grp[a]*p_grp[b]) 
                              if a!=b else np.zeros((p,p)) )
    elif bias_type =='nodewise':
        Z_til = lambda a,i: np.sum([np.eye(p)[i][:,None]*(Z[a]*(1-np.eye(p)[i]))[None]/p_grp[a] - 
                                    np.eye(p)[i][:,None]*(Z[b]*(1-np.eye(p)[i]))[None]/p_grp[b] if a!=b else 
                                    np.zeros((p,p)) for b in range(g)],axis=0)
    else:
        raise ValueError(INVALID_BIAS)

    Theta_hat = cp.Variable((p,p), PSD=True)
    obj = 0
    constr = []
    non_diag = ~np.eye(p, dtype=bool)
    
    if bias_type =='dp':
        dp = (1/(g*(g-1))) * cp.sum( [ cp.sum( cp.multiply( Z_til(a,b), Theta_hat ) )**2
                                    for a in range(g) for b in np.delete(np.arange(g),a)] )
    elif bias_type =='nodewise':
        dp = (1/(p*g*(g-1)**2)) * cp.sum( [ cp.sum( cp.multiply( Z_til(a,i), Theta_hat ) )**2
                                            for a in range(g) for i in range(p) ] )
    else:
        raise ValueError(INVALID_BIAS)

    obj = cp.trace(Sigma@Theta_hat) - cp.log_det(Theta_hat + epsilon*np.eye(p))
    obj = obj + mu1 * cp.norm( Theta_hat[non_diag], 1 )
    obj = obj + mu2 * dp
    constr += [ cp.diag(Theta_hat)>=0 ]
    constr += [ Theta_hat[non_diag]<=0 ]

    prob = cp.Problem(cp.Minimize(obj),constr)
    try:
        obj = prob.solve(solver='MOSEK', verbose=False)
    except cp.SolverError:

        print('DEBUG: MOSEK NOT WORKING!')

        try:
            obj = prob.solve(solver='CVXOPT', verbose=False)
        except cp.SolverError:
            try:
                obj = prob.solve(solver='ECOS', verbose=False)
            except cp.SolverError:
                print('Solver error. Proceed with caution.')
                return None

    return Theta_hat.value

# ----------------------------------------------------

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


# NOTE: Seems to be the same as FairGLASSO_cvx
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