import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import commentjson as json
import os
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans

madcmap = 'viridis'
num_total = 25

blue_base = np.array([.267,.467,.831])
blue_min = np.array([.800,.875,1.00])
blue_max = np.array([.090,.165,.302])
blues = [blue_min]
blues1 = list((blue_min[None] + (blue_base-blue_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
blues2 = list((blue_base[None] + (blue_max-blue_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
blues += blues1 + blues2

gray_base = np.array([.584,.588,.592])
gray_min = np.array([.894,.894,.894])
gray_max = np.array([.216,.220,.224])
grays = [gray_min]
grays1 = list((gray_min[None] + (gray_base-gray_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
grays2 = list((gray_base[None] + (gray_max-gray_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
grays += grays1 + grays2

red_base = np.array([.831,.267,.443])
red_min = np.array([.969,.835,.878])
red_max = np.array([.302,.090,.157])
reds = [red_min]
reds1 = list((red_min[None] + (red_base-red_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
reds2 = list((red_base[None] + (red_max-red_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
reds += reds1 + reds2

green_base = np.array([.337,0.761,0.620])
green_min = np.array([.792,0.933,0.886])
green_max = np.array([.059,0.333,0.243])
greens = [green_min]
greens1 = list((green_min[None] + (green_base-green_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
greens2 = list((green_base[None] + (green_max-green_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
greens += greens1 + greens2

yellow_base = np.array([.867,0.608,0.231])
yellow_min = np.array([.984,0.855,0.663])
yellow_max = np.array([.392,0.235,0.000])
yellows = [yellow_min]
yellows1 = list((yellow_min[None] + (yellow_base-yellow_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
yellows2 = list((yellow_base[None] + (yellow_max-yellow_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
yellows += yellows1 + yellows2

purple_base = np.array([.576,.463,.816])
purple_min = np.array([.812,.757,.980])
purple_max = np.array([.282,.125,.498])
purples = [purple_min]
purples1 = list((purple_min[None] + (purple_base-purple_min)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
purples2 = list((purple_base[None] + (purple_max-purple_base)[None]*np.linspace(0,1,int((num_total-3)/2)+1)[1:][:,None]))
purples += purples1 + purples2

def madimshow(mat,cmap:str=madcmap,xlabel:str='',ylabel:str='',axis=True,figsize=(4,4),vmin=None,vmax=None):
    fig = plt.figure(figsize=figsize)
    ax = fig.subplots()
    args = {'cmap':cmap}
    if vmin is not None:
        args['vmin'] = vmin
    if vmax is not None:
        args['vmax'] = vmax
    ax.imshow(mat,**args)
    if len(xlabel)>0:
        ax.set_xlabel(xlabel)
    if len(ylabel)>0:
        ax.set_ylabel(ylabel)
    if not axis:
        ax.axis('off')
    fig.tight_layout()

# ---------------------------

def vec(X):
    return la.vstack(X)

def soft_thresh(z, lmbda):
    return np.maximum(np.abs(z) - lmbda, 0)*np.sign(z)

def mat2lowtri(A):
    assert len(A.shape)==2 and A.shape[0]==A.shape[1], "Must provide square matrix."
    return A.T[np.triu_indices_from(A,1)]

def lowtri2mat(a):
    N = .5 + np.sqrt(2*len(a) + .25)
    assert np.abs(N-int(N))<1e-13
    N = int(N)
    A = np.full((N,N), 0, dtype=type(a[0]))
    low_tri_indices = np.triu_indices(N,1)
    A[low_tri_indices[1],low_tri_indices[0]] = a
    A += A.T
    return A

def kchoose2(N:int):
    return int(N*(N-1)/2) + int(N==0)

def get_upptri_inds(N:int,K:int=1):
    inds = mat2lowtri(np.arange(N**2).reshape((N,N),order='F').T)
    if K>1:
        inds = np.concatenate(list(map(lambda k:inds+k*N**2,np.arange(K))))
    return inds

def get_lowtri_inds(N:int,K:int=1):
    inds = mat2lowtri(np.arange(N**2).reshape((N,N),order='F'))
    if K>1:
        inds = np.concatenate(list(map(lambda k:inds+k*N**2,np.arange(K))))
    return inds

def block_diag(As:list):
    assert all([As[i].ndim in [1,2] for i in range(len(As))]), 'Invalid dimension of array.'
    As = [As[i] if As[i].ndim==2 else As[i][:,None] for i in range(len(As))]
    return np.concatenate([np.concatenate([As[j] if j==i else np.zeros((As[i].shape[0],As[j].shape[1])) for j in range(len(As))],axis=1) for i in range(len(As))],axis=0)

# ---------------------------

def generate_erdos_renyi(N:int=20,edge_prob=.2):
    assert N>0, 'Invalid number of nodes.'
    assert edge_prob>=0 and edge_prob<=1, 'Invalid edge probability.'
    return lowtri2mat(np.random.binomial(1,edge_prob,kchoose2(N)))

def generate_regular(N:int,k:int):
    assert N>0, 'Invalid number of nodes.'
    assert 0<k<int(N/2), 'Invalid degree.'
    A = np.zeros((N,N))
    for n in range(N):
        nbr_inds = np.arange(n-k,n+k+1)%N
        A[n,nbr_inds] = 1
        A[nbr_inds,n] = 1
    np.fill_diagonal(A,0)
    return A

def generate_connected_er(N:int,edge_prob:float):
    assert N>0, 'Invalid number of nodes.'
    assert edge_prob>=0 and edge_prob<=1, 'Invalid edge probability.'
    A = lowtri2mat(np.random.binomial(1,edge_prob,int(N*(N-1)/2)))
    L = np.diag(np.sum(A,axis=0)) - A
    while np.sum(np.abs(la.eigh(L)[0])<1e-9)>1:
        A = lowtri2mat(np.random.binomial(1,edge_prob,int(N*(N-1)/2)))
        L = np.diag(np.sum(A,axis=0)) - A
    return A

def generate_ksbm(N:int=20,k:int=2,in_prob=.8,out_prob=.1,block_assign=None):
    assert N>0, 'Invalid number of nodes.'
    assert k>0, 'Invalid number of blocks.'
    assert in_prob>=0 and in_prob<=1 and out_prob>=0 and out_prob<=1, 'Invalid edge probability.'
    if block_assign is None:
        block_assign=np.random.choice(k,N)
    else:
        assert len(block_assign)==N
    A=np.zeros((N,N),dtype=int)
    in_inds=np.where(block_assign[:,None]==block_assign[None])
    in_inds=tuple(np.sort(in_inds,axis=0))
    out_inds=np.where(block_assign[:,None]!=block_assign[None])
    out_inds=tuple(np.sort(out_inds,axis=0))

    A[in_inds]=np.random.binomial(1,in_prob,len(in_inds[0]))
    A[out_inds]=np.random.binomial(1,out_prob,len(out_inds[0]))
    A=A+A.T
    A[np.eye(N)==1]=0
    return A

def create_filter(A,L:int=3,h=None):
    if h is not None:
        L = len(h)
    else:
        h = np.random.rand(L)
    h = h/la.norm(h,1)
    H = np.sum([h[l]*la.matrix_power(A,l) for l in range(L)],axis=0)
    return H

def create_poly_cov(A=None,H=None,L:int=3):
    assert (A is not None or H is not None), 'Must provide adjacency or filter.'
    if H is None:
        H = create_filter(A,L)
    C = H@H
    return C

def create_gmrf_cov(A):
    (N,N) = A.shape
    eigvals = la.eigh(A)[0]
    C_inv = (.01-eigvals.min())*np.eye(N) + (.9+.1*np.random.rand())*A
    C = la.inv(C_inv)
    return C

def create_mtp2_cov(A):
    (p,p) = A.shape
    eigvals = la.eigh(A)[0]
    Theta = 2*(.01 - eigvals.min())*np.eye(p) - lowtri2mat( mat2lowtri(A) * ( .9 + .1*np.random.rand(kchoose2(p)) ) )
    while np.min(la.eigvalsh(Theta)[0]) < 0:
        Theta[np.eye(p)==1] = np.diag(Theta) * 1.01
    Sigma = la.inv(Theta)
    return Sigma, Theta

def poly_samples(H, M:int=None):
    (N,N) = H.shape
    if M is None:
        M = 1000
    X = H@np.random.randn(N,M)
    return X

def gmrf_samples(C, M:int=None):
    (N,N) = C.shape
    if M is None:
        M = 1000
    X = np.random.multivariate_normal(np.zeros(N),C,M).T
    return X

def est_cov(X):
    M = X.shape[1]
    C_est = X@X.T/M
    return C_est

# ---------------------------

def compute_dp2(Theta,Z):
    (g,p) = Z.shape
    p_grp = Z.sum(axis=1)
    Z_til = lambda a,b: ( ((Z[a][:,None]*Z[a][None]) *(1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
                          ((Z[a][:,None]*Z[b][None]) *(1-np.eye(p)))/(p_grp[a]*p_grp[b]) 
                          if a!=b else np.zeros_like(Theta) )
    dp = (1/(g*(g-1))) * np.sum( [ np.sum( Z_til(a,b) * Theta )**2 
                                   for a in range(g) for b in np.delete(np.arange(g),a)] )
    # Z_til = [[ ((Z[a][:,None]*Z[a][None]) *(1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
    #            ((Z[a][:,None]*Z[b][None]) *(1-np.eye(p)))/(p_grp[a]*p_grp[b]) if a!=b else
    #             np.zeros_like(Theta) for b in range(g)] for a in range(g)]
    # dp = (1/(g*(g-1))) * np.sum( [ np.sum( Z_til[a][b] * Theta )**2 
    #                                for a in range(g) for b in np.delete(np.arange(g),a)] )
    return dp

def compute_dp1(Theta,Z):
    (g,p) = Z.shape
    p_grp = Z.sum(axis=1)
    Z_til = lambda a,b: ( ((Z[a][:,None]*Z[a][None]) *(1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
                          ((Z[a][:,None]*Z[b][None]) *(1-np.eye(p)))/(p_grp[a]*p_grp[b]) 
                          if a!=b else np.zeros_like(Theta) )
    dp = (1/(g*(g-1))) * np.sum( [ np.abs (np.sum( Z_til(a,b) * Theta ) )
                                   for a in range(g) for b in np.delete(np.arange(g),a)] )
    # Z_til = [[ ((Z[a][:,None]*Z[a][None]) *(1-np.eye(p)))/(p_grp[a]*(p_grp[a]-1)) - 
    #            ((Z[a][:,None]*Z[b][None]) *(1-np.eye(p)))/(p_grp[a]*p_grp[b]) if a!=b else
    #             np.zeros_like(Theta) for b in range(g)] for a in range(g)]
    # dp = (1/(g*(g-1))) * np.sum( [ np.abs(np.sum( Z_til[a][b] * Theta ))
    #                                for a in range(g) for b in np.delete(np.arange(g),a)] )
    return dp

def compute_nodedp2(Theta,Z):
    (g,p) = Z.shape
    p_grp = Z.sum(axis=1)
    Z_til = lambda a,i: np.sum([np.eye(p)[i][:,None]*(Z[a]*(1-np.eye(p)[i]))[None]/p_grp[a] - 
                                np.eye(p)[i][:,None]*(Z[b]*(1-np.eye(p)[i]))[None]/p_grp[b] 
                                if a!=b else np.zeros_like(Theta) for b in range(g)],axis=0)
    dp = (1/(p*g*(g-1)**2)) * np.sum( [ 
        np.sum( Z_til(a,i) * Theta )**2
        for a in range(g) for i in range(p) ] )
    # Z_til = [[np.sum([np.eye(p)[i][:,None]*(Z[a]*(1-np.eye(p)[i]))[None]/p_grp[a] - 
    #                   np.eye(p)[i][:,None]*(Z[b]*(1-np.eye(p)[i]))[None]/p_grp[b] if a!=b else 
    #                   np.zeros_like(Theta) for b in range(g)],axis=0)
    #         for i in range(p)] for a in range(g)]
    # dp = (1/(p*g*(g-1)**2)) * np.sum( [ 
    #     np.sum( Z_til[a][i] * Theta )**2
    #     for a in range(g) for i in range(p) ] )
    return dp

def compute_nodedp1(Theta,Z):
    (g,p) = Z.shape
    p_grp = Z.sum(axis=1)
    Z_til = lambda a,i: np.sum([np.eye(p)[i][:,None]*(Z[a]*(1-np.eye(p)[i]))[None]/p_grp[a] - 
                                np.eye(p)[i][:,None]*(Z[b]*(1-np.eye(p)[i]))[None]/p_grp[b] 
                                if a!=b else np.zeros_like(Theta) for b in range(g)],axis=0)
    dp = (1/(p*g*(g-1)**2)) * np.sum( [ 
        np.abs( np.sum( Z_til(a,i) * Theta ) )
        for a in range(g) for i in range(p) ] )
    # Z_til = [[np.sum([np.eye(p)[i][:,None]*(Z[a]*(1-np.eye(p)[i]))[None]/p_grp[a] - 
    #                   np.eye(p)[i][:,None]*(Z[b]*(1-np.eye(p)[i]))[None]/p_grp[b] if a!=b else 
    #                   np.zeros_like(Theta) for b in range(g)],axis=0)
    #         for i in range(p)] for a in range(g)]
    # dp = (1/(p*g*(g-1)**2)) * np.sum( [ 
    #     np.abs( np.sum( Z_til[a][i] * Theta ) )
    #     for a in range(g) for i in range(p) ] )
    return dp

def compute_bias(A,Z,bias_type:str='weighted_dp'):
    assert A.shape[0]==A.shape[1], 'Invalid adjacency matrix.'
    assert Z.shape[1]==A.shape[0], 'Inconsistent number of nodes.'
    N = A.shape[0]
    G = Z.shape[0]
    Ng = np.sum(Z,axis=1).astype(int)

    if bias_type=='dp':
        # Unbiased if for every pair of groups, the likelihoods of an edge connecting two of the same group and two different groups are the same
        return (1/(G*(G-1))) * np.sum( [np.abs( np.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - np.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='dp_unweighted':
        # Unbiased if for every pair of groups, the likelihoods of an edge connecting two of the same group and two different groups are the same (binary version)
        A_uw = A / ( np.max(A) + int(np.max(A)==0) )
        return (1/(G*(G-1))) * np.sum( [np.abs( np.sum( A_uw[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - 
                                                np.sum( A_uw[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) 
                                        for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='dp_scaled':
        # Unbiased if for every pair of groups, the likelihoods of an edge connecting two of the same group and two different groups are the same
        return ( (N*(N-1)) / ( np.sum(A) + int(np.sum(A)==0) ) ) * (1/(G*(G-1))) * np.sum( [np.abs( np.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - 
                                                                                                  np.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) ) 
                                                                                          for g1 in range(G) for g2 in np.delete(np.arange(G),g1)] )
    elif bias_type=='ratio':
        # Unbiased if for every node and one of its edges, the likelihoods of connecting to a node from the same group or a different group are the same
        return (1/(G*(G-1))) * np.sum( [np.abs( np.sum( [np.sum( A[Z[g1]==1][:,Z[g1]==1] ) / ( np.sum(A[Z[g1]==1]) + int(np.sum(A[Z[g1]==1])==0) ) - 
                                                         np.sum( A[Z[g1]==1][:,Z[g2]==1] ) / ( np.sum(A[Z[g1]==1]) + int(np.sum(A[Z[g1]==1])==0) ) 
                                                         for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] )
        # return (1/(G-1)) * np.sum( [np.abs( np.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.sum( A[Z[g1]==1] ) - 1/G ) for g1 in range(G)] )
    elif bias_type=='global':
        # Unbiased if the likelihoods of an edge connecting nodes from the same group or a different group are the same
        return (1/(N*(N-1))) * np.abs( np.sum( [np.sum( A[Z[g1]==1][:,Z[g1]==1] ) - np.sum( [np.sum( A[Z[g1]==1][:,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) for g1 in range(G)] ) )
    elif bias_type=='groupwise':
        # Unbiased if for every group the likelihoods of an edge connecting to the same group or a different group are the same
        return (1/(G*(G-1))) * np.sum( [np.abs( np.sum( [np.sum( A[Z[g1]==1][:,Z[g1]==1] ) / np.maximum(Ng[g1]*(Ng[g1]-1),1) - np.sum( A[Z[g1]==1][:,Z[g2]==1] ) / np.maximum(Ng[g1]*Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] )
    elif bias_type=='tot_corr':
        # Unbiased if for each node and group, the number of edges connecting that node to the same group or different group are the same
        return (1/(N*G*(G-1))) * np.sum( [np.sum( [np.abs( np.sum( [np.sum( A[i,Z[g1]==1]) - np.sum(A[i,Z[g2]==1] ) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    elif bias_type=='nodewise':
        # Unbiased if for each node and group, the likelihoods of an edge connecting that node to the same group or different group are the same
        return (1/(N*G*(G-1))) * np.sum( [np.sum( [np.abs( np.sum( [np.sum( A[i,Z[g1]==1])/np.maximum(Ng[g1],1) - np.sum(A[i,Z[g2]==1] )/np.maximum(Ng[g2],1) for g2 in np.delete(np.arange(G),g1)] ) ) for g1 in range(G)] ) for i in range(N)] )
    elif bias_type=='nonsmooth':
        # Unbiased if attribute vector is nonsmooth
        L = np.diag(np.sum(A,axis=0)) - A
        return -np.trace(Z@L@Z.T)
    else:
        print('Invalid bias type.')
        return
    
def compute_pcc(x1,x2):
    assert len(x1)==len(x2), 'Inconsistent number of samples.'
    return np.sum((x1-np.mean(x1))*(x2-np.mean(x2)))/np.sqrt(np.sum((x1-np.mean(x1))**2)*np.sum((x2-np.mean(x2))**2))

def compute_tpcc(X1,x2):
    assert X1.shape[0]==len(x2), 'Inconsistent number of samples.'
    assert X1.ndim==2, 'Invalid data matrix.'
    return np.mean([np.abs(compute_pcc(x1,x2)) for x1 in X1.T])

def compute_inv_err(Theta_hat, Sigma):
    (p,p) = Theta_hat.shape
    return la.norm( Theta_hat@Sigma - np.eye(p), 'fro' )**2

def compute_frob_err(Theta_hat, Theta, pre_norm=False):
    norm_Theta = la.norm(Theta,'fro') if la.norm(Theta,'fro') else 1
    if pre_norm:
        norm_Theta_hat = la.norm(Theta_hat,'fro') if la.norm(Theta_hat,'fro') else 1
        return la.norm( Theta_hat/norm_Theta_hat - Theta/norm_Theta, 'fro' )**2
    else:
        return (la.norm( Theta_hat - Theta, 'fro' )/norm_Theta)**2

def compute_f1_score(Theta_hat, Theta, eps_thresh=.1):
    Theta_hat = Theta_hat / ( Theta_hat.max() + int(Theta_hat.max()==0) )
    Theta_hat = ( np.abs(Theta_hat)>eps_thresh ).astype(int)
    Theta = Theta / ( Theta.max() + int(Theta.max()==0) )
    Theta = ( np.abs(Theta)>eps_thresh ).astype(int)
    return f1_score( Theta_hat.flatten(), Theta.flatten() )

def create_Z(p, group_prop):
    assert np.sum(group_prop) == 1
    g = len(group_prop)
    nodes_per_group = [int(p*prop) for prop in group_prop]
    
    Z = np.zeros((g, p))
    cont = 0
    for i, n_nodes in enumerate(nodes_per_group):
        if i == g - 1:
            Z[i,cont:] = 1
        else:
            Z[i,cont:n_nodes+cont] = 1
            cont += n_nodes

    return Z