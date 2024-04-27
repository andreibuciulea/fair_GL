from utils import *
from opt import *
import commentjson as json
import os
import pandas as pd

norm_scale = lambda A:np.sum(A[0])

# -----------------------------------

class NTIExperiment:
    def __init__(self,
                 A=None,
                 Z=None,
                 verbose:bool=False):
        assert (A is not None) or (Z is not None), 'Provide at least A or Z.'
        if A is not None:
            assert A.shape[0]==A.shape[1], 'Invalid adjacency matrix.'
        if Z is not None and A is not None:
            assert Z.shape[1]==A.shape[0], 'Inconsistent number of nodes.'

        # ---------------------------
        self.A = A
        self.Z = Z
        self.N = A.shape[0] if A is not None else Z.shape[1]
        self.G = Z.shape[0] if Z is not None else None
        self.Ng = np.sum(Z,axis=1).astype(int) if Z is not None else None
        self.verbose = verbose
        # ---------------------------


        # ---------------------------
        for key in ['C','A_est']:
            setattr(self, key, None)
        # ---------------------------
    def set_Z(self,Z):
        if self.A is not None:
            assert Z.shape[1]==self.A.shape[0], 'Inconsistent number of nodes.'
        else:
            self.N = Z.shape[1]
        self.Z = Z
        self.G = Z.shape[0]
        self.Ng = np.sum(Z,axis=1).astype(int)
    def set_C(self,C):
        assert (C is not None and C.shape[0]==C.shape[1] and all(np.abs(np.linalg.eigvalsh(C))>=1e-9)), 'Invalid covariance matrix.'
        self.C = C
    def generate_C(self,
                   M:int=1000,
                   L:int=3,
                   signal_model:str='poly'
                   ):
        assert M>0, 'Invalid number of samples.'
        assert signal_model in ['gmrf','poly'], 'Invalid graph signal model.'
        assert L>0 or signal_model!='poly', 'Invalid filter order.'
        
        # ---------------------------
        if signal_model=='gmrf':
            C = create_gmrf_cov(A=self.A)
        elif signal_model=='poly':
            H = create_filter(A=self.A,L=L)
            C = create_poly_cov(H=H)
        else:
            assert False, print('Invalid graph signal model.')
        # ---------------------------

        # ---------------------------
        if signal_model=='gmrf':
            X = gmrf_samples(C=C,M=M)
        elif signal_model=='poly':
            X = poly_samples(H=H,M=M)
        else:
            assert False, print('Invalid graph signal model.')
        self.C = est_cov(X=X)
        # ---------------------------

    def est_A(self,
              params:dict,
              REWEIGHTED:bool=False,
              UPPER_BOUND:bool=False,
              BIAS_PENALTY:bool=False,
              C=None
              ):
        # ---------------------------
        assert 'alpha' in params.keys(), 'Missing parameter alpha.'
        assert (C is None and hasattr(self,'C')) or (C is not None and C.shape[0]==C.shape[1] and all(np.abs(np.linalg.eigvalsh(C))>=1e-9)), 'No valid covariance matrix found.'

        assert (not BIAS_PENALTY) or (self.Z is not None), 'Missing group assignments for fair network inference.'
        assert (not BIAS_PENALTY) or (all([key in params.keys() for key in ['beta','bias_type']])), 'Bias mitigation missing beta or bias_type parameters.'

        assert (not REWEIGHTED) or (all([key in params.keys() for key in ['delta','eps_thresh','verbose']])), 'Reweighting missing delta, eps_thresh or verbose parameters.'

        assert (not UPPER_BOUND) or (all([key in params.keys() for key in ['eps_init','factor_eps','max_iters_eps']])), 'Upper bound constraint missing eps_init, factor_eps, or max_iter_eps parameters.'
        assert (UPPER_BOUND) or (all([key in params.keys() for key in ['mu']])), 'Penalty missing mu parameter.'
        # ---------------------------
        
        
        # ---------------------------
        if C is None:
            C = self.C
        # ---------------------------
        
        
        # ---------------------------
        gsr_args = {
            'C':C,
            'alpha':params['alpha']
        }
        if REWEIGHTED:
            gsr_args = gsr_args | {
                'delta':params['delta'],
                'eps_thresh':params['eps_thresh'],
                'verbose':params['verbose']
            }
        if not UPPER_BOUND:
            gsr_args = gsr_args | {
                'mu':params['mu']
            }
        if BIAS_PENALTY:
            gsr_args = gsr_args | {
                'Z':self.Z,
                'beta':params['beta'],
                'bias_type':params['bias_type']
            }
        # ---------------------------


        # ---------------------------
        if REWEIGHTED and BIAS_PENALTY:
            EstGraph = FGSR_reweighted
        elif REWEIGHTED:
            EstGraph = GSR_reweighted
        elif BIAS_PENALTY:
            EstGraph = FGSR
        else:
            EstGraph = GSR
        # ---------------------------
        

        # ---------------------------
        if UPPER_BOUND:
            gsr_args['eps'] = params['eps_init']
            for i_eps in range(params['max_iters_eps']):
                A_est = EstGraph(**gsr_args)
                if A_est is not None:
                    break
                gsr_args['eps'] *= params['factor_eps']
                # print(f"Upper bound too small. Increasing to {gsr_args['eps']:.3e}")
        else:
            A_est = EstGraph(**gsr_args)
        if A_est is None:
            print('Problem did not converge. Proceed with caution.')
            self.A_est = None
            return
        self.A_est = A_est
        # ---------------------------

    def compute_frob_error(self):
        assert hasattr(self,'A') and self.A is not None, 'No true adjacency matrix found.'
        assert hasattr(self,'A_est') and (self.A_est is not None), 'Missing estimated network.'
        return compute_frob_error(self.A_est,self.A)

    def compute_f1_score(self):
        assert hasattr(self,'A') and self.A is not None, 'No true adjacency matrix found.'
        assert hasattr(self,'A_est') and (self.A_est is not None), 'Missing estimated network.'
        return compute_f1_score(self.A_est,self.A)

    def compute_est_bias(self,bias_type:str='dp'):
        assert hasattr(self,'A_est') and (self.A_est is not None), 'Missing estimated network.'
        assert (self.Z is not None), 'Missing group assignments.'
        A_norm = self.A_est/norm_scale(self.A_est) if norm_scale(self.A_est) else np.zeros_like(self.A_est)
        return compute_bias(A_norm,self.Z,bias_type=bias_type)
        
    def compute_true_bias(self,bias_type:str='dp'):
        assert (self.Z is not None), 'Missing group assignments.'
        A_norm = self.A/norm_scale(self.A) if norm_scale(self.A) else np.zeros_like(self.A)
        return compute_bias(A_norm,self.Z,bias_type=bias_type)

    def copy_experiment(self):
         exp2 = NTIExperiment(A=self.A, Z=self.Z, verbose=self.verbose)
         exp2.set_C(self.C)
         exp2.A_est = self.A_est.copy() if self.A_est is not None else None
         return exp2
    
    def beta_bias_gridsearch(self,beta_range,
                             params:dict,
                             REWEIGHTED:bool=False,
                             UPPER_BOUND:bool=False,
                             bias_type:str='dp'):
        assert (self.Z is not None), 'Missing group assignments.'
        bias_range = np.zeros(len(beta_range))

        bias_opt = np.inf
        A_opt = None
        beta_opt = None
        for i,beta in enumerate(beta_range):
            params['beta'] = beta
            self.est_A(params,REWEIGHTED=REWEIGHTED,UPPER_BOUND=UPPER_BOUND,BIAS_PENALTY=True)
            bias = self.compute_est_bias(bias_type)
            bias_range[i] = bias
            if bias < bias_opt:
                bias_opt = bias
                A_opt = self.A_est.copy()
                beta_opt = beta
        
        self.A_est = A_opt.copy()
        return beta_opt, bias_range

    def param_gridsearch(self,
                         alpha_range,
                         beta_range,
                         params:dict,
                         REWEIGHTED:bool=False,
                         UPPER_BOUND:bool=False,
                         frob_err:bool=True,
                         bias_weight=1,
                         bias_type:str='dp'):
        assert (self.Z is not None), 'Missing group assignments.'
        assert (self.A is not None), 'Missing true adjacency matrix.'
        err_range = np.zeros((len(alpha_range),len(beta_range)))
        bias_range = np.zeros((len(alpha_range),len(beta_range)))

        err_bias_opt = np.inf
        A_opt = None
        alpha_opt = None
        beta_opt = None
        for i,alpha in enumerate(alpha_range):
            for j,beta in enumerate(beta_range):
                params['alpha'] = alpha
                params['beta'] = beta
                self.est_A(params,REWEIGHTED=REWEIGHTED,UPPER_BOUND=UPPER_BOUND,BIAS_PENALTY=True)
                err = self.compute_frob_error() if frob_err else 1-self.compute_f1_score(eps_thresh=.2)
                bias = self.compute_est_bias(bias_type)
                err_range[i,j] = err
                bias_range[i,j] = bias
                if err + bias_weight*bias < err_bias_opt:
                    err_bias_opt = err + bias_weight*bias
                    A_opt = self.A_est.copy()
                    alpha_opt = alpha
                    beta_opt = beta
        
        self.A_est = A_opt.copy()
        return alpha_opt, beta_opt, err_range, bias_range


        

        
        


# -----------------------------------

def load_senate(senate_number:int):
    members = pd.read_csv(f'senate data/S{str(senate_number).zfill(3)}_members.csv')
    parties = pd.read_csv(f'senate data/S{str(senate_number).zfill(3)}_parties.csv')
    rollcalls = pd.read_csv(f'senate data/S{str(senate_number).zfill(3)}_rollcalls.csv')
    votes = pd.read_csv(f'senate data/S{str(senate_number).zfill(3)}_votes.csv')

    num_votes = rollcalls.shape[0]

    party_code = np.array(members['party_code'])
    state_icpsr = np.array(members['state_icpsr'])
    icpsr = np.array(members['icpsr'])

    icpsr_to_party = dict(zip(icpsr,party_code))
    icpsr_to_state = dict(zip(icpsr,state_icpsr))
    castcode_to_vote = { 9:0, 7:0, 1:1, 6:-1 }
    def party_to_color(code):
        if code==100:
            return 0
        elif code==200:
            return 1
        else:
            return 2

    votes_per_rollcall = [votes.groupby('rollnumber').get_group(i_vote+1) for i_vote in range(num_votes)]

    state_list = []
    votes_list = []
    nodes = []
    for i_vote in range(num_votes):
        mat = np.array(votes_per_rollcall[i_vote][['icpsr','cast_code']])
        states_curr = np.array(list(map(lambda x:icpsr_to_state[x],mat[:,0])))
        votes_curr = np.array(list(map(lambda x:castcode_to_vote[x],mat[:,1])))
        parties_curr = np.array(list(map(lambda x:icpsr_to_party[x],mat[:,0])))

        state_inds = [np.where(states_curr==s)[0] for s in np.unique(states_curr)]

        state_list.append( np.unique(states_curr) )
        votes_list.append( np.array([np.mean(votes_curr[s_inds]) for s_inds in state_inds]) )
        nodes.append( list( map( party_to_color, np.array([np.mean(parties_curr[s_inds]) for s_inds in state_inds]) ) ) )

    X = np.array(votes_list).T
    z = nodes[0]
    G = len(np.unique(z))
    Z = np.eye(G)[z].T
    N = X.shape[1]
    M = X.shape[0]
    C_est = est_cov(X=X)

    return X,Z,C_est

def load_params(config_name:str):
    path = os.getcwd()
    config_path = path + '/' + config_name
    assert os.path.exists(config_path), "Config file does not exist."
    with open(config_path,'r') as f:
        params = json.load(f)
    
    assert type(params['N'])==int and params['N']>0, "Invalid number of nodes."
    assert type(params['G'])==int and params['G']>0, "Invalid number of groups."
    assert type(params['L'])==int and params['L']>0, "Invalid filter order."
    assert type(params['M'])==int and params['M']>0, "Invalid number of signals."

    assert type(params['max_iters_eps'])==int and params['max_iters_eps']>0, "Invalid number of iterations."
    for key in ['alpha','beta','delta','eps_thresh','mu','eps_init','factor_eps']:
        assert params[key]>0, f'Invalid parameter {key}.'
    
    assert params['opt_bias_type'] in ['dp','dp_scaled','global','groupwise','tot_corr','nodewise'], 'Invalid bias penalty for optimization.'
    assert params['eval_bias_type'] in ['dp','dp_scaled','ratio','global','groupwise','tot_corr','nodewise','nonsmooth'], 'Invalid bias penalty for evaluation.'

    assert (params['edge_prob']<=1 and params['edge_prob']>=0 and 
            params['in_prob']<=1 and params['in_prob']>=0 and 
            params['out_prob']<=1 and params['out_prob']>=0), 'Invalid edge probability.'
    assert type(params['deg'])==int and params['deg']>0, "Invalid node degree."
    return params

