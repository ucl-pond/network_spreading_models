import numpy as np
import pandas as pd
from find_optimal_timepoint import find_optimal_timepoint, mysse
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize

class NDM():
    """
    class containing the paramters and implimenting the Network diffusion model
    """
    def __init__(self, connectome_fname, gamma, t, ref_list, seed_region=None, x0=None):
        '''
        inputs: connectome_fname = filename of connectome
                gamma = diffusivity constant
                t = list of time points at which to predict accumulation
                seed_region = region to use as seed for NDM
                ref_list = list of regions, in same order as connectome and pathology data. 
                            assuming they are right and left hemispheres (need to be subscripted with _L and _R)
        '''
        self.connectome_fname = connectome_fname
        self.gamma = gamma
        self.t = t
        self.seed_region = seed_region
        self.x0 = x0
        self.ref_list = ref_list
        self.dt = self.t[1]-self.t[0]
        self.Nt = len(self.t)
 
    def seed2idx(self):
        '''
        Parameters:
            seed: string eg. "Entorhinal"
            ref_list: list containing the regions in the order used in the data
        Returns:
            seed_l_ind: index of left seed
            seed_r_ind: index of right seed

        '''
        seed_l_ind = self.ref_list.index(self.seed_region + "_L")
        seed_r_ind = self.ref_list.index(self.seed_region + "_R")
        return (seed_l_ind, seed_r_ind)

    def get_initial_conditions(self):
        if self.x0 is not None:
            return self.x0
        seed_l_ind, seed_r_ind = self.seed2idx()
        x_0 = np.zeros(len(self.ref_list))
        x_0[seed_l_ind] = 1
        x_0[seed_r_ind] = 1
        return x_0


    def prep_connectome(self):
        '''
        load connectome and ensure it is symmetrical
        '''
        with open(self.connectome_fname) as f:
            first_line = f.readline()
            if "," in first_line:
                delimiter = ","
            elif "\t" in first_line:
                delimiter = "\t"
            elif " " in first_line:
                delimiter = " "
            else:
                raise ValueError("Delimiter not found")        
        C = np.loadtxt(self.connectome_fname,delimiter=delimiter)

        # check connectome is 2D square
        assert C.shape[0] == C.shape[1]
        assert len(C.shape) ==2

        # make symmetric matrix
        return np.triu(C,1)+np.tril(C.T)
    
    def NDM_dx(self,H,x):

        return (-self.gamma * (H @ x)) * self.dt

    def get_Laplacian(self):
        '''
        get Laplacian matrix from connectome
        '''
        C = self.prep_connectome()
        rowdegree = np.sum(C, 1)
        D = np.diag(np.sum(C,0))
        H = D - C                           # this does not include self-connections in the Laplacian
        H = np.diag(1/(rowdegree+np.finfo(float).eps)) @ H
        return H
    
    def run_NDM(self):
        '''
        run the NDM network diffusion model by Raj et al.
        inputs: C = connectome,
                x0 = initial tau accumulation in each region,
                gamma = diffusivity constant,
                t = list of time points at which to predict accumulation
        output:
        '''

        H = self.get_Laplacian()
        
        #loop through time points, estimating tau accumulation at each point
        x_t = np.empty([len(H),self.Nt])
        x_t[:] = 0

        x_t[:,0] = self.get_initial_conditions() # set first time point to initial conditions.

        for kt in range(1,self.Nt):  #iterate through time points, calculating the node atrophy as you go along
                x_t[:,kt] = x_t[:,kt-1] + self.NDM_dx(H,x_t[:,kt-1])

        return x_t/np.max(x_t,axis=0)
    
    def get_regions(self):
        '''
        get the bilateral regions from the reference list
        assumes that the entries in the reference list are of the form "region_L" and "region_R"
        '''
        #assert that entries in ref_list are of the form "region_L" and "region_R"
        assert all([r[-2:]=="_L" or 
                    r[-2:]=="_R" or 
                    r[-2:]=="_r" or
                    r[-2:]=="_l" for r in self.ref_list])
        
        regions = [r[:-2] for r in self.ref_list]
        return list(set(regions))
    
    def optimise_seed_region(self, target_data):
        '''
        optimise the seed region for the NDM model
        '''
        regions = self.get_regions()
        SSE = np.zeros(len(regions))

        for i,r in enumerate(regions):
            ndm = NDM(connectome_fname=self.connectome_fname,
                        gamma=self.gamma,
                        t=self.t,
                        seed_region=r,
                        ref_list=self.ref_list
                        )
            model_output = ndm.run_NDM()
            min_idx, prediction, SSE[i] = find_optimal_timepoint(model_output, target_data)
        
        optimal_params = {}
        optimal_params["seed"] = regions[np.argmin(SSE)]
        res = pd.DataFrame({"seed":regions, "SSE":SSE})
        
        return res, optimal_params

    
