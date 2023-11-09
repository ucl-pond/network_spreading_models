import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import expm
import matplotlib.pyplot as plt


class NDM():
    """
    class containing the paramters and implimenting the Network diffusion model
    """
    def __init__(self, connectome_fname, gamma, t, x_0):
        self.connectome_fname = connectome_fname
        #self.rnoi = self.connectome.shape[0]
        self.gamma = gamma
        self.t = t
        self.x_0 = x_0 # initial conditions
 


    def prep_connectome(self):
        '''
        load connectome and ensure it is symmetrical
        '''
        C = np.loadtxt(self.connectome_fname, delimiter=",")

        # check connectome is 2D square
        assert C.shape[0] == C.shape[1]
        assert len(C.shape) ==2

        # make symmetric matrix
        return np.triu(C,1)+np.tril(C.T)

    
    def run_NDM(self):
        '''
        run the NDM network diffusion model by Raj et al.
        inputs: C = connectome,
                x0 = initial tau accumulation in each region,
                gamma = diffusivity constant,
                t = list of time points at which to predict accumulation
        output:
        '''
        C = self.prep_connectome()
        dt = self.t[1]-self.t[0]
        Nt = len(self.t)

        rowdegree = np.sum(C, 1)
        D = np.diag(np.sum(C,0))
        H = D - C                           # this does not include self-connections in the Laplacian
        H = np.diag(1/(rowdegree+np.finfo(float).eps)) @ H  # since all brain regions are not the same size, each row and column is normalised by its sum
        # note: this normalisation prevents means the sum of tau at each time is not consistent
        # note +eps above, this prevents H containing NaNs if the rowdegree contains 0

        #loop through time points, estimating tau accumulation at each point
        x_t = np.empty([len(C),Nt])
        x_t[:] = 0

        x_t[:,0] = x_0 # set first time point to initial conditions.

        for kt in range(1,Nt):  #iterate through time points, calculating the node atrophy as you go along
                x_t[:,kt] = expm(-self.gamma*H*dt) @ x_t[:,kt-1]

        return x_t/np.max(x_t,axis=0)