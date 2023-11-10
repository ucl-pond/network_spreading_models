from network_diffusion_model import NDM
import numpy as np

class FKPP_class(NDM):
    def __init__(self, connectome_fname, gamma, t, seed_region, ref_list, alpha):
        super().__init__(connectome_fname, gamma, t, seed_region, ref_list)
        self.alpha = alpha # logistic growth rate

    def logistic_model(self, x):
        return x*(1-x)

    def run_FKPP(self):

        H = self.get_Laplacian()

        #loop through time points, estimating tau accumulation at each point
        x_t = np.empty([len(H),self.Nt])
        x_t[:] = 0

        x_t[:,0] = self.get_initial_conditions() # set first time point to initial conditions.

        for kt in range(1,self.Nt):  #iterate through time points, calculating the node atrophy as you go along
            x_t[:,kt] = x_t[:,kt-1] + self.alpha*self.NDM_dx(H,x_t[:,kt-1]) + (1-self.alpha)*self.logistic_model(x_t[:,kt-1])*self.dt

        return x_t