from network_diffusion_model import NDM
import numpy as np

class FKPP_class(NDM):
    def __init__(self, connectome_fname, gamma, t, seed_region, ref_list, alpha):
        super().__init__(connectome_fname, gamma, t, seed_region, ref_list)
        self.alpha = alpha # logistic growth rate

    def logistic_model(self, x):
        return x*(1-x)

    def run_FKPP(self):

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

        x_t[:,0] = self.get_initial_conditions() # set first time point to initial conditions.

        for kt in range(1,Nt):  #iterate through time points, calculating the node atrophy as you go along
            x_t[:,kt] = x_t[:,kt-1] + self.alpha*self.NDM_dx(H,x_t[:,kt-1],dt) + (1-self.alpha)*self.logistic_model(x_t[:,kt-1])*dt

        return x_t