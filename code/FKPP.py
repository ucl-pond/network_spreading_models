from network_diffusion_model import NDM
import numpy as np
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from find_optimal_timepoint import find_optimal_timepoint

class FKPP_class(NDM):
    def __init__(self, connectome_fname, gamma, t, ref_list, alpha=None, seed_region=None, x0=None, production_weights=None, spreading_weights=None):
        super().__init__(connectome_fname, gamma, t, ref_list, seed_region, x0)
        self.alpha = alpha # logistic growth rate
        self.production_weights = production_weights # weighting for logistic growth term
        self.spreading_weights = spreading_weights # weighting for the connectome spread

    def logistic_model(self, x):
        return x*(1-x)

    def run_FKPP(self):
        '''
        run the FKPP with optional weighting on the logistic model term.
        e.g. vector with amyloid SUVR
        '''

        #if weights is none use a vector of ones
        if self.production_weights is None:
            self.production_weights = np.ones(len(self.ref_list))

        if self.spreading_weights is None:
            self.spreading_weights = np.ones(len(self.ref_list))

        H = self.get_Laplacian()

        #loop through time points, estimating tau accumulation at each point
        x_t = np.empty([len(H),self.Nt])
        x_t[:] = 0

        x_t[:,0] = self.get_initial_conditions() # set first time point to initial conditions.

        for kt in range(1,self.Nt):  #iterate through time points, calculating the node atrophy as you go along
            x_t[:,kt] = x_t[:,kt-1] \
                + self.alpha*self.NDM_dx(H,self.spreading_weights*x_t[:,kt-1]) \
                + (1-self.alpha)*self.logistic_model(x_t[:,kt-1])*self.production_weights*self.dt

        return x_t
    
    def optimise_fkpp(self, target_data, n_calls=200, n_initial_points=128):
        '''
        optimise seed and alpha parameter for fkpp model
        '''
        regions = self.get_regions()

        space  = [Categorical(regions, name ='seed_region'),
                  Real(0, 1, name='alpha')]
        
        @use_named_args(space)
        def objective(**params):
            fkpp = FKPP_class(connectome_fname = self.connectome_fname,
                gamma = self.gamma,
                t = self.t,
                ref_list=self.ref_list,
                alpha=params["alpha"],
                seed_region=params["seed_region"],
                production_weights=self.production_weights,
                spreading_weights=self.spreading_weights
            )
            model_output = fkpp.run_FKPP()
            min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
            return SSE

        res = gp_minimize(objective, dimensions=space,
                        acq_func="gp_hedge",
                        n_calls=n_calls,
                        n_initial_points=n_initial_points,
                        random_state=42,
                        initial_point_generator="sobol"
                        )
        
        optimal_params = {}
        optimal_params["seed"] = res["x"][0]
        optimal_params["alpha"] = res["x"][1]

        return res, optimal_params