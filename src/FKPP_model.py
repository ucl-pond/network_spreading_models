from src.network_diffusion_model import NDM
import numpy as np
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from src.find_optimal_timepoint import find_optimal_timepoint
from scipy.integrate import solve_ivp

class FKPP(NDM):
    def __init__(self, connectome_fname, gamma, t, ref_list, alpha=None, seed_region=None, x0=None, weights=None):
        super().__init__(connectome_fname, gamma, t, ref_list, seed_region, x0)
        self.alpha = alpha # logistic growth rate
        self.weights = weights # weighting for logistic growth term

    def logistic_model(self, x):
        return x*(1-x)

    def run_FKPP(self):
        '''
        run the FKPP with optional weighting on the logistic model term.
        e.g. vector with amyloid SUVR
        '''

        #if weights is none use a vector of ones
        if self.weights is None:
            self.weights = np.ones(len(self.ref_list))

        H = self.get_Laplacian()

        # define the ODE system
        def fkpp_ode(t,x):
            diffusion = self.NDM_dx(H, x)
            logistic = self.logistic_model(x) * self.weights
            return (self.alpha * diffusion) + ((1 - self.alpha) * logistic)
        
        # initial conditions
        x0_full = self.get_initial_conditions()

        # solve the ODE system
        sol = solve_ivp(
            fun=fkpp_ode,
            t_span=(self.t[0], self.t[-1]),
            y0=x0_full,
            method='RK45',
            t_eval=self.t,
            vectorized=False,
            rtol=1e-6, # minimum value to treat as 0 change in gradient, if takes too long to run can 1e-3 
            atol=1e-6, # minimum value to treat as 0 change in gradient, if takes too long to run can 1e-3 
        )
 
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
 
        # Extract solution 
        x_t = sol.y
 
        return x_t
    
    def optimise_fkpp(self, target_data, n_calls=200, n_initial_points=64):
        '''
        optimise seed and alpha parameter for fkpp model
        increasing n_calls and n_initial points can improve model performance in some cases, but will slow down the optimisation.

        '''
        regions = self.get_regions()

        space  = [Categorical(regions, name ='seed_region'),
                  Real(0, 1, name='alpha')]
        
        @use_named_args(space)
        def objective(**params):
            fkpp = FKPP(connectome_fname = self.connectome_fname,
                gamma = self.gamma,
                t = self.t,
                ref_list=self.ref_list,
                alpha=params["alpha"],
                seed_region=params["seed_region"],
                weights=self.weights
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
