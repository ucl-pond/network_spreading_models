from src.network_diffusion_model import NDM
import numpy as np
from src.find_optimal_timepoint import find_optimal_timepoint
from scipy.integrate import solve_ivp
from scipy import optimize
from scipy.optimize import minimize
from joblib import Parallel, delayed

class FKPP(NDM):
    def __init__(self, connectome_fname, gamma, t, ref_list, alpha=None, seed_region=None, x0=None, weights=None,
                 connectome_array=None, cortical_idx=None):
        
        super().__init__(connectome_fname, gamma, t, ref_list, seed_region, x0, connectome_array, cortical_idx)
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
            rtol=1e-3, # minimum value to treat as 0 change in gradient, if takes too long to run can 1e-3 
            atol=1e-3, # minimum value to treat as 0 change in gradient, if takes too long to run can 1e-3 
        )
 
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")
 
        # Extract solution 
        x_t = sol.y
 
        return x_t[self.cortical_idx]
    
    
    def optimise_alpha(self, target_data, n_iter=100, T=0.1):
        '''
        optimise alpha parameter for fkpp model
        alpha is optimised for a given seed region
        uses scipy basinhopping to avoid local minima
        '''
        def objective(alpha):
            fkpp = FKPP(connectome_fname = self.connectome_fname,
                gamma = self.gamma,
                t = self.t,
                ref_list=self.ref_list,
                alpha=alpha,
                seed_region=self.seed_region,
                weights=self.weights,
                connectome_array=self.connectome_array,
                cortical_idx=self.cortical_idx
                )
            
            model_output = fkpp.run_FKPP()
            min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
            return SSE

        minimizer_kwargs = {"method": "L-BFGS-B",
                            "bounds": [(0, 1)]} # bounds for alpha
        result = optimize.basinhopping(objective,
                                        x0=0.5,
                                        niter=n_iter,
                                        stepsize=0.1,
                                        T=T,
                                        minimizer_kwargs=minimizer_kwargs)
        best_alpha = result.x[0]
        return best_alpha

    def optimise_fkpp(self, target_data, n_iter=100, T=0.1, seed_list=None):
        '''
        optimise parameters region for fkpp model
        alpha is optimised for each seed, and the best seed is selected
        according to the highest correlation with target data

        n_iter: number of iterations for basinhopping
        T: temperature for basinhopping
        seed_list: list of seed regions to optimise over, if None, use all regions
        '''

        if seed_list is not None:
            regions = seed_list
            regions_check = self.get_regions() # getting the bilateral regions from the ref_list
            # check that the regions are in the connectome
            for region in regions:
                if region not in regions_check:
                    raise ValueError(f"seed region {region} not in reference list")

        else:
            regions = self.get_regions()


        def evaluate_region(region):
            fkpp = FKPP(connectome_fname=self.connectome_fname,
                        gamma=self.gamma,
                        t=self.t,
                        ref_list=self.ref_list,
                        seed_region=region,
                        weights=self.weights,
                        connectome_array=self.connectome_array,
                        cortical_idx=self.cortical_idx
                        )
            alpha = fkpp.optimise_alpha(target_data)
            fkpp.alpha = alpha
            model_output = fkpp.run_FKPP()
            min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
            r = np.corrcoef(target_data, prediction)[0, 1]
            return region, alpha, r

        results = Parallel(n_jobs=-1)(delayed(evaluate_region)(region) for region in regions)

        # Find the best result
        best_region, best_alpha, best_r = max(results, key=lambda x: x[2])
        return best_region, best_alpha, best_r

    
            