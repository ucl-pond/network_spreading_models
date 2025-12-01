from src.FKPP_model import FKPP
import numpy as np
from scipy.integrate import solve_ivp
from scipy import optimize
from src.find_optimal_timepoint import find_optimal_timepoint
from joblib import Parallel, delayed
import pandas as pd


class IWFKPP(FKPP):
    def __init__(self, connectome_fname, gamma, t, ref_list, alpha=None, gamma_amyloid=None,
                 amyloid_data=None, seed_region=None, x0=None, connectome_array=None, 
                 cortical_idx=None, lateral_seeding=False):
        """
        Interaction-Weighted FKPP Model extending FKPP with bounded formulation:
        dx_tau(t)/dt = -alpha * H * x_tau(t) + (1-alpha) * x_tau(t) * (1-x_tau(t)) * [(1-gamma_amyloid) + gamma_amyloid * x_Abeta(t)]
        
        Parameters:
        -----------
        gamma_amyloid : float, optional
            Coupling parameter that determines maximum amyloid influence (0≤gamma_amyloid≤1)
        amyloid_data : array-like, optional
            Regional amyloid deposition data (same order as ref_list)
        """
        super().__init__(connectome_fname, gamma, t, ref_list, alpha, seed_region, x0, 
                         connectome_array=connectome_array, cortical_idx=cortical_idx, 
                         lateral_seeding=lateral_seeding)
        self.gamma_amyloid = gamma_amyloid if gamma_amyloid is not None else 0.0
        self.amyloid_data = amyloid_data if amyloid_data is not None else np.zeros(len(ref_list))
        
    def run_model(self):
        """Run the bounded Interaction-Weighted FKPP Model."""
        H = self.get_Laplacian()
        amyloid_effect = (1.0 - self.gamma_amyloid) + self.gamma_amyloid * self.amyloid_data
        
        # Precompute constant coefficient for diffusion
        diffusion_coef = self.alpha * (-self.gamma)
        growth_coef = 1 - self.alpha
        
        def system(t, y):
            # Use precomputed coefficients
            diffusion = diffusion_coef * (H @ y)
            logistic = growth_coef * y * (1-y) * amyloid_effect
            return diffusion + logistic
        
        initial_state = self.get_initial_conditions()
        
        result = solve_ivp(system, [0, self.t[-1]], 
                           initial_state,
                           t_eval=self.t, method='RK45',
                           rtol=1e-6,
                           atol=1e-6)
        
        return result.y[self.cortical_idx, :]
    
    def optimise_alpha_gamma(self, target_data, n_iter=100, T=0.1):
        """
        Optimise alpha and gamma_amyloid parameters for IWFKPP model.
        Uses scipy basinhopping to avoid local minima.
        
        Parameters:
        -----------
        target_data : array-like
            Target data to fit the model to
        n_iter : int
            Number of iterations for basinhopping
        T : float
            Temperature for basinhopping
            
        Returns:
        --------
        tuple : (best_alpha, best_gamma_amyloid)
        """
        def objective(params):
            alpha, gamma_amyloid = params
            model = IWFKPP(
                connectome_fname=self.connectome_fname,
                gamma=self.gamma,
                t=self.t,
                ref_list=self.ref_list,
                alpha=alpha,
                gamma_amyloid=gamma_amyloid,
                amyloid_data=self.amyloid_data,
                seed_region=self.seed_region,
                connectome_array=self.connectome_array,
                cortical_idx=self.cortical_idx,
                lateral_seeding=self.lateral_seeding
            )
            model_output = model.run_model()
            min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
            return SSE

        minimizer_kwargs = {
            "method": "L-BFGS-B",
            "bounds": [(0, 1), (0, 1)]  # bounds for alpha and gamma
        }
        result = optimize.basinhopping(
            objective,
            x0=[0.5, 0.5],
            niter=n_iter,
            stepsize=0.1,
            T=T,
            minimizer_kwargs=minimizer_kwargs
        )
        best_alpha = result.x[0]
        best_gamma_amyloid = result.x[1]
        return best_alpha, best_gamma_amyloid

    def optimise_iwfkpp_params(self, target_data, n_iter=100, T=0.1, seed_list=None):
        """
        Optimise parameters for IWFKPP model.
        Alpha and gamma_amyloid are optimised for each seed, and the best seed is selected
        according to the lowest SSE with target data.
        
        Parameters:
        -----------
        target_data : array-like
            Target data to fit the model to
        n_iter : int
            Number of iterations for basinhopping
        T : float
            Temperature for basinhopping
        seed_list : list, optional
            List of potential seed regions to consider
            
        Returns:
        --------
        tuple : (results DataFrame, optimal_params dict)
        """
        if seed_list is not None:
            regions = seed_list
            regions_check = self.get_regions()
            for region in regions:
                if region not in regions_check:
                    raise ValueError(f"seed region {region} not in reference list")
        elif not self.lateral_seeding:
            regions = self.get_regions()
        else:
            regions = self.ref_list

        def evaluate_region(region):
            model = IWFKPP(
                connectome_fname=self.connectome_fname,
                gamma=self.gamma,
                t=self.t,
                ref_list=self.ref_list,
                seed_region=region,
                amyloid_data=self.amyloid_data,
                connectome_array=self.connectome_array,
                cortical_idx=self.cortical_idx,
                lateral_seeding=self.lateral_seeding
            )
            alpha, gamma_amyloid = model.optimise_alpha_gamma(target_data, n_iter=n_iter, T=T)
            model.alpha = alpha
            model.gamma_amyloid = gamma_amyloid
            model_output = model.run_model()
            min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
            r = np.corrcoef(target_data, prediction)[0, 1]
            return region, alpha, gamma_amyloid, r, SSE

        res = Parallel(n_jobs=-1)(delayed(evaluate_region)(region) for region in regions)

        # Find the best result (minimum SSE)
        optimal_params = {}
        best_result = min(res, key=lambda x: x[4])
        optimal_params["seed"] = best_result[0]
        optimal_params["alpha"] = best_result[1]
        optimal_params["gamma_amyloid"] = best_result[2]
        optimal_params["r"] = best_result[3]
        optimal_params["SSE"] = best_result[4]

        res_df = pd.DataFrame(res, columns=["seed_region", "alpha", "gamma_amyloid", "r", "SSE"])
        return res_df, optimal_params