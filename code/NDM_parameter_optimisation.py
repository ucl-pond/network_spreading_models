import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from network_diffusion_model import NDM
from network_diffusion_model import NDM
from find_optimal_timepoint import find_optimal_timepoint, mysse
import json

from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.plots import plot_convergence, plot_objective

np.int = np.int64

data_dir = "/Users/ellie/Documents/UCL_cloud/cmic-hacks-2023/network_spreading_models/simulated_data"
ref_list = pd.read_csv(data_dir + "/TauRegionList.csv")["Raj_label"].tolist()

target_data_path = data_dir + "/FKPP_invivo_tractography.csv"

target_data = pd.read_csv(target_data_path, header=None).values
target_data = target_data[:,500]

regions = [r[:-2] for r in ref_list]
regions = list(set(regions))

connectome_fname = data_dir + "/tractography.csv"

results_dir = "/Users/ellie/Documents/UCL_cloud/cmic-hacks-2023/network_spreading_models/results/"
results_name = "NDM_optimisation"

t = np.arange(0, 100, 1)


space  = [Categorical(regions, name ='seed_region')]


@use_named_args(space)
def objective(**params):
        ndm = NDM(connectome_fname = connectome_fname,
            gamma = 0.01, # fixing gamma at this value for now
            t = t,
            seed_region=params["seed_region"],
            ref_list=ref_list
          )
        model_output = ndm.run_NDM()
        min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)
        return SSE

res = gp_minimize(objective, dimensions=space, 
                  acq_func="gp_hedge", 
                  n_calls=200, 
                  n_initial_points=50,
                  random_state=42,
                  initial_point_generator="sobol"
                  )

plt.figure()
plot_convergence(res)

plt.figure()
plot_objective(res)

optimal_params = {}
optimal_params["seed"] = res["x"][0]

print(f"optimal seed = {optimal_params['seed']}")

# run with the optimal parameters to get the prediction accuracy
ndm_optimal = NDM(connectome_fname = connectome_fname,
            gamma = 0.01,
            t = t,
            seed_region=optimal_params["seed"],
            ref_list=ref_list
          )

model_output = ndm_optimal.run_NDM()
min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)

plt.figure()
plt.scatter(target_data, prediction)
plt.xlabel("Target data")
plt.ylabel("Model prediction")

print(f"Pearson's correlation coefficient: {np.corrcoef(prediction, target_data)[0,1]}")

# find the model residuals for model comparison
residuals = target_data - prediction
np.savetxt(results_dir + "/" + results_name + "_residuals.csv", residuals, delimiter=",")
np.savetxt(results_dir + "/" + results_name + "_prediction.csv", prediction, delimiter=",")

with open(results_dir + "/" + results_name + "params.txt", "w") as fp:
    json.dump(optimal_params, fp)  # encode dict into JSON
print("Done writing dict into .txt file")