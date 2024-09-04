import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.network_diffusion_model import NDM
from src.find_optimal_timepoint import find_optimal_timepoint
from skopt.plots import plot_convergence, plot_objective

'''
testing the optimisation of the NDM model on simulated data
'''
np.int = np.int64

path = Path(__file__).parent.absolute()

data_dir = path.parent / "simulated_data"
ref_list = pd.read_csv(data_dir / "TauRegionList.csv")["Raj_label"].tolist()

target_data_path = data_dir / "FKPP_invivo_tractography.csv"

target_data = pd.read_csv(target_data_path, header=None).values
target_data = target_data[:,500]

connectome_fname = data_dir / "tractography.csv"

results_dir = path.parent / "results"
results_name = "NDM_optimisation"

t = np.arange(0, 100, 1)

ndm = NDM(connectome_fname = connectome_fname,
            gamma = 0.01, # fixing gamma at this value for now
            t = t,
            ref_list=ref_list
          )

res, optimal_params = ndm.optimise_seed_region(target_data)

print(f"optimal seed = {optimal_params['seed']}")

# run with the optimal parameters to get the prediction accuracy
ndm.seed_region = optimal_params["seed"]

model_output = ndm.run_NDM()
min_idx, prediction, SSE = find_optimal_timepoint(model_output, target_data)

plt.figure()
plt.scatter(target_data, prediction)
plt.xlabel("Target data")
plt.ylabel("Model prediction")

print(f"Pearson's correlation coefficient: {np.corrcoef(prediction, target_data)[0,1]}")

# find the model residuals for model comparison
residuals = target_data - prediction
np.savetxt(results_dir / f"{results_name}_residuals.csv", residuals, delimiter=",")
np.savetxt(results_dir / f"{results_name}_prediction.csv", prediction, delimiter=",")

with open(results_dir / f"{results_name}_params.txt", "w", encoding="utf-8") as fp:
  json.dump(optimal_params, fp)  # encode dict into JSON
print("Done writing dict into .txt file")
