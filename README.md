## Network Spreading Models Toolbox
Connectome-based models of disease propagation are used to probe mechanisms of
pathology spread in neurodegenerative disease. We present our network spreading model toolbox that allows the user to compare model fits across different models and parameters.

## Available models
- <b>Network Diffusion Model (NDM)</b>
  models diffusive spread of pathology between connected brain regions ([Raj et al.](https://pubmed.ncbi.nlm.nih.gov/22445347/))
- <b>Fisher-Kolmogorov-Petrovsky-Piscounov (FKPP)</b>: 
  network spreading plus uniform local production of pathology ([Weickenmeier et al.](https://www.sciencedirect.com/science/article/pii/S0022509618307063?via%3Dihub))
- <b>Weighted-FKPP</b>: 
  spreading plus regionally-varying production weighted by a vector of choice ([He et al.](https://link.springer.com/chapter/10.1007/978-3-031-43993-3_45))

Weighting the production rates in the FKPP model by different values across the brain can improve the ability of the model to capture pathology patterns, by taking into account different regional characteristics during disease progression [1,2]. We define this new model as weighted-FKPP, and show that weighting pathology production with regional amyloid improves the model fit to tau-PET data [3].

Equations and simulated pathology over time are provided in the figure below:

![docs/models.png](docs/images/models.png)

## Get started
Run [NDM_optimisation_simulated_data.py](./code/NDM_optimisation_simulated_data.py) or [FKPP_optimisation_simulated_data.py](./code/FKPP_optimisation_simulated_data.py) to run example code for the NDM or the FKPP models. This selects optimal model parameters for each model using a simulated dataset.

## About us
This toolbox originated from project for CMICHACKS 2023.
The team included: Ellie Thompson, Anna Schroder, Tiantian He, Neil Oxtoby, James Cole, Antoine Legouhy and Xin Zhao.
  
## References
[1].	He, T., Schroder, A., Thompson, E., Oxtoby, N. P., Abdulaal, A., Barkhof, F., & Alexander, D. C. (2023). Coupled pathology appearance and propagation models of neurodegeneration. The Organization for Human Brain Mapping (OHBM) 2023 Annual Meeting, [Oral Presentation]

[2].	He, T., Thompson, E., Schroder, A., Oxtoby, N. P., Abdulaal, A., Barkhof, F., & Alexander, D. C. (2023, October). A coupled-mechanisms modelling framework for neurodegeneration. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 459-469). Cham: Springer Nature Switzerland.
https://link.springer.com/chapter/10.1007/978-3-031-43993-3_45

[3]. Thompson, E., Schroder, A., He, T., Legouhy, A., Zhao, X., Cole, J.H., Oxtoby, N.P. and Alexander, D.C., (2024, July). Demonstration of an open-source toolbox for network spreading models: regional amyloid burden promotes tau production in Alzheimer's disease. In Alzheimer's Association International Conference. ALZ.

[3]. Thompson, E., Schroder, A., He, T., Legouhy, A., Zhao, X., Cole, J.H., Oxtoby, N.P. and Alexander, D.C., (2024, July). Demonstration of an open-source toolbox for network spreading models: regional amyloid burden promotes tau production in Alzheimer's disease. In Alzheimer's Association International Conference. ALZ.
