# network_spreading_models
Project for CMICHACKS 2023

Team: Ellie, Anna, Tiantian, Neil, James, Antoine, Xin

## Code

- [pet_merger.py](./code/pet_merger.py) for merging PET CSV(s) with ADNIMERGE. PET is missing VISCODE.

## Idea for synthetic data

- Three regions of interest (ROIs): A, B, C
- Simple connectome with A => B => C (see below)
- Add a "bucket" of pathology to A, run the NDM forward to simulate the spread.

### Simple connectome:

$$
\begin{matrix}
0 & 1 & 0 \\
0 & 0 & 1 \\
0 & 0 & 0
\end{matrix}
$$

synthetic data

### Models included
- Network Diffusion Model(NDM)
  Pathology spreading between connected brain regions.
- Fisher-Kolmogorov-Petrovsky-Piscounov (FKPP)
  Spreading plus uniform local production
- Weighted-FKPP models
  
  Spreading plus regionally-varying production weighted by a vector of choice.
  It has been discovered [1,2] that weighting the production rate of FKPP with regional vectors improves the original FKPP model by taking into account the difference in the regional characteristics during disease progression. We define this new model as weighted FKPP and display its advantage using regional amyloid as the weight here.

  
## References
[1].	He, T., Schroder, A., Thompson, E., Oxtoby, N. P., Abdulaal, A., Barkhof, F., & Alexander, D. C. (2023). Coupled pathology appearance and propagation models of neurodegeneration. The Organization for Human Brain Mapping (OHBM) 2023 Annual Meeting, [Oral Presentation]

[2].	He, T., Thompson, E., Schroder, A., Oxtoby, N. P., Abdulaal, A., Barkhof, F., & Alexander, D. C. (2023, October). A coupled-mechanisms modelling framework for neurodegeneration. In International Conference on Medical Image Computing and Computer-Assisted Intervention (pp. 459-469). Cham: Springer Nature Switzerland.
https://link.springer.com/chapter/10.1007/978-3-031-43993-3_45
  

