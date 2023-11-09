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
