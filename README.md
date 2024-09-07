# Heterogeneous GNNs, Over-Smoothing, and Dirichlet Energy
 
This repository provides code for analyzing the behavior of RGCN with a focus on over-smoothing and Dirichlet energy. Uses the IMDB and DBLP datasets, and implements homogeneous methods to the RGCN, particularly Gradient Gate, REsidual Connections and PairNorm regularization. Below is a brief description of each folder and its purpose:

- variance: Contains code to compute and save variances for datasets and models into CSV files. This includes both training and propagation phases.

- plotters: Generates plots using the CSV files. It can visualize water tables or node variances.

- f_1 calc: Trains the model and propagates it to calculate both micro and macro F1 scores.

- hyperparameter_sweep: Sets up a Bayesian search for optimizing RGCN hyperparameters.

- energies: Contains code for calculating various energy metrics.

- norm_loss: Used to analyze gradient behavior.