# TensorFlowLikelihoods

## Introduction
The "TensorFlowLikelihoods" repository is part of a Bachelor’s project that focuses on enhancing the computation of cosmological observables. The core objective of this project is to enable rapid and precise computations of profile likelihoods, leveraging the power of auto-differentiation through TensorFlow. This initiative involves converting existing likelihood functions, sourced from the [MontePython](https://github.com/brinckmann/montepython_public/tree/3.6/montepython/likelihoods) repository, into a format compatible with TensorFlow, thereby facilitating gradient-based computational methods.

## Folder Structure
This repository consists of five primary folders, each serving a distinct purpose in the project:

### 1. `modified_connect`
This folder contains modified files from [CONNECT](https://github.com/AarhusCosmology/connect_public) (version 23.5.0) necessary to reproduce all results from the project. These modifications are essential for integrating the likelihood functions with TensorFlow.

### 2. `temporary_likelihoods`
Includes an example of the intermediate translation step, showcasing the transition from loop-based to non-loop based code. The example provided is for the case of the 'bao' likelihood.

### 3. `tf_likelihoods`
This folder hosts all the likelihood codes that, while translated into TensorFlow, still rely on MontePython and CLASS. These codes represent an important step in the translation process.

### 4. `profile_likelihoods`
Contains all the fully translated and CONNECT-adapted likelihood codes. These codes represent the final product of the translation process and are ready for use in gradient-based computations with CONNECT.

### 5. `notebooks`
Here you'll find all the Jupyter notebooks (`.ipynb` files) used to generate the plots and results for this project. These notebooks serve as practical examples for utilizing the translated likelihood functions.
