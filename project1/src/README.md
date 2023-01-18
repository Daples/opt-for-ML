# PROJECT 1: Implementation of Clustering Algorithms

Project by:
    - Luisa Rummel
    - Juan Sebastian Mesa
    - David Plazas

## Requirements

The code was developed and tested in `Python 3.10.8`. Backwards compatibility is not
guaranteed. Moreover, the codebase has the following dependencies:

- `numpy==1.23.5`
- `scipy==1.9.3`
- `pandas==1.5.1`
- `matplotlib==3.6.2`
- `click==8.1.3`

These are also specified in the `requirements.txt` file, which can be easily installed
using `pip` with

`> pip install -r requirements.txt`

## General usage

The `.zip` file contains three main scripts, namely, `main.py`, `project_scripts.py` and
`run_eig.py`. 

The first file `main.py` provides an interactive CLI for clustering data
using the methods. The CLI contains a help interface that can be accessed by running

`> python main.py --help`

The second file `project_scripts.py` contains the script to run the requested algorithms
for the project, and extract some basic results.

The third file `run_eig.py` creates the unnormalized and normalized graph Laplacians and
computes the eigenvalues and vectors for both matrices, and stores them in compressed
`.npz` files for later use. This allowed us to only run the eigen decomposition only
once. Remark: each eigen decomposition takes around 2h.