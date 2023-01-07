# PROJECT 2: Linear Algebra and Optimization for Machine Learning

Project by:
    - Juan Sebastian Mesa
    - David Plazas

## Requirements

The code was developed and tested in `Python 3.10.9`. Backwards compatibility is not
guaranteed. Moreover, the codebase has the following dependencies:

- `numpy==1.23.5`
- `scipy==1.9.3`
- `pandas==1.5.1`
- `matplotlib==3.6.2`
- `scikit-learn==1.2.0`
- `tqdm=4.64.1`

These are also specified in the `requirements.txt` file, which can be easily installed
using `pip` with:

`> pip install -r requirements.txt`

## General usage

The `.zip` file contains one main script, namely, `project_scripts.py`.

The second file `project_scripts.py` contains the script to run the requested algorithms
for the project, and extract the results for the report. The execution time is around
13min.

This script assumes there exists a folder `figs` in the same folder of execution. It
also assumes that the data is in the same folder and named `heart.csv`. The script can
be run by executing on a terminal from the same folder:

`> python project_scripts.py`

The script creates all plots and, additionally, it stores useful information after each
method on JSON files.
