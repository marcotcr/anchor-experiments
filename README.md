# Anchor experiments
This repository has code for replicating the experiments in [*High-Precision Model-Agnostic Explanations*](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf).

If you actually want to try anchors, use the code in [this repository](https://github.com/marcotcr/anchor) instead.

## Installation
Inside a virtualenv, run:

    pip install lime==0.1.1.29 sklearn matplotlib seaborn numpy argparse xgboost==0.4a30

#### Examples
Run:

    python run_compute_explanations.py
    python run_process_results.py
    python make_graphs_and_table.py

Results will be in folder `results`
