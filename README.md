<div align="center">

# Out-of-Distribution Detection with Markov Logic Networks

Code for the paper “Improving Out-of-Distribution Detection with Markov Logic Networks” as accepted at ICML2025.

</div>


## Setup

```bash
pip install -r requirements.txt
````

The code uses [Hydra](https://hydra.cc/) for configuration.

## Download Datasets 

You will have to download the CUB200 and GTSRB dataset and extract them to a folder called `data/`.

## Training DNNs and Feature Extraction

Each dataset code lives in its own subfolder with a `train.py` that:

* Trains neural networks (serving as interpretations of predicates and functions in the constraints).
* Saves:

  * Model checkpoints
  * Network predictions on in-distribution (ID) and out-of-distribution (OOD) datasets
  * Network feature representations for ID and OOD data

## Training the MLN and Evaluation

Run `test.py` to:

1. Train a Markov Logic Network (MLN)
2. Evaluate OOD detectors using the pre-extracted predictions and features

## Constraint Mining

Use `mining.py` to automatically mine logical constraints from the data.

## Project Structure

* `compiler.py`
  Compiles a domain-specific FOL notation into Python code.
* `detectors.py`
  Wraps detectors from the `pytorch-ood` library for easy use.
* `mln.py`
  Implements the Markov Logic Network.
* `shared.py`
  Utilities shared across datasets (e.g., MLN training loops, evaluation helpers).
