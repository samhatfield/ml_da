# Machine learning and data assimilation with a quasigeostrophic vorticity system

This repository contains code for investigating machine learning and data assimilation with a quasigeostrophic vorticity model.

## Repository structure

- `ai` contains scripts for training a neural net to replace the numerical model.
- `assimilation` contains scripts for running data assimilation with the numerical model and the trained model.
- `forecast` contains scripts for comparing "weather forecasts" run with the numerical model and the trained model.
- `numerical_model` contains FORTRAN90 code for running the numerical model. This is compiled with f2py so it can be called from Python.

## Dependencies
- python=3.6
- numpy
- keras
- seaborn
- matplotlib
- netCDF4
- iris

## Installation

1. Set up conda environment. Run `conda activate lorenz96_machine_learning`.
2. Run `source setup.sh` from the root directory. This will build the numerical model and add it to the Python path.
