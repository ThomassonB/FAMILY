# FAMILY
**Framework for Analysing MultI-scaLe sYstems**

FAMILY is a framework designed to analyse multiscale gaseous structures and their hierarchical relationships across spatial scales. It provides tools to:

- build parent–child relationships between gas structures identified at different spatial scales,
- compare the resulting statistics with a theoretical model,
- generate fragmented core mass functions with Monte-Carlo simulation.

This repository is intended for the study of hierarchical structure formation and fragmentation in astrophysical gaseous systems. 

## Project status - public version (`v1.0.0`).  

> This repository can be used by researchers to investigate the parentality between gas structures and stars in molecular clouds.  
> If you encounter a problem, please open an issue in this GitHub repository. If you would like to fix a bug or contribute, feel free to fork the project and submit a pull request.


## Table of Contents
1. [General Info](#general-info)
2. [Content Code](#content-code)
3. [Installation](#installation)
4. [Related Papers](#related-papers)

## General Info

Version **v1.0.0**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Content Code

### Tool

The FAMILY framework for building parentality relationships is available in the `analysis` folder.  

### Model

The theoretical model used to compare the parentality results is available in the `model` folder.

### [Optional] Benchmark simulation

The `benchmark` folder contains benchmark data used for the tutorial. It also contains in `benchmark/simulation` Monte Carlo code simulations and that randomly sample ellipses to estimate uncertainties, assess projection effects, and generate test data. The code was originally written in Python but was rewritten in Julia to improve performance. This code is optional, unless you want to rebuild specific samples.

## Installation

### FAMILY Tool

Clone this repository:

`git clone https://github.com/ThomassonB/FAMILY.git`

It is preferable that you create a virtual environment and install the dependencies :

`cd FAMILY`<br>
`python3 -m venv FAMILY_venv`<br>
`source ./FAMILY_venv/bin/activate`<br>
`python3 -m pip install --upgrade pip`<br>
`python3 -m pip install -r ./requirements.txt`

The commands may vary depending on your operating system and Python executable. Adjust them as needed.
You can then launch the Tutorial notebook by running :

`jupyter lab ./Tutorial.ipynb`

### [Optional] Benchmark

Install Julia first (https://julialang.org/). The code works on Julia 1.13.

`julia --project=./benchmark/simulation -e 'using Pkg; Pkg.instantiate()'`

Then you can open a jupyter lab and select a Julia kernel to play with the notebooks located in `./benchmark/simulation`.

## Related Papers

[A graph-theory-based multi-scale analysis of hierarchical cascade in molecular clouds (Thomasson et al. 2022)](https://doi.org/10.1051/0004-6361/202243275)

[A stochastic and analytical model of hierarchical fragmentation (Thomasson et al. 2024)](https://doi.org/10.1051/0004-6361/202449649 )

## If you want more

The dev branch contains additional functions for analysing spatial correlations between polygons and the background image. It also includes scripts to derive an IMF from a CMF through a hierarchical fragmentation process ([The coupled origin of the stellar initial mass function and multiplicity (Thomasson et al. 2026)](https://doi.org/10.1051/0004-6361/202554829))