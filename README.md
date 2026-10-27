# FAMILY
**Framework for Analysing MultI-scaLe sYstems**

FAMILY is a framework designed to analyse multiscale gaseous structures and their hierarchical relationships across spatial scales. It provides tools to:

- build parent–child relationships between gas structures identified at different spatial scales,
- compare the resulting statistics with a theoretical model,
- generate fragmented core mass functions with Monte-Carlo simulation.

This repository is intended for the study of hierarchical structure formation and fragmentation in astrophysical gaseous systems. 

> ## Project status - Early public version (`v0.9.0`).  

> This repository is currently being reorganized and documented.  
> Some sections (including project information, usage tutorials, requirements, and installation instructions) are still incomplete or subject to revision.

> They will be clarified and expanded in upcoming updates of the repository.

## Table of Contents
1. [General Info](#general-info)
2. [Content Code](#content-code)
3. [Installation](#installation)
4. [Related Papers](#related-papers)

## General Info

Version **v0.9.0**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Content Code

### Tool

The FAMILY framework for building parentality relationships is available in the `analysis` folder.  
Function **x** is doing this and that

### Model

The theoretical model used to compare the parentality results is available in the `model` folder.

### [Optional] Benchmark simulation

The `benchmark` folder contains benchmark data used for the tutorial. It also contains in `benchmark/simulation` Monte Carlo code simulations and that randomly sample ellipses to estimate uncertainties, assess projection effects, and generate test data. The code was originally written in Python but was rewritten in Julia to improve performance. This code is optional, unless you want to rebuild specific samples.

## Installation

### FAMILY Tool

Clone this repository:

`git clone https://github.com/ThomassonB/FAMILY.git`

Create a virtual environment and install the dependencies :

`cd FAMILY`
`python3 -m venv FAMILY_venv`
`source ./FAMILY_venv/bin/activate`
`python3 -m pip install --upgrade pip`
`python3 -m pip install -r ./requirements.txt`

You can then launch the Tutorial notebook by running :

`jupyter lab ./Tutorial.ipynb`

### [Optional] Benchmark

Install Julia first (https://julialang.org/).

`cd ./benchmark/simulation`
`julia --project=. -e 'using Pkg; Pkg.instantiate()'`

Then you can open a jupyter lab and select a Julia kernel to play with the notebooks

`jupyter lab`

## Related Papers

Thomasson et al. 2022, Astronomy and Astrophysics, 665, A119. doi:10.1051/0004-6361/202243275

Thomasson et al. 2024, Astronomy and Astrophysics, 689, A133. doi:10.1051/0004-6361/202449649