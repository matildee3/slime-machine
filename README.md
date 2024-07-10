# slime-machine


This repository contains the AAARDM.py script, developed as part of Matilde Sartori's Master's Degree Thesis for the Design Research Course at BAU - College of Arts and Design of Barcelona & Universitat de Vic - Universitat Central de Catalunya (2024).

## Overview

AAARDM.py is a computer vision algorithm inspired by the behavior of slime moulds, particularly the species Physarum Polycephalum. The algorithm focuses on edge detection, starting from randomness and searching for solutions rather than predicting them. This approach provides both a philosophical and practical framework for a more ecological and inclusive form of computing.

## Features

- Utilizes the MEALPY library for the latest meta-heuristic algorithms in Python.
- Implements a modified version of the Slime Mould Algorithm (SMA).
- Generates truly random seeds using the Random.org JSON-RPC API.
- Includes caching for efficient objective function evaluations.

## Requirements

- Python 3.x
- Libraries: requests, numpy, matplotlib, scipy, pillow, functools, mealpy

## Installation

To install the required libraries, run:
```bash
pip install requests numpy matplotlib scipy pillow mealpy
```

## Usage

import aaardm

```bash
api_key = 'YOUR_API_KEY'
true_random_seed = aaardm.generate_random_numbers(api_key, 1, 1, 1000000)

# Initialize and solve the optimization problem
model = aaardm.DevSMA(epoch=30000, pop_size=200, p_t=0.05)
g_best = model.solve(aaardm.problem_dict)

# Display the results
aaardm.display_results(g_best)

```

