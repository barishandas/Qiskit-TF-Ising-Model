# Ising Model Variational Quantum Eigensolver (VQE)

## Overview

`isingvqe.py` is a sophisticated quantum computing simulation tool for studying the Transverse Field Ising model using the Variational Quantum Eigensolver (VQE) approach. This implementation provides a flexible framework for exploring quantum systems with configurable connectivity and interaction patterns.

## Features

- Supports quantum simulation of Ising models with up to 8+ qubits
- Configurable connectivity options:
  - Linear (nearest-neighbor interactions)
  - Full (all-to-all interactions)
- Adaptive hardware-efficient quantum circuit ansatz
- Comprehensive energy optimization using scipy's minimize function
- Exact diagonalization for smaller systems
- Visualization of quantum circuit and energy convergence
- Detailed performance tracking and error analysis

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Qiskit
- SciPy

## Installation

```bash
pip install numpy matplotlib qiskit scipy
```

## Quick Start

```python
from isingvqe import LargeIsingVQE

# Create an 8-qubit Ising model simulation
vqe = LargeIsingVQE(
    num_qubits=8,
    J=6.0,  # Interaction strength
    h=3.0,  # Transverse field strength
    connectivity='linear'
)

# Visualize the quantum circuit
vqe.visualize_ansatz()

# Run optimization
results = vqe.optimize(max_iter=2000)

# Plot energy convergence
vqe.plot_convergence()
```

## Connectivity Options

- `'linear'`: Nearest-neighbor interactions (1D chain)
- `'full'`: All-to-all interactions between qubits

## Parameters

- `num_qubits`: Number of qubits in the system
- `J`: Interaction strength between qubits
- `h`: Transverse field strength
- `connectivity`: Interaction topology
- `shots`: Number of measurement shots

## Visualization

The library provides:
- Quantum circuit visualization
- Energy convergence plot
- Optional exact ground state energy comparison

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
