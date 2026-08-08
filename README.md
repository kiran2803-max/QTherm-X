<p align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple)
![Quantum Computing](https://img.shields.io/badge/Quantum-Hybrid-green)
![PINN](https://img.shields.io/badge/PINN-Physics%20Informed-orange)
![Research](https://img.shields.io/badge/Research-WISER%202026-success)
![License](https://img.shields.io/badge/License-MIT-blue)

[![CI](https://github.com/kiran2803-max/QTherm-X/actions/workflows/ci.yml/badge.svg)](https://github.com/kiran2803-max/QTherm-X/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kiran2803-max/QTherm-X/actions/workflows/codeql.yml/badge.svg)](https://github.com/kiran2803-max/QTherm-X/actions/workflows/codeql.yml)

![Black](https://img.shields.io/badge/code%20style-black-000000.svg)
![Linting](https://img.shields.io/badge/linting-flake8-yellow)
![Security](https://img.shields.io/badge/security-CodeQL-blue)

![GitHub last commit](https://img.shields.io/github/last-commit/kiran2803-max/QTherm-X)
![GitHub issues](https://img.shields.io/github/issues/kiran2803-max/QTherm-X)

</p>

# QTherm-X

### Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics

QTherm-X is a hybrid quantum-classical Physics-Informed Neural Network framework designed for solving thermal physics problems using **Variational Quantum Circuits (VQCs)** and classical neural networks.

The project investigates the integration of **quantum feature extraction with physics-informed learning** for solving the one-dimensional heat equation and explores its potential application to aerospace thermal monitoring and predictive maintenance.

The framework combines **PennyLane** for quantum computation and **PyTorch** for classical deep learning and automatic differentiation.

---

## Project Overview

Traditional numerical approaches for solving thermal PDEs can become computationally expensive when dealing with complex physical systems.

Physics-Informed Neural Networks address this problem by incorporating governing physical equations directly into the training objective.

QTherm-X extends this concept by introducing a **quantum feature extraction layer** into the PINN architecture.

The main objective is to investigate whether quantum-enhanced representations can provide a useful alternative to conventional neural-network representations for thermal PDE modelling.

---

## Research Objective

The primary objective of QTherm-X is to develop and evaluate a hybrid quantum-classical PINN for thermal physics.

The project focuses on:

* Solving the one-dimensional heat equation.
* Integrating Variational Quantum Circuits with PINNs.
* Comparing Classical PINN and Hybrid QAPINN performance.
* Evaluating different quantum circuit configurations.
* Measuring PDE consistency and prediction accuracy.
* Analysing model complexity and computational cost.
* Exploring the potential application of the framework to aerospace thermal monitoring.

---

## System Architecture

The QTherm-X architecture follows a hybrid quantum-classical pipeline:

```text
                 Input
                (x, t)
                   │
                   ▼
        ┌─────────────────────┐
        │   Quantum Layer     │
        │                     │
        │  Angle Embedding    │
        │         │           │
        │         ▼           │
        │ Strongly Entangling │
        │      Layers         │
        │         │           │
        │         ▼           │
        │ Pauli-Z Expectation │
        └──────────┬──────────┘
                   │
                   ▼
          Quantum Feature Vector
                   │
                   ▼
        ┌─────────────────────┐
        │   Classical Neural  │
        │      Network        │
        │                     │
        │ Linear → Tanh       │
        │ Linear → Tanh       │
        │ Linear → Output     │
        └──────────┬──────────┘
                   │
                   ▼
            Temperature
              u(x,t)
                   │
                   ▼
        ┌─────────────────────┐
        │ Physics-Informed    │
        │      Loss           │
        │                     │
        │ PDE + IC + BC       │
        └─────────────────────┘
```

The quantum circuit acts as a trainable feature extractor, while the classical neural network produces the final temperature prediction.

---

## Physics Model

QTherm-X uses the one-dimensional transient heat equation:

$$
\frac{\partial u}{\partial t}
=============================

\alpha
\frac{\partial^2 u}{\partial x^2}
$$

where:

* $u(x,t)$ represents temperature.
* $x$ represents spatial position.
* $t$ represents time.
* $\alpha$ represents thermal diffusivity.

The physics-informed objective combines the PDE residual with initial and boundary condition losses:

$$
\mathcal{L}
===========

\mathcal{L}*{PDE}
+
\mathcal{L}*{IC}
+
\mathcal{L}_{BC}
$$

This allows the model to learn a solution that satisfies the governing physical constraints.

---

## Quantum Model

The quantum component uses a Variational Quantum Circuit implemented using PennyLane.

### Quantum configuration

| Component            | Configuration              |
| -------------------- | -------------------------- |
| Quantum framework    | PennyLane                  |
| Quantum simulator    | `default.qubit`            |
| Qubit configurations | 2, 3, 4                    |
| Circuit depth        | 2 layers                   |
| Encoding             | Angle Embedding            |
| Variational circuit  | Strongly Entangling Layers |
| Measurement          | Pauli-Z expectation        |
| Classical framework  | PyTorch                    |

The quantum circuit converts the spatial-temporal input $(x,t)$ into a quantum feature representation.

These features are then passed to the classical neural network.

---

## Experimental Setup

The benchmark experiments compare:

1. Classical PINN
2. Hybrid QAPINN with 2 qubits
3. Hybrid QAPINN with 3 qubits
4. Hybrid QAPINN with 4 qubits

The models were evaluated under controlled training conditions.

### Training configuration

| Parameter       |                     Value |
| --------------- | ------------------------: |
| Training points |                     5,000 |
| Training epochs |                     3,000 |
| Optimizer       |                      Adam |
| Learning rate   |                     0.001 |
| Quantum layers  |                         2 |
| Execution       |                       CPU |
| Quantum backend | PennyLane `default.qubit` |

---

## Benchmark Evaluation

The models are evaluated using multiple metrics:

* Total training loss
* PDE residual loss
* Initial condition loss
* Boundary condition loss
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Relative $L_2$ error
* Training time
* Number of trainable parameters

This provides a broader evaluation than using training loss alone.

---

## Training Behaviour

The recorded benchmark experiments show convergence for both the Classical PINN and Hybrid QAPINN configurations.

The Classical PINN converged more rapidly, while the Hybrid QAPINN models demonstrated progressive reduction of the physics-informed loss during training.

Example final-stage training losses:

| Model          | Qubits |     Loss at 2500 epochs |
| -------------- | -----: | ----------------------: |
| Classical PINN |     -- | $6.733710\times10^{-4}$ |
| Hybrid QAPINN  |      2 | $7.078764\times10^{-2}$ |
| Hybrid QAPINN  |      3 | $9.090878\times10^{-2}$ |
| Hybrid QAPINN  |      4 | $9.230375\times10^{-2}$ |

These values are training-loss observations from the benchmark runs and are not presented as evidence of unconditional quantum advantage.

---

## Benchmark Results

The repository contains the generated benchmark figures and experimental outputs.

### Total Loss

![Total Training Loss](figures/TotalLoss.png)

### PDE Loss

![PDE Residual Loss](figures/PDELoss.png)

### Relative L2 Error

![Relative L2 Error](figures/RelativeL2.png)

### Mean Squared Error

![Mean Squared Error](figures/MSE.png)

### Mean Absolute Error

![Mean Absolute Error](figures/MAE.png)

### Training Time

![Training Time](figures/TrainingTime.png)

### Trainable Parameters

![Trainable Parameters](figures/Parameters.png)

> If your figures are stored under a different folder, update the paths above to match the repository structure.

---

## NASA C-MAPSS Validation

QTherm-X also includes an aerospace-oriented validation component based on the **NASA C-MAPSS turbofan engine degradation dataset**.

Two subsets are considered:

* FD001
* FD004

### FD001

FD001 represents a comparatively simpler degradation scenario with a single operating condition and fault mode.

### FD004

FD004 represents a more challenging scenario involving multiple operating conditions and degradation modes.

The processed data are used to investigate:

* Sensor behaviour
* Thermal trends
* Degradation patterns
* Thermal health indicators
* Remaining Useful Life (RUL)
* Aerospace predictive-maintenance applications

The NASA C-MAPSS component is intended as an application-oriented validation direction rather than replacing the controlled heat-equation benchmark.

---

## Project Workflow

```text
Problem Definition
        │
        ▼
One-Dimensional Heat Equation
        │
        ▼
Training Point Generation
        │
        ▼
Classical PINN Baseline
        │
        ▼
Quantum Feature Extraction
        │
        ▼
Hybrid QAPINN
        │
        ▼
Physics-Informed Training
        │
        ▼
Benchmark Evaluation
        │
        ├── PDE Loss
        ├── IC Loss
        ├── BC Loss
        ├── MAE
        ├── MSE
        ├── Relative L2
        ├── Training Time
        └── Parameters
        │
        ▼
Thermal Visualisation
        │
        ▼
NASA C-MAPSS Application
        │
        ▼
Aerospace Thermal Monitoring
```

---

## Repository Structure

```text
QTherm-X/
│
├── benchmark.py
├── model.py
├── quantum_layer.py
├── train.py
├── losses.py
├── evaluation.py
│
├── preprocessing/
│   └── preprocess_nasa.py
│
├── figures/
│   ├── TotalLoss.png
│   ├── PDELoss.png
│   ├── RelativeL2.png
│   ├── MSE.png
│   ├── MAE.png
│   ├── TrainingTime.png
│   ├── Parameters.png
│   ├── system_architecture.png
│   ├── workflow.png
│   └── quantum_circuit.png
│
├── outputs/
│
├── report/
│   └── QTherm-X_Research_Paper.pdf
│
├── requirements.txt
│
└── README.md
```

---

## Technologies

| Technology | Purpose                                                 |
| ---------- | ------------------------------------------------------- |
| Python     | Core implementation                                     |
| PyTorch    | Classical neural networks and automatic differentiation |
| PennyLane  | Quantum circuit implementation                          |
| NumPy      | Numerical computation                                   |
| Matplotlib | Scientific visualisation                                |
| SciPy      | Numerical processing                                    |
| Git        | Version control                                         |
| GitHub     | Repository and reproducibility                          |

---

## Development Environment

The experiments were developed and executed using a CPU-based environment.

The primary software stack consists of:

```text
Python
PyTorch
PennyLane
NumPy
Matplotlib
SciPy
Git
GitHub
```

---

## Team

| Member                 | Role                                                                                                                                                |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Harikiran**       | Project Lead, QTherm-X Architecture, PINN/QAPINN Implementation, Quantum Circuit Integration, Benchmarking, Experimental Analysis and Documentation |
| **A. Immanuel Prince** | Research Support, Literature Review, Thermal Physics Analysis and Project Documentation                                                             |
| **R. Jeyaprasanna**    | Validation Support, Results Review, Visualization and Presentation/Submission Support                                                               |

The project was developed collaboratively as part of the **WISER Global Quantum + AI Summer Program 2026 / BQP Challenge**.

---

## Research Paper

The complete research paper is available below:

### [Read the QTherm-X Research Paper (PDF)](report/QTherm-X_Research_Paper.pdf)

The paper contains the complete methodology, mathematical formulation, experimental setup, benchmark results, NASA C-MAPSS application discussion, and conclusions.

---

## Reproducibility

Clone the repository:

```bash
git clone https://github.com/kiran2803-max/QTherm-X.git
cd QTherm-X
```

Create the environment:

```bash
conda create -n qthermx python=3.11
conda activate qthermx
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the benchmark:

```bash
python benchmark.py
```

The benchmark evaluates the Classical PINN and multiple Hybrid QAPINN configurations.

---

## Outputs

The repository contains the generated experimental outputs, including:

* Training loss curves
* PDE loss plots
* Relative $L_2$ error plots
* MSE comparison
* MAE comparison
* Training-time comparison
* Parameter comparison
* Thermal visualizations
* Model checkpoints
* Benchmark data

These outputs support the results discussed in the accompanying research paper.

---

## Explainability

QTherm-X is designed to make the contribution of each component observable.

The framework allows analysis of:

* Classical versus quantum feature representations
* Qubit-count effects
* Quantum circuit depth
* Physics-informed loss components
* Prediction accuracy
* Model complexity
* Training cost
* Thermal behaviour

Rather than assuming that the introduction of quantum computing automatically improves performance, QTherm-X evaluates the hybrid architecture through controlled benchmarking.

---

## Project Status

### Completed

* [x] Classical PINN implementation
* [x] Hybrid QAPINN implementation
* [x] Variational Quantum Circuit integration
* [x] 2-qubit experiment
* [x] 3-qubit experiment
* [x] 4-qubit experiment
* [x] Benchmark evaluation
* [x] Loss analysis
* [x] MAE and MSE evaluation
* [x] Relative $L_2$ evaluation
* [x] Training-time analysis
* [x] Parameter analysis
* [x] Thermal visualisation
* [x] NASA C-MAPSS preprocessing
* [x] FD001 analysis
* [x] FD004 analysis
* [x] Research paper
* [x] GitHub repository

---

## Limitations

The current implementation uses a classical quantum simulator rather than physical quantum hardware.

Therefore, the reported quantum computation times represent simulation overhead and should not be interpreted as measurements of future fault-tolerant quantum hardware.

The current thermal PDE benchmark is one-dimensional, and the quantum circuit is evaluated using a limited number of qubits.

These limitations motivate future investigation using larger circuits, higher-dimensional thermal PDEs, and real quantum devices.

---

## Future Work

Future development will focus on:

* Higher-dimensional thermal PDEs
* Larger quantum circuits
* Real quantum hardware
* Improved quantum encoding strategies
* Thermal digital twins
* Aerospace engine thermal monitoring
* Advanced degradation modelling
* Real-time predictive maintenance
* Physics-informed aerospace prognostics

---

## References

The primary references used in the project include:

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.

2. M. Schuld, R. Sweke, and J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," *Physical Review A*, vol. 103, no. 3, 032430, 2021.

3. S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," *SIAM Journal on Scientific Computing*, vol. 43, no. 5, pp. A3055–A3081, 2021.

4. N. Rahaman et al., "On the spectral bias of neural networks," *Proceedings of the 36th International Conference on Machine Learning*, 2019.

5. NASA, "Prognostics Center of Excellence: C-MAPSS Turbofan Engine Degradation Simulation Dataset."

6. PennyLane Documentation — quantum machine learning and differentiable quantum computing.

7. PyTorch Documentation — deep learning and automatic differentiation.

---

## Acknowledgement

This work was developed as part of the **WISER Global Quantum + AI Summer Program 2026 / BQP Challenge**, with a focus on exploring hybrid quantum-classical approaches for Physics-Informed Neural Networks and scientific computing.

---

## License

This project is intended for academic and research purposes.

---

<p align="center">
<b>QTherm-X</b><br>
Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics
</p>
