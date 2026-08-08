# QTherm-X

### Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)](https://pytorch.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum-purple)](https://pennylane.ai/)
[![Quantum Computing](https://img.shields.io/badge/Quantum-Hybrid-green)](https://www.ibm.com/quantum)
[![PINN](https://img.shields.io/badge/PINN-Physics%20Informed-orange)](https://maziarraissi.github.io/PINNs/)
[![Research](https://img.shields.io/badge/Research-WISER%202026-success)](https://www.bosonqpsi.com/)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

---

## Research Report

**QTherm-X: Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics**

[**Read the Full Research Paper (PDF)**](report/QTherm-X_Research_Paper.pdf)

The research paper contains the complete methodology, mathematical formulation, experimental setup, model comparison, NASA C-MAPSS validation, results, discussion, and conclusion.

---

## Overview

QTherm-X is a hybrid **Quantum-Assisted Physics-Informed Neural Network (QAPINN)** framework developed for solving thermal physics problems using a combination of classical deep learning and quantum computing.

The framework integrates:

* Physics-Informed Neural Networks (PINNs)
* Variational Quantum Circuits (VQCs)
* PennyLane
* PyTorch
* Automatic differentiation
* One-dimensional heat equation modelling
* NASA C-MAPSS aerospace engine data for validation

The primary objective is to investigate quantum feature extraction within physics-informed learning for thermal modelling and engineering applications.

---

## Project Objective

The project investigates:

> **Can quantum-assisted feature extraction be effectively integrated with Physics-Informed Neural Networks for solving thermal partial differential equations while maintaining physical consistency?**

The proposed framework combines quantum feature extraction with classical physics-informed learning and evaluates its behaviour on thermal PDE problems.

---

## Key Contributions

1. Developed a hybrid Quantum-Assisted Physics-Informed Neural Network for thermal physics.
2. Integrated a Variational Quantum Circuit with a classical neural network using PennyLane and PyTorch.
3. Solved the one-dimensional transient heat equation using physics-informed learning.
4. Evaluated Classical PINN and Hybrid QAPINN models using multiple performance metrics.
5. Investigated different quantum configurations involving qubit count and circuit structure.
6. Evaluated PDE residual, initial-condition loss, boundary-condition loss, MSE, MAE, and Relative $L_2$ error.
7. Analysed training time and trainable model parameters.
8. Extended the framework toward aerospace thermal-health analysis using NASA C-MAPSS data.
9. Developed visualizations for model convergence, thermal behaviour, and degradation analysis.

---

## Team

| Member                 | Role                                                                                                                                                            |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Harikiran**       | Lead Researcher, QTherm-X Architecture, PINN/QAPINN Development, Quantum Circuit Implementation, Experiments, Analysis, Documentation and Repository Management |
| **A. Immanuel Prince** | Research Support, Literature Review, Experimental Analysis and Documentation                                                                                    |
| **R. Jeyaprasanna**    | Research Support, Result Analysis, Visualization and Documentation                                                                                              |

### Contact

| Member             | Email                                                       |
| ------------------ | ----------------------------------------------------------- |
| A. Harikiran       | [harikiran1328@gmail.com](mailto:harikiran1328@gmail.com)   |
| A. Immanuel Prince | [imman.augustin@gmail.com](mailto:imman.augustin@gmail.com) |
| R. Jeyaprasanna    | [okprasanna8@gmail.com](mailto:okprasanna8@gmail.com)       |

---

## System Architecture

The QTherm-X architecture follows a hybrid quantum-classical pipeline:

```text
                 Thermal Physics Problem
                          │
                          ▼
                 Spatial / Time Input
                       (x, t)
                          │
                          ▼
              ┌─────────────────────┐
              │  Quantum Encoding   │
              │  Angle Embedding    │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ Variational Quantum │
              │      Circuit        │
              │                     │
              │ Strongly Entangling │
              │      Layers         │
              └──────────┬──────────┘
                         │
                         ▼
              Pauli-Z Expectation
                    Measurements
                         │
                         ▼
              ┌─────────────────────┐
              │ Classical Neural    │
              │      Network        │
              └──────────┬──────────┘
                         │
                         ▼
                  Temperature
                    u(x,t)
                         │
                         ▼
              Physics-Informed Loss
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           PDE Loss    IC Loss    BC Loss
              │          │          │
              └──────────┼──────────┘
                         ▼
                   Total Loss
                         │
                         ▼
                  Model Optimization
```

---

## Computational Workflow

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
┌───────────────────────────────┐
│       Classical PINN          │
└───────────────┬───────────────┘
                │
                │
                ▼
┌───────────────────────────────┐
│        Hybrid QAPINN          │
│                               │
│  Quantum Feature Extraction   │
│             +                 │
│    Classical Neural Network   │
└───────────────┬───────────────┘
                │
                ▼
          Model Training
                │
                ▼
       Loss / Error Evaluation
                │
                ▼
       Comparative Analysis
                │
                ▼
       Thermal Visualization
                │
                ▼
     NASA C-MAPSS Validation
```

---

## Governing Equation

QTherm-X uses the one-dimensional transient heat equation:

$$
\frac{\partial u(x,t)}{\partial t}
==================================

\alpha
\frac{\partial^2 u(x,t)}{\partial x^2}
$$

where:

* $u(x,t)$ represents temperature.
* $x$ represents the spatial coordinate.
* $t$ represents time.
* $\alpha$ represents thermal diffusivity.

The PINN learns a solution that satisfies the governing PDE together with the initial and boundary conditions.

---

## Physics-Informed Loss

The total training objective is defined as:

$$
\mathcal{L}_{total}
===================

\mathcal{L}*{PDE}
+
\mathcal{L}*{IC}
+
\mathcal{L}_{BC}
$$

where:

* $\mathcal{L}_{PDE}$ is the PDE residual loss.
* $\mathcal{L}_{IC}$ is the initial-condition loss.
* $\mathcal{L}_{BC}$ is the boundary-condition loss.

The PDE residual is:

$$
r(x,t)
======

## \frac{\partial u}{\partial t}

\alpha
\frac{\partial^2u}{\partial x^2}
$$

and the PDE loss is calculated using:

$$
\mathcal{L}_{PDE}
=================

\frac{1}{N}
\sum_{i=1}^{N}
r(x_i,t_i)^2
$$

---

## Quantum Layer

The QAPINN uses a Variational Quantum Circuit implemented using PennyLane.

### Quantum configuration

| Parameter           | Configuration              |
| ------------------- | -------------------------- |
| Quantum simulator   | PennyLane `default.qubit`  |
| Qubits              | 2, 3 and 4 configurations  |
| Circuit depth       | 2 layers                   |
| Encoding            | Angle Embedding            |
| Variational circuit | Strongly Entangling Layers |
| Measurement         | Pauli-Z expectation        |
| Classical framework | PyTorch                    |
| Optimizer           | Adam                       |
| Learning rate       | 0.001                      |
| Training epochs     | 3000                       |
| Training points     | 5000                       |

The quantum circuit transforms the spatial-temporal input $(x,t)$ into a quantum feature representation. The resulting expectation values are passed to the classical neural network.

---

## Models

### Classical PINN

```text
Input (x,t)
    │
    ▼
Linear Layer
    │
   Tanh
    │
    ▼
Hidden Layer
    │
   Tanh
    │
    ▼
Hidden Layer
    │
   Tanh
    │
    ▼
Temperature u(x,t)
```

### Hybrid QAPINN

```text
Input (x,t)
    │
    ▼
Angle Embedding
    │
    ▼
Variational Quantum Circuit
    │
    ▼
Pauli-Z Measurements
    │
    ▼
Quantum Features
    │
    ▼
Classical Neural Network
    │
    ▼
Temperature u(x,t)
```

---

## Evaluation Metrics

The project evaluates the models using:

* Total Training Loss
* PDE Residual Loss
* Initial Condition Loss
* Boundary Condition Loss
* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Relative $L_2$ Error
* Training Time
* Number of Trainable Parameters

These metrics provide complementary information about model accuracy, physics consistency, computational cost, and model complexity.

---

## NASA C-MAPSS Validation

QTherm-X is extended toward aerospace thermal-health analysis using the NASA C-MAPSS turbofan engine degradation dataset.

Two subsets were considered:

* **FD001**
* **FD004**

FD001 represents a comparatively simpler operating scenario, while FD004 contains multiple operating conditions and degradation modes.

### Validation workflow

```text
NASA C-MAPSS Dataset
        │
        ▼
Data Preprocessing
        │
        ▼
Sensor Selection
        │
        ▼
Feature Engineering
        │
        ▼
Thermal / Health Indicators
        │
        ▼
Degradation Analysis
        │
        ▼
FD001 / FD004 Analysis
        │
        ▼
Thermal Visualization
```

### FD001

[**View FD001 Thermal Surface**](figures/fd001_surface.png)

[**View FD001 Health Analysis**](figures/fd001_health.png)

### FD004

[**View FD004 Thermal Surface**](figures/fd004_surface.png)

[**View FD004 Health Analysis**](figures/fd004_health.png)

---

## Repository Structure

```text
QTherm-X/
│
├── benchmark.py
├── model.py
├── hybrid_model.py
├── train.py
├── evaluate.py
├── pinn_loss.py
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
│   ├── fd001_surface.png
│   ├── fd001_health.png
│   ├── fd004_surface.png
│   └── fd004_health.png
│
├── outputs/
│
├── report/
│   └── QTherm-X_Research_Paper.pdf
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

| Technology   | Purpose                                       |
| ------------ | --------------------------------------------- |
| Python       | Main development language                     |
| PyTorch      | Neural networks and automatic differentiation |
| PennyLane    | Quantum circuit implementation                |
| NumPy        | Numerical computation                         |
| Matplotlib   | Visualization                                 |
| Git          | Version control                               |
| GitHub       | Repository management                         |
| NASA C-MAPSS | Aerospace degradation validation              |

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/kiran2803-max/QTherm-X.git
cd QTherm-X
pip install -r requirements.txt
```

---

## Running the Project

Run the benchmark:

```bash
python benchmark.py
```

Run the NASA C-MAPSS preprocessing:

```bash
python preprocessing/preprocess_nasa.py
```

The generated results and visualizations are stored in the corresponding project directories.

---

## Research Report

The complete research paper is available here:

[**QTherm-X Research Paper — PDF**](report/QTherm-X_Research_Paper.pdf)

The paper includes:

* Introduction
* Related Work
* QTherm-X Architecture
* Mathematical Formulation
* Experimental Setup
* Classical PINN
* Hybrid QAPINN
* Model Comparison
* Error Analysis
* NASA C-MAPSS Validation
* Results and Discussion
* Conclusion
* References

---

## Project Status

| Component             | Status            |
| --------------------- | ----------------- |
| Classical PINN        | Complete          |
| Hybrid QAPINN         | Complete          |
| Quantum Circuit       | Complete          |
| Model Comparison      | Complete          |
| Error Metrics         | Complete          |
| Thermal Visualization | Complete          |
| NASA C-MAPSS Analysis | Complete          |
| Research Paper        | Complete          |
| GitHub Repository     | Finalization      |
| Submission            | Final Preparation |

---

## References

1. M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.

2. M. Schuld, R. Sweke, and J. J. Meyer, "Effect of data encoding on the expressive power of variational quantum-machine-learning models," *Physical Review A*, vol. 103, 032430, 2021.

3. S. Wang, Y. Teng, and P. Perdikaris, "Understanding and mitigating gradient flow pathologies in physics-informed neural networks," *SIAM Journal on Scientific Computing*, vol. 43, no. 5, pp. A3055–A3081, 2021.

4. NASA C-MAPSS Dataset, NASA Prognostics Center of Excellence.

5. PennyLane Documentation — Quantum machine learning and differentiable quantum programming.

6. PyTorch Documentation — Deep learning and automatic differentiation framework.

---

## Acknowledgement

This project was developed as part of the **WISER Global Quantum + AI Summer Program 2026 / BosonQ Psi Challenge**, focusing on hybrid quantum-classical approaches for scientific machine learning and Physics-Informed Neural Networks.

---

## License

This repository is intended for academic and research purposes.

See [LICENSE](LICENSE) for licensing information.
