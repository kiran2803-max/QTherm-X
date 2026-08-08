<p align="center">

<img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch" alt="PyTorch">
<img src="https://img.shields.io/badge/PennyLane-Quantum-purple" alt="PennyLane">
<img src="https://img.shields.io/badge/Quantum-Hybrid-green" alt="Quantum Hybrid">
<img src="https://img.shields.io/badge/PINN-Physics%20Informed-orange" alt="Physics-Informed Neural Network">
<img src="https://img.shields.io/badge/Research-WISER%202026-success" alt="WISER 2026">
<img src="https://img.shields.io/badge/License-MIT-blue" alt="MIT License">

</p>

<p align="center">

<img src="https://img.shields.io/github/stars/kiran2803-max/QTherm-X?style=social" alt="GitHub Stars">
<img src="https://img.shields.io/github/forks/kiran2803-max/QTherm-X?style=social" alt="GitHub Forks">
<img src="https://img.shields.io/github/last-commit/kiran2803-max/QTherm-X" alt="Last Commit">
<img src="https://img.shields.io/github/issues/kiran2803-max/QTherm-X" alt="GitHub Issues">

</p>

<h1 align="center">QTherm-X</h1>

<h3 align="center">
Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics
</h3>

<p align="center">
A hybrid quantum-classical framework for solving thermal partial differential equations using Physics-Informed Neural Networks and Variational Quantum Circuits.
</p>

<p align="center">
<b>BQP WISER Global Quantum + AI Summer Challenge 2026</b>
</p>

---

# 🔬 Overview

**QTherm-X** is an Explainable Quantum-Assisted Physics-Informed Neural Network framework developed to investigate the integration of quantum computing with physics-informed machine learning for thermal physics.

The framework combines a **Classical Physics-Informed Neural Network (PINN)** with a **Variational Quantum Circuit (VQC)** to solve the **one-dimensional transient heat equation** while enforcing physical constraints during model training.

The quantum component is implemented using **PennyLane**, while the classical neural network and automatic differentiation pipeline are implemented using **PyTorch**.

The primary objective of QTherm-X is not to claim unconditional quantum advantage. Instead, the project investigates the feasibility of quantum feature extraction within physics-informed learning and evaluates its behaviour through controlled experiments.

---

# 🎯 Research Objective

The central research question investigated by QTherm-X is:

> **Can quantum feature extraction be effectively integrated with Physics-Informed Neural Networks to model thermal systems while maintaining physical consistency?**

To investigate this question, the project compares a conventional Classical PINN with Hybrid QAPINN configurations using different numbers of quantum qubits.

The evaluated configurations include:

* Classical PINN
* Hybrid QAPINN — 2 Qubits
* Hybrid QAPINN — 3 Qubits
* Hybrid QAPINN — 4 Qubits

---

# 🚀 Key Features

* 🧠 Physics-Informed Neural Network for thermal PDE modelling
* ⚛️ Variational Quantum Circuit integration using PennyLane
* 🔗 Hybrid quantum-classical architecture
* 🔬 Automatic differentiation using PyTorch
* 🌡️ One-dimensional transient heat equation modelling
* 📊 Classical PINN vs QAPINN benchmarking
* ⚙️ Multi-qubit experiments
* 📈 PDE residual analysis
* 📉 MSE, MAE and Relative $L_2$ evaluation
* ⏱️ Training-time comparison
* 🔢 Trainable-parameter comparison
* 📊 Automated benchmark visualizations
* 🔎 Explainable quantum feature extraction
* 🧪 Reproducible experimental workflow

---

# 🧮 Governing Physics

QTherm-X solves the one-dimensional transient heat equation:

$$
\frac{\partial u}{\partial t}
=============================

\alpha
\frac{\partial^2 u}{\partial x^2}
$$

where:

* $u(x,t)$ represents temperature.
* $x$ represents the spatial coordinate.
* $t$ represents time.
* $\alpha$ represents thermal diffusivity.

The neural network is trained not only to minimize prediction error but also to satisfy the governing physical equation.

---

# 🧠 Physics-Informed Learning

The total physics-informed objective is composed of three major terms:

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

* $\mathcal{L}_{PDE}$ represents the PDE residual loss.
* $\mathcal{L}_{IC}$ represents the initial-condition loss.
* $\mathcal{L}_{BC}$ represents the boundary-condition loss.

The PDE residual is calculated using automatic differentiation:

$$
r(x,t)
======

## \frac{\partial u}{\partial t}

\alpha
\frac{\partial^2u}{\partial x^2}
$$

The PDE loss is then calculated from the residual:

$$
\mathcal{L}_{PDE}
=================

\frac{1}{N}
\sum_{i=1}^{N}
r(x_i,t_i)^2
$$

This allows the neural network to learn a solution that respects the underlying thermal physics.

---

# ⚛️ Hybrid QAPINN Architecture

The proposed architecture follows:

```text
                    Input
                   (x, t)
                     │
                     ▼
          ┌─────────────────────┐
          │ Quantum Feature     │
          │ Extraction Layer    │
          │                     │
          │ Angle Embedding     │
          │        +            │
          │ Variational Layers  │
          └──────────┬──────────┘
                     │
                     ▼
             Quantum Features
          Pauli-Z Expectation
                     │
                     ▼
          ┌─────────────────────┐
          │ Classical Neural    │
          │ Network             │
          │                     │
          │ Linear + Tanh       │
          │ Linear + Tanh       │
          │ Linear              │
          └──────────┬──────────┘
                     │
                     ▼
              Temperature
                 u(x,t)
                     │
                     ▼
          Automatic Differentiation
                     │
              ┌──────┴──────┐
              ▼             ▼
             u_t           u_xx
              │             │
              └──────┬──────┘
                     ▼
            Heat Equation Residual
                     │
                     ▼
          Physics-Informed Loss
                     │
                     ▼
               Optimization
```

---

# 🔄 Project Workflow

```text
              Thermal Physics Problem
                       │
                       ▼
                Heat Equation
                       │
                       ▼
             Training Point Sampling
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
      Classical PINN       Hybrid QAPINN
             │                   │
             │             Quantum Circuit
             │                   │
             │             Feature Extraction
             │                   │
             │             Classical Layers
             │                   │
             └─────────┬─────────┘
                       ▼
             Physics-Informed Loss
                       │
                       ▼
                 Model Training
                       │
                       ▼
              Benchmark Evaluation
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       PDE Loss       MSE          MAE
          │            │            │
          └────────────┼────────────┘
                       ▼
                 Relative L2
                       │
                       ▼
              Training Time
                       │
                       ▼
             Parameter Comparison
                       │
                       ▼
               Final Analysis
```

---

# ⚛️ Quantum Circuit

The quantum layer uses the spatial and temporal coordinates as inputs to the quantum circuit.

The workflow is:

```text
(x, t)
  │
  ▼
Angle Embedding
  │
  ▼
Variational Quantum Layers
  │
  ▼
Entanglement
  │
  ▼
Pauli-Z Measurements
  │
  ▼
Quantum Feature Vector
  │
  ▼
Classical Neural Network
  │
  ▼
Temperature Prediction
```

The expectation values obtained from Pauli-Z measurements form the quantum feature representation used by the downstream classical network.

---

# 📊 Benchmark Configuration

The benchmark was designed to compare the classical baseline with multiple quantum configurations.

| Configuration  | Qubits | Epochs | Training Points |
| -------------- | -----: | -----: | --------------: |
| Classical PINN |      — |   3000 |            5000 |
| Hybrid QAPINN  |      2 |   3000 |            5000 |
| Hybrid QAPINN  |      3 |   3000 |            5000 |
| Hybrid QAPINN  |      4 |   3000 |            5000 |

### Common Training Configuration

| Parameter              | Value           |
| ---------------------- | --------------- |
| Optimizer              | Adam            |
| Learning Rate          | 0.001           |
| Epochs                 | 3000            |
| Training Points        | 5000            |
| Quantum Circuit Layers | 2               |
| Quantum Backend        | `default.qubit` |
| Execution              | CPU             |

---

# 📈 Evaluation Metrics

The models are evaluated using multiple complementary metrics.

### Total Loss

Measures the combined physics-informed objective:

$$
\mathcal{L}_{total}
===================

\mathcal{L}*{PDE}
+
\mathcal{L}*{IC}
+
\mathcal{L}_{BC}
$$

### PDE Loss

Measures how well the predicted solution satisfies the heat equation.

### MSE

$$
MSE =
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat{y}_i)^2
$$

### MAE

$$
MAE =
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat{y}_i|
$$

### Relative $L_2$ Error

$$
L_2^{rel}
=========

\frac{
|u-\hat{u}|_2
}{
|u|_2
}
$$

### Computational Metrics

The benchmark also evaluates:

* Training time
* Number of trainable parameters

---

# 🧪 Experimental Results

The benchmark experiments evaluate Classical PINN and Hybrid QAPINN configurations under the same overall experimental framework.

The training logs demonstrate convergence of the investigated models over 3000 epochs.

### Classical PINN

```text
Epoch 0       : 8.384546e-01
Epoch 500     : 2.221719e-02
Epoch 1000    : 2.100524e-03
Epoch 1500    : 5.814580e-04
Epoch 2000    : 4.674458e-04
Epoch 2500    : 6.733710e-04
```

### Hybrid QAPINN — 2 Qubits

```text
Epoch 0       : 3.756981e-01
Epoch 500     : 1.519051e-01
Epoch 1000    : 1.247129e-01
Epoch 1500    : 1.024728e-01
Epoch 2000    : 7.719737e-02
Epoch 2500    : 7.078764e-02
```

### Hybrid QAPINN — 3 Qubits

```text
Epoch 0       : 5.209951e-01
Epoch 500     : 1.647203e-01
Epoch 1000    : 1.125970e-01
Epoch 1500    : 9.238566e-02
Epoch 2000    : 8.902448e-02
Epoch 2500    : 9.090878e-02
```

### Hybrid QAPINN — 4 Qubits

```text
Epoch 0       : 3.606607e-01
Epoch 500     : 1.406562e-01
Epoch 1000    : 1.130650e-01
Epoch 1500    : 1.100800e-01
Epoch 2000    : 9.650256e-02
Epoch 2500    : 9.230375e-02
```

> **Note:** Detailed final benchmark metrics such as MAE, MSE, Relative $L_2$, training time, and parameter count are available in the generated experiment outputs and accompanying report. Only experimentally generated values should be reported in the final results table.

---

# 📊 Benchmark Visualizations

The repository contains visualizations for the experimental comparison, including:

* Total training loss
* PDE residual loss
* Relative $L_2$ error
* MSE
* MAE
* Training time
* Trainable parameters
* Temperature field visualization
* Thermal surface visualization

Recommended output organization:

```text
outputs/
│
├── figures/
│   ├── TotalLoss.png
│   ├── PDELoss.png
│   ├── RelativeL2.png
│   ├── MSE.png
│   ├── MAE.png
│   ├── TrainingTime.png
│   └── Parameters.png
│
├── metrics/
├── models/
└── logs/
```

---

# ✈️ Aerospace Application

QTherm-X is motivated by thermal monitoring applications in engineering systems such as:

* Aircraft engines
* Marine propulsion systems
* Industrial thermal systems
* Energy systems
* Thermal management systems

The framework provides a foundation for future integration with real-world engineering datasets.

---

# 🛩️ NASA C-MAPSS Validation Direction

The project also considers the **NASA C-MAPSS turbofan engine degradation dataset** as an aerospace-oriented validation direction.

The investigated subsets include:

* **FD001**
* **FD004**

FD001 provides a comparatively simpler degradation scenario, while FD004 represents a more complex operating environment with multiple operating conditions and degradation patterns.

Potential applications include:

* Thermal health monitoring
* Degradation analysis
* Health-index estimation
* Predictive maintenance
* Remaining Useful Life (RUL) analysis

The C-MAPSS component is intended to extend the thermal PDE modelling framework toward practical aerospace prognostics.

---

# 🗂️ Repository Structure

```text
QTherm-X/
│
├── benchmark.py
├── model.py
├── quantum_layer.py
├── pinn_losses.py
│
├── preprocessing/
│   └── preprocess_nasa.py
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── logs/
│
├── report/
│   ├── QTherm-X_Paper.pdf
│   └── ...
│
├── presentation/
│   ├── QTherm-X_Presentation.pdf
│   └── ...
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Technology     | Purpose                                                |
| -------------- | ------------------------------------------------------ |
| **Python**     | Core implementation                                    |
| **PyTorch**    | Classical neural network and automatic differentiation |
| **PennyLane**  | Quantum circuit and hybrid quantum-classical learning  |
| **NumPy**      | Numerical computation                                  |
| **Matplotlib** | Visualization                                          |
| **Git**        | Version control                                        |
| **GitHub**     | Repository and collaboration                           |

---

# ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/kiran2803-max/QTherm-X.git
cd QTherm-X
```

Create and activate the project environment:

```bash
conda create -n qthermx python=3.11
conda activate qthermx
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Benchmark

Run the benchmark:

```bash
python benchmark.py
```

The benchmark evaluates:

```text
Classical PINN
        ↓
Hybrid QAPINN — 2 Qubits
        ↓
Hybrid QAPINN — 3 Qubits
        ↓
Hybrid QAPINN — 4 Qubits
```

The generated outputs can be stored in the `outputs/` directory.

---

# 🔁 Reproducibility

The project is designed to provide a reproducible experimental workflow.

The repository contains:

* Source code
* Training configuration
* Physics-informed loss implementation
* Quantum circuit implementation
* Benchmark scripts
* Generated figures
* Experimental outputs
* Documentation
* Research paper
* Presentation

Experiments use fixed training configurations and controlled model comparisons wherever applicable.

---

# 👥 Team

## Authors

| Member                 | Role                                     | Responsibility                                                                                                                                                                                                                                             |
| ---------------------- | ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A. Harikiran**       | **Project & Technical Lead**             | Overall research direction, QTherm-X architecture, Classical PINN implementation, Hybrid QAPINN implementation, quantum circuit integration, training pipeline, benchmarking, metric evaluation, visualization, technical report, and project integration. |
| **A. Immanuel Prince** | **Research & Validation Support**        | Research review, methodology review, benchmark observation review, validation support, and documentation review.                                                                                                                                           |
| **R. Jeyaprasanna**    | **Documentation & Presentation Support** | Documentation review, presentation organization, visualization/result review, and final submission material support.                                                                                                                                       |

### Contribution Structure

The project was organized around three complementary areas:

**Technical Development**

* PINN implementation
* QAPINN implementation
* Quantum circuit integration
* Model training
* Benchmarking
* Metric evaluation

**Research & Validation**

* Literature review
* Methodology review
* Experimental validation
* Result interpretation

**Documentation & Communication**

* Technical documentation
* Figures
* Presentation
* Submission preparation

The core technical implementation and experimental development were led by **A. Harikiran**.

---

# 📄 Research Paper

The complete technical paper describing the QTherm-X methodology, mathematical formulation, architecture, experiments, results, and conclusions is available in the repository.

**Paper:**
`report/`

---

# 🎤 Project Presentation

The project presentation containing the research motivation, architecture, methodology, benchmark results, and conclusions is available in:

**Presentation:**
`presentation/`

---

# 📚 References

### Physics-Informed Neural Networks

Raissi, M., Perdikaris, P., and Karniadakis, G. E.
*Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.*
Journal of Computational Physics, 2019.

### Quantum Machine Learning

Schuld, M., Sweke, R., and Meyer, J. J.
*Effect of data encoding on the expressive power of variational quantum-machine-learning models.*
Physical Review A, 2021.

### PINN Optimization

Wang, S., Teng, Y., and Perdikaris, P.
*Understanding and mitigating gradient pathologies in physics-informed neural networks.*
SIAM Journal on Scientific Computing, 2021.

### NASA C-MAPSS

NASA Prognostics Center of Excellence.
*Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) Dataset.*

---

# 🤖 Generative AI Disclosure

Generative AI tools were used as supporting tools during the development of the project.

They assisted with:

* Brainstorming and research organization
* Understanding technical concepts
* Coding assistance and debugging
* Documentation refinement
* Research-paper language refinement
* Presentation organization

Generative AI was **not used to fabricate experimental results or replace the experimental evaluation**. The implementation, model training, benchmark execution, generated outputs, and final technical decisions were performed and verified by the project team.

---

# 🏆 Project Context

QTherm-X was developed as part of the **BQP WISER Global Quantum + AI Summer Challenge 2026**.

The project investigates hybrid quantum-classical machine learning for scientific computing, with a particular focus on physics-informed learning for thermal systems.

The research was conducted using CPU-based quantum simulation through PennyLane's `default.qubit` backend. The project emphasizes feasibility, reproducibility, controlled benchmarking, and future applicability rather than claiming immediate quantum advantage.

---

# 🔮 Future Work

Future development will focus on:

* Scaling to higher-dimensional thermal PDEs
* Increasing quantum circuit depth and expressivity
* Testing additional quantum encoding strategies
* Evaluating larger qubit configurations
* Running experiments on real quantum hardware
* Improving computational efficiency
* Integrating richer aerospace thermal datasets
* Extending toward thermal digital twins
* Exploring predictive maintenance applications
* Investigating real-time thermal health monitoring

---

# 📜 License

This project is released under the **MIT License** for academic and research purposes.

---

# 🙏 Acknowledgements

This work was developed as part of the:

**BQP WISER Global Quantum + AI Summer Challenge 2026**

with a focus on hybrid quantum-classical machine learning, Physics-Informed Neural Networks, and scientific computing.

---

<p align="center">

<b>QTherm-X</b><br> <i>Explainable Quantum-Assisted Physics-Informed Neural Networks for Thermal Physics</i>

</p>

<p align="center">
⭐ If you find this project useful, consider starring the repository.
</p>
