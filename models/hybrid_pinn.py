import torch
import torch.nn as nn

from quantum.qnn import QuantumLayer


class HybridPINN(nn.Module):

    def __init__(self, n_qubits=4, n_layers=2):

        super(HybridPINN, self).__init__()

        # Quantum Layer
        self.quantum = QuantumLayer(
            n_qubits=n_qubits,
            n_layers=n_layers
        )

        # Classical Layers
        self.classical = nn.Sequential(

            nn.Linear(n_qubits, 32),
            nn.Tanh(),

            nn.Linear(32, 32),
            nn.Tanh(),

            nn.Linear(32, 1)

        )

    def forward(self, x):

        x = self.quantum(x)
        x = self.classical(x)

        return x