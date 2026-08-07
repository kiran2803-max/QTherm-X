import pennylane as qml
import torch

class QuantumLayer(torch.nn.Module):

    def __init__(self, n_qubits=4, n_layers=2):

        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers

        dev = qml.device(
            "default.qubit",
            wires=self.n_qubits
        )

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):

            qml.AngleEmbedding(
                inputs,
                wires=range(self.n_qubits)
            )

            qml.StronglyEntanglingLayers(
                weights,
                wires=range(self.n_qubits)
            )

            return [
                qml.expval(qml.PauliZ(i))
                for i in range(self.n_qubits)
            ]

        weight_shapes = {
            "weights": (
                self.n_layers,
                self.n_qubits,
                3
            )
        }

        self.qlayer = qml.qnn.TorchLayer(
            quantum_circuit,
            weight_shapes
        )

    def forward(self, x):
        return self.qlayer(x)