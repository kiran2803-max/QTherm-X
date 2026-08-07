import torch

from quantum.qnn import QuantumLayer

print("=" * 50)
print("QTherm-X Quantum Test")
print("=" * 50)

model = QuantumLayer()

sample = torch.tensor([0.2, 0.4, 0.6, 0.8])

output = model(sample)

print("\nQuantum Output:")
print(output)

print("\nNumber of Qubits : 4")

print("\nQuantum Layer Working Successfully!")