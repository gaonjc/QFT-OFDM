from qiskit import QuantumCircuit
import numpy as np


def qft(circ: QuantumCircuit, qubits):
    """
    In-place QFT on `qubits` (little-endian list: qubits[0] is LSB).
    Includes final swaps to match standard DFT ordering.
    """
    n = len(qubits)
    for j in range(n):
        circ.h(qubits[j])
        for k in range(j + 1, n):
            angle = np.pi / (2 ** (k - j))
            circ.cp(angle, qubits[k], qubits[j])

    # Reverse order
    for i in range(n // 2):
        circ.swap(qubits[i], qubits[n - i - 1])


def iqft(circ: QuantumCircuit, qubits):
    """
    In-place inverse QFT (adjoint of QFT).
    """
    n = len(qubits)

    # Undo swaps
    for i in range(n // 2):
        circ.swap(qubits[i], qubits[n - i - 1])

    # Reverse controlled phases and H
    for j in reversed(range(n)):
        for k in reversed(range(j + 1, n)):
            angle = -np.pi / (2 ** (k - j))
            circ.cp(angle, qubits[k], qubits[j])
        circ.h(qubits[j])
