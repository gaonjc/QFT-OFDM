from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qft_utils import qft, iqft

from qiskit_aer.noise import NoiseModel, phase_damping_error



@dataclass
class SimResult:
    circuit: QuantumCircuit
    counts: Dict[str, int]
    p_correct: float
    ser: float


def int_to_bitstring(x: int, n: int) -> str:
    return format(x, f"0{n}b")


def encode_symbol(qc: QuantumCircuit, qubits: Sequence[int], symbol: int) -> None:
    """
    Prepare computational basis |symbol>.
    qubits are assumed little-endian: qubits[0] is LSB.
    """
    for i, q in enumerate(qubits):
        if (symbol >> i) & 1:
            qc.x(q)


def apply_phase_channel(qc: QuantumCircuit, qubits: Sequence[int], phases: Sequence[float]) -> None:
    """
    Toy phase channel: per-qubit Rz(phi).
    phases length must equal len(qubits).
    """
    if len(phases) != len(qubits):
        raise ValueError("phases must have the same length as qubits")
    for q, phi in zip(qubits, phases):
        qc.rz(float(phi), q)


def build_qft_ofdm_circuit(symbol: int, n_qubits: int, phases: Sequence[float]) -> QuantumCircuit:
    """
    TX: encode |symbol> then IQFT
    Channel: Rz phases
    RX: QFT then measure
    """
    if symbol < 0 or symbol >= (1 << n_qubits):
        raise ValueError(f"symbol must be in [0, {2**n_qubits - 1}]")
    if len(phases) != n_qubits:
        raise ValueError("phases must have length n_qubits")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qubits = list(range(n_qubits))  # [0..n-1], little-endian

    # Encode
    encode_symbol(qc, qubits, symbol)

    # TX: spread with IQFT
    iqft(qc, qubits)

    # Channel
    apply_phase_channel(qc, qubits, phases)

    # RX: demux with QFT
    qft(qc, qubits)

    # Measure into classical bits (same indices)
    qc.measure(qubits, qubits)
    return qc


def simulate(symbol: int, n_qubits: int, phases: Sequence[float], shots: int = 2048) -> SimResult:
    sim = AerSimulator()
    qc = build_qft_ofdm_circuit(symbol, n_qubits, phases)

    result = sim.run(qc, shots=shots).result()

    counts = result.get_counts()

    # Qiskit counts keys are bitstrings with the leftmost bit = highest classical bit.
    # Since we measured qubit i -> classical bit i, the correct key is just binary of symbol.
    correct_key = int_to_bitstring(symbol, n_qubits)
    total = sum(counts.values())
    correct = counts.get(correct_key, 0)
    p_correct = correct / total if total else 0.0
    ser = 1.0 - p_correct

    return SimResult(circuit=qc, counts=counts, p_correct=p_correct, ser=ser)


def sweep_phase_one_qubit(
    symbol: int,
    n_qubits: int,
    sweep_qubit: int = 0,
    num_points: int = 41,
    shots: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sweep phase on one qubit from 0..2π, keep others at 0.
    Returns phases, SER array.
    """
    if sweep_qubit < 0 or sweep_qubit >= n_qubits:
        raise ValueError("sweep_qubit out of range")

    phase_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    ser_vals = np.zeros_like(phase_vals)

    for i, phi in enumerate(phase_vals):
        phases = [0.0] * n_qubits
        phases[sweep_qubit] = float(phi)
        res = simulate(symbol, n_qubits, phases, shots=shots)
        ser_vals[i] = res.ser

    return phase_vals, ser_vals

def sweep_phase_avg_ser_all_symbols(
    n_qubits: int,
    sweep_qubit: int = 0,
    num_points: int = 41,
    shots: int = 1024,
):
    import numpy as np
    phase_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    ser_avg = np.zeros_like(phase_vals)

    for i, phi in enumerate(phase_vals):
        phases = [0.0] * n_qubits
        phases[sweep_qubit] = float(phi)

        sers = []
        for symbol in range(1 << n_qubits):
            res = simulate(symbol, n_qubits, phases, shots=shots)
            sers.append(res.ser)

        ser_avg[i] = float(np.mean(sers))

    return phase_vals, ser_avg

def make_phase_damping_noise_model(gamma: float) -> NoiseModel:
    """
    Build a simple phase damping (dephasing) noise model.
    gamma in [0, 1]. Higher gamma => more dephasing.
    """
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in [0, 1]")

    nm = NoiseModel()
    err_1q = phase_damping_error(gamma)

    # Apply to common 1-qubit gates (what your circuit uses)
    one_q_gates = ["h", "x", "rz"]
    for g in one_q_gates:
        nm.add_all_qubit_quantum_error(err_1q, g)

    # Also apply to 2-qubit controlled-phase and swap
    # (QFT uses cp and swap; dephasing per-qubit is a reasonable toy model)
    two_q_gates = ["cp", "swap"]
    for g in two_q_gates:
        nm.add_all_qubit_quantum_error(err_1q.tensor(err_1q), g)

    return nm


def simulate_noisy(symbol: int, n_qubits: int, phases: Sequence[float], gamma: float, shots: int = 2048) -> SimResult:
    """
    Same as simulate(), but with phase damping noise.
    """
    noise_model = make_phase_damping_noise_model(gamma)
    sim = AerSimulator(noise_model=noise_model)

    qc = build_qft_ofdm_circuit(symbol, n_qubits, phases)

    result = sim.run(qc, shots=shots).result()
    counts = result.get_counts()

    correct_key = int_to_bitstring(symbol, n_qubits)
    total = sum(counts.values())
    correct = counts.get(correct_key, 0)
    p_correct = correct / total if total else 0.0
    ser = 1.0 - p_correct

    return SimResult(circuit=qc, counts=counts, p_correct=p_correct, ser=ser)

def sweep_phase_one_qubit_noisy(
    symbol: int,
    n_qubits: int,
    gamma: float,
    sweep_qubit: int = 0,
    num_points: int = 41,
    shots: int = 2048,
):
    import numpy as np
    phase_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    ser_vals = np.zeros_like(phase_vals)

    for i, phi in enumerate(phase_vals):
        phases = [0.0] * n_qubits
        phases[sweep_qubit] = float(phi)
        res = simulate_noisy(symbol, n_qubits, phases, gamma=gamma, shots=shots)
        ser_vals[i] = res.ser

    return phase_vals, ser_vals

def sweep_phase_avg_ser_all_symbols_noisy(
    n_qubits: int,
    gamma: float,
    sweep_qubit: int = 0,
    num_points: int = 41,
    shots: int = 1024,
):
    import numpy as np
    phase_vals = np.linspace(0.0, 2.0 * np.pi, num_points)
    ser_avg = np.zeros_like(phase_vals)

    for i, phi in enumerate(phase_vals):
        phases = [0.0] * n_qubits
        phases[sweep_qubit] = float(phi)

        sers = []
        for symbol in range(1 << n_qubits):
            res = simulate_noisy(symbol, n_qubits, phases, gamma=gamma, shots=shots)
            sers.append(res.ser)

        ser_avg[i] = float(np.mean(sers))

    return phase_vals, ser_avg




