import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

from qft_ofdm import (
    simulate,
    sweep_phase_one_qubit,
    sweep_phase_avg_ser_all_symbols,
    sweep_phase_avg_ser_all_symbols_noisy,
)

FIG_DIR = "figures"


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def save_histogram(counts, title, filename):
    fig = plot_histogram(counts)
    fig.suptitle(title)
    outpath = os.path.join(FIG_DIR, filename)
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def save_ser_plot(x, y, title, filename):
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Phase offset (rad)")
    plt.ylabel("Symbol error rate (SER)")
    plt.title(title)
    plt.grid(True)
    outpath = os.path.join(FIG_DIR, filename)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def save_multi_ser_plot(xs_ys_labels, title, filename, ylabel="Avg SER (over all symbols)"):
    """
    xs_ys_labels: list of (x_array, y_array, label)
    """
    plt.figure()
    for x, y, label in xs_ys_labels:
        plt.plot(x, y, marker="o", label=label)
    plt.xlabel("Phase offset (rad)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    outpath = os.path.join(FIG_DIR, filename)
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def main():
    ensure_dirs()

    # -----------------------
    # 3-qubit setup
    # -----------------------
    n = 3
    symbol = 0b101  # '101' (any 0..7)

    # --- Experiment 1: histogram (ideal channel) ---
    res_ideal = simulate(symbol=symbol, n_qubits=n, phases=[0.0, 0.0, 0.0], shots=4096)
    save_histogram(
        res_ideal.counts,
        title=f"Ideal channel histogram (n={n}, symbol={symbol:0{n}b})",
        filename="n3_hist_ideal.png",
    )
    print("Ideal p_correct:", res_ideal.p_correct)

    # --- Experiment 2: histogram (phase-distorted) ---
    phases_dist = [0.30 * np.pi, -0.20 * np.pi, 0.15 * np.pi]
    res_dist = simulate(symbol=symbol, n_qubits=n, phases=phases_dist, shots=4096)
    save_histogram(
        res_dist.counts,
        title=f"Phase-distorted histogram (n={n}, symbol={symbol:0{n}b})\nphases={phases_dist}",
        filename="n3_hist_phase_distorted.png",
    )
    print("Distorted p_correct:", res_dist.p_correct)

    # --- Experiment 3: SER vs phase sweep for ONE symbol (qubit 0) ---
    phase_vals, ser_vals = sweep_phase_one_qubit(
        symbol=symbol,
        n_qubits=n,
        sweep_qubit=0,
        num_points=61,
        shots=2048
    )
    save_ser_plot(
        phase_vals, ser_vals,
        title=f"QFT-OFDM (n=3): SER vs phase on qubit 0 (symbol={symbol:0{n}b})",
        filename="n3_ser_vs_phase_q0_symbol101.png"
    )

    # --- Experiment 4: Avg SER vs phase sweep over ALL symbols (qubit 0, noiseless) ---
    phase_vals2, ser_avg = sweep_phase_avg_ser_all_symbols(
        n_qubits=n,
        sweep_qubit=0,
        num_points=61,
        shots=1024
    )
    save_ser_plot(
        phase_vals2, ser_avg,
        title="QFT-OFDM (n=3): Avg SER over all symbols vs phase on qubit 0 (noiseless)",
        filename="n3_avg_ser_vs_phase_q0.png"
    )

    # --- Experiment 5: Avg SER vs phase with PHASE DAMPING noise (multiple gammas, qubit 0) ---
    gammas = [0.0, 0.02, 0.05, 0.10]

    plt.figure()
    for gamma in gammas:
        phase_vals_n, ser_avg_n = sweep_phase_avg_ser_all_symbols_noisy(
            n_qubits=n,
            gamma=gamma,
            sweep_qubit=0,
            num_points=61,
            shots=1024
        )
        plt.plot(phase_vals_n, ser_avg_n, marker="o", label=f"gamma={gamma}")

    plt.xlabel("Phase offset (rad)")
    plt.ylabel("Avg SER (over all symbols)")
    plt.title("QFT-OFDM (n=3): Avg SER vs phase on qubit 0 with phase damping")
    plt.grid(True)
    plt.legend()
    outpath = os.path.join(FIG_DIR, "n3_avg_ser_vs_phase_q0_phase_damping.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")

    # --------------------------------------------------------------------
    # Option A: Compare sensitivity of qubit 0 vs 1 vs 2 (noiseless + noisy)
    # --------------------------------------------------------------------

    # --- A1) Noiseless: avg SER vs phase for sweep_qubit = 0,1,2 on one plot
    curves = []
    for q in [0, 1, 2]:
        ph, ser = sweep_phase_avg_ser_all_symbols(
            n_qubits=n,
            sweep_qubit=q,
            num_points=61,
            shots=1024
        )
        curves.append((ph, ser, f"noiseless sweep qubit {q}"))

    save_multi_ser_plot(
        curves,
        title="QFT-OFDM (n=3): Avg SER vs phase (noiseless) — sweep qubit 0/1/2",
        filename="n3_avg_ser_vs_phase_compare_qubits_noiseless.png",
    )

    # --- A2) Noisy: for each gamma, compare qubit 0/1/2 on its own plot
    gammas_compare = [0.02, 0.05, 0.10]
    for gamma in gammas_compare:
        curves = []
        for q in [0, 1, 2]:
            ph, ser = sweep_phase_avg_ser_all_symbols_noisy(
                n_qubits=n,
                gamma=gamma,
                sweep_qubit=q,
                num_points=61,
                shots=1024
            )
            curves.append((ph, ser, f"sweep qubit {q}"))

        save_multi_ser_plot(
            curves,
            title=f"QFT-OFDM (n=3): Avg SER vs phase with phase damping (gamma={gamma}) — sweep qubit 0/1/2",
            filename=f"n3_avg_ser_vs_phase_compare_qubits_gamma_{str(gamma).replace('.','p')}.png",
        )


if __name__ == "__main__":
    main()
