import matplotlib.pyplot as plt
import numpy as np

from pulser import Pulse, Sequence, Register
from pulser.devices import Chadoq2, DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from pulser_simulation import QutipEmulator

from scipy.spatial.distance import pdist, squareform

def plot_distribution(C, first_n=None, path=None):
    n = len(C.items())
    total = sum(C.values())
    if first_n:
        n = first_n
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True)[:n])
    normalized_occ = [(v / total)*100 for v in C.values()]
    plt.figure(figsize=(12, 6))
    plt.xlabel("Bitstrings")
    plt.ylabel("Percentage of readings")
    plt.bar(C.keys(), normalized_occ, width=0.3)
    plt.xticks(rotation="vertical")
    
    if path:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.show()

def get_cost_colouring(bitstring, Q):
    z = np.array(list(bitstring), dtype=int)
    cost = z.T @ Q @ z
    return cost


def get_cost(counter, Q):
    cost = sum(counter[key] * get_cost_colouring(key, Q) for key in counter)
    return cost / sum(counter.values())  # Divide by total samples

def evaluate_mapping(new_coords, *args):
    """Cost function to minimize. Ideally, the pairwise
    distances are conserved"""
    Q, shape = args
    new_coords = np.reshape(new_coords, shape)
    new_Q = squareform(Chadoq2.interaction_coeff / pdist(new_coords) ** 6)
    # print(np.linalg.norm(new_Q - Q))
    return np.linalg.norm(new_Q - Q)

def qaa(params, reg, Q, show_output=False):
    Omega, delta_0, T = params
    # rounding T to the closest multiple of 4 to avoid warnings
    T = round(T / 4) * 4
    delta_f = -delta_0
    adiabatic_pulse = Pulse(
    InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
    InterpolatedWaveform(T, [delta_0, 0, delta_f]),
    0,
    )

    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel('ising', 'rydberg_global')
    seq.add(adiabatic_pulse, 'ising')

    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()
    final = results.get_final_state()
    count_dict = results.sample_final_state()

    return -get_cost(count_dict, Q) / 3