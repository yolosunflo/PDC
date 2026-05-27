import numpy as np

from config import (
    PREAMBLE_AMPLITUDE, PREAMBLE_LENGTH,
    QPSK_AMPLITUDE, N_MAX, ENERGY_MAX,
    N_INFO_BITS, CONV_TAIL_BITS, N_CODED_BITS,
    K, G1, G2
)
from utils import text_to_bits
"""
Convolutional Encoding Principle:
At each time step, one input bit enters a 6-bit shift register
(memory = K-1 = 6). The encoder therefore operates on a 7-bit vector:
    [current_bit, s1, s2, s3, s4, s5, s6]
where:
    - current_bit is the new input bit
    - s1...s6 are the previous 6 input bits stored in memory

For each input bit, the encoder produces two output bits.
Each output bit is computed as a modulo-2 sum (XOR) of selected
positions of the 7-bit register.

The selected positions are defined by two generator polynomials:
    g1 = 133₈ = 1011011₂
    g2 = 171₈ = 1111001₂

Each '1' in the binary representation indicates that the corresponding
register bit participates in the modulo-2 sum for that output.

Thus, for every input bit:
    - The shift register is updated.
    - Two coded bits are produced.

This yields a rate-1/2 code:
    1 input bit -> 2 coded bits.

The encoder has free distance d_free = 10,
which provides strong error-correction capability when decoded
with the Viterbi algorithm.

Tail bits (6 zeros) are appended to the input sequence to force
the shift register back to the zero state, allowing exact Viterbi
traceback at the receiver.
"""

def conv_encode(bits: np.ndarray) -> np.ndarray:
    """
    Rate-1/2, K=7 convolutional encoder (NASA standard).

    Appends CONV_TAIL_BITS=6 zero tail bits to flush the shift register
    back to state 0, enabling exact Viterbi traceback.

    Parameters:
        bits: 1D array of N_INFO_BITS=240 info bits in {0, 1}
    Returns:
        1D array of N_CODED_BITS=492 coded bits in {0, 1}
    """
    padded = np.concatenate([bits, np.zeros(CONV_TAIL_BITS, dtype=int)])
    output = np.zeros(2 * len(padded), dtype=int)
    state  = 0   # 6-bit shift register; MSB = most recently entered bit

    for i, u in enumerate(padded):
        u = int(u)
        # Full K-element register: [current, s1, s2, ..., s6]
        full = np.array(
            [u] + [(state >> (K - 2 - j)) & 1 for j in range(K - 1)],
            dtype=int,
        )
        output[2 * i]     = int(np.dot(G1, full)) % 2
        output[2 * i + 1] = int(np.dot(G2, full)) % 2
        # Shift in u: u becomes new MSB
        state = ((state >> 1) | (u << (K - 2))) & ((1 << (K - 1)) - 1)

    return output


def map_bpsk(coded: np.ndarray, Aq: float) -> np.ndarray:
    """Maps coded bits {0, 1} to BPSK chips ±Aq:  0 -> +Aq,  1 -> −Aq."""
    return Aq * (1.0 - 2.0 * coded.astype(float))


def build_preamble(A: float, repeat: int = PREAMBLE_LENGTH) -> np.ndarray:
    assert repeat > 0 and repeat % 2 == 0, "Preamble length must be positive and even."
    return np.full(repeat, A, dtype=float)


def encode_message(text: str) -> np.ndarray:
    """
    Full transmitter pipeline:
        text -> bits -> conv encode -> BPSK chips -> preamble -> x

    Parameters:
        text: 40-character string from ALPHABET
    Returns:
        Real-valued transmitted signal x of length PREAMBLE_LENGTH + N_CODED_BITS = 496
    """
    bits = text_to_bits(text)                           # 240 bits in {0,1}
    coded = conv_encode(bits)                           # 492 bits in {0,1}
    chips = map_bpsk(coded, QPSK_AMPLITUDE)             # 492 chips in ±Aq
    preamble = build_preamble(PREAMBLE_AMPLITUDE)       # 4 chips of +A

    x = np.concatenate([preamble, chips])

    assert len(x) <= N_MAX, f"Signal too long: {len(x)} > {N_MAX}"
    assert len(x) % 2 == 0, "Signal length must be even."
    assert np.sum(x**2) <= ENERGY_MAX, f"Energy constraint violated: {np.sum(x**2):.2f} > {ENERGY_MAX}"

    return x
