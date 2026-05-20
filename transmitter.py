import numpy as np

from config import (
    PREAMBLE_AMPLITUDE,
    PREAMBLE_LENGTH,
    QPSK_AMPLITUDE,
    N_MAX,
    ENERGY_MAX,
    N_INFO_BITS
)
from utils import text_to_bits

# Builds the Hadamard matrix H_(2^m) recursively.
# Parameters:
#   m - recursion depth (matrix size = 2^m)
# Returns:
#   Hadamard matrix of shape (2^m, 2^m) with values ±1
def build_hadamard_matrix(m: int) -> np.ndarray:
    """
    Builds the Hadamard matrix H_(2^m) recursively using:
    H_(2n) = [[H_n,  H_n],
              [H_n, -H_n]]
    """

    if m == 0:
        return np.array([[1]])

    H_prev = build_hadamard_matrix(m - 1)

    top = np.concatenate([H_prev, H_prev], axis=1)
    bottom = np.concatenate([H_prev, -H_prev], axis=1)

    return np.concatenate([top, bottom], axis=0)


# Encodes 4 information bits into one Hadamard codeword of length 8.
# Parameters:
#   bits4 - array of 4 bits ordered as [b3, b2, b1, b0]
# Returns:
#   1D array of 8 values in {-1, +1}
def hadamard_encode_block(bits4: np.ndarray) -> np.ndarray:
    """
    RM(1,3) Hadamard encoding.

    Mapping:
        - b2,b1,b0 select a row of H_8
        - b3 controls the sign of the codeword
    """

    assert len(bits4) == 4

    H = build_hadamard_matrix(3)
    b3, b2, b1, b0 = bits4

    index = b2 * 4 + b1 * 2 + b0
    codeword = H[index].copy()

    if b3 == 1:
        codeword *= -1

    return codeword


# Encodes all 240 information bits using RM(1,3).
# Parameters:
#   bits - 1D array of 240 bits
# Returns:
#   1D array of 480 coded chips in {-1, +1}
def hadamard_encode_all(bits: np.ndarray) -> np.ndarray:
    """
    Splits the input into 60 blocks of 4 bits,
    encodes each block independently,
    then concatenates all codewords.
    """

    assert len(bits) == 240

    blocks = bits.reshape(60, 4)
    encoded_blocks = [hadamard_encode_block(block) for block in blocks]

    return np.concatenate(encoded_blocks)


# Maps Hadamard chips to QPSK coordinates.
# Parameters:
#   coded_pm1 - array of ±1 values
#   Aq        - QPSK amplitude
# Returns:
#   Real-valued QPSK vector
def map_to_qpsk(coded_pm1: np.ndarray, Aq: float) -> np.ndarray:
    """
    QPSK modulation.
    Each pair:
        (b1, b2) ∈ {-1,+1}²
    is mapped to:
        (b1*Aq, b2*Aq)
    Ex: [+1, -1] -> [Aq, -Aq]
    """

    coded_pm1 = np.asarray(coded_pm1, dtype=float)

    assert len(coded_pm1) % 2 == 0
    assert set(np.unique(coded_pm1)).issubset({-1, 1})

    # Grouper par paires
    pairs = coded_pm1.reshape(-1, 2)

    # Appliquer l'amplitude
    qpsk_symbols = Aq * pairs

    # Retourner sous forme plate
    return qpsk_symbols.flatten()


# Builds the synchronization preamble.
# Parameters:
#   A      - preamble amplitude
#   repeat - number of repetitions (must be even)
# Returns:
#   1D array containing repeated A values
def build_preamble(A: float, repeat: int = 4) -> np.ndarray:
    """
    Builds a preamble of the form: (A, A, A, ..., A)
    """

    assert repeat > 0
    assert repeat % 2 == 0, "Le préambule doit avoir une longueur paire."

    return np.full(repeat, A, dtype=float)


# Full encoding pipeline: text -> transmitted signal x.
# Parameters:
#   text - 40-character input string
# Returns:
#   Real-valued transmitted signal x
def encode_message(text: str) -> np.ndarray:
    """
    Full transmitter pipeline:
        text
        -> bits
        -> Hadamard encoding
        -> QPSK modulation
        -> preamble insertion
        -> transmitted vector x
    """

    bits = text_to_bits(text)
    assert len(bits) == N_INFO_BITS

    coded = hadamard_encode_all(bits)
    symbols = map_to_qpsk(coded, QPSK_AMPLITUDE)
    preamble = build_preamble(PREAMBLE_AMPLITUDE, PREAMBLE_LENGTH)
    x = np.concatenate([preamble, symbols])

    assert len(x) <= N_MAX
    assert len(x) % 2 == 0
    assert np.sum(x**2) <= ENERGY_MAX

    return x