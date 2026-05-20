import numpy as np
from config import PREAMBLE_AMPLITUDE, PREAMBLE_LENGTH
from utils import apply_T_inverse, bits_to_text


# Identifies which rotation T1/T2/T3/T4 the channel applied, using the received preamble.
# Parameters: y_preamble (4 received values), A (preamble amplitude from config)
# Returns: k ∈ {1, 2, 3, 4}, the estimated rotation index
def estimate_T(y_preamble: np.ndarray, A: float) -> int:
    candidates = np.array([
        [ A,  A,  A,  A],   # T1
        [-A,  A, -A,  A],   # T2
        [-A, -A, -A, -A],   # T3
        [ A, -A,  A, -A],   # T4
    ])
    scores = candidates @ y_preamble   # dot product with all candidates at once
    return int(np.argmax(scores)) + 1  # +1 because T is indexed 1..4


# Pass-through soft demodulation: each component is already a soft bit after de-rotation.
# Parameters: y_data - 1D array of 480 floats
# Returns: same array unchanged (Hadamard decoder handles the final decision)
def demap_qpsk(y_data: np.ndarray) -> np.ndarray:
    return y_data


# Computes the Hadamard transform of a vector using the butterfly algorithm.
# Parameters: x1D array of floats (length must be a power of 2)
# Returns: 1D array of floats. Property: FHT(FHT(x)) = n * x
def fast_hadamard_transform(x: np.ndarray) -> np.ndarray:
    x = x.copy().astype(float)
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                a, b = x[j], x[j + h]
                x[j]     = a + b
                x[j + h] = a - b
        h *= 2
    return x


# Decodes a single block of 8 soft bits into 4 info bits.
# Parameters: soft8 — 1D array of 8 floats
# Returns: 1D array of 4 ints ∈ {0, 1} ordered as [b3, b2, b1, b0]
def hadamard_decode_block(soft8: np.ndarray) -> np.ndarray:
    transformed = fast_hadamard_transform(soft8)
    idx = int(np.argmax(np.abs(transformed)))  # most likely row index
    b3 = 1 if transformed[idx] > 0 else 0     # sign gives b3
    b2 = (idx >> 2) & 1
    b1 = (idx >> 1) & 1
    b0 = idx & 1
    return np.array([b3, b2, b1, b0], dtype=int)


# Decodes 480 soft bits into 240 info bits by processing 60 blocks of 8.
# Parameters: soft_bits — 1D array of 480 floats (de-rotated QPSK values)
# Returns: 1D array of 240 ints ∈ {0, 1}
def hadamard_decode_all(soft_bits: np.ndarray) -> np.ndarray:
    blocks = soft_bits.reshape(60, 8)
    decoded = np.concatenate([hadamard_decode_block(block) for block in blocks])
    return decoded.astype(int)


# Full decoding pipeline: received signal → text.
# Parameters: y — 1D array of 484 floats (channel output)
# Returns: string of 40 characters from ALPHABET
def decode_message(y: np.ndarray) -> str:
    y_preamble = y[:PREAMBLE_LENGTH]
    y_data     = y[PREAMBLE_LENGTH:]

    k_hat       = estimate_T(y_preamble, PREAMBLE_AMPLITUDE)  # estimate rotation
    y_derotated = apply_T_inverse(y_data, k_hat)              # undo rotation
    soft        = demap_qpsk(y_derotated)                     # soft QPSK demodulation
    bits        = hadamard_decode_all(soft)                   # soft Hadamard decode

    return bits_to_text(bits)
