""" 
to implement:

text_to_bits()
bits_to_text()
apply_T()
apply_T_inverse()

"""

import numpy as np
from config import ALPHABET, N_CHARS, BITS_PER_CHAR, N_INFO_BITS

CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ALPHABET)}

# Transforms text to bits by mapping each char to its index in ALPHABET 
# and then to 6 bits (MSB first).
def text_to_bits(text: str) -> np.ndarray:
    assert len(text) == N_CHARS # est-ce que ça doit être strictement égal à 40 ????????????????????????????????????????
    bits = []
    for c in text:
        idx = CHAR_TO_IDX[c]
        for i in reversed(range(BITS_PER_CHAR)):
            bits.append((idx >> i) & 1)
    return np.array(bits, dtype=int)

# Transforms bits to text by grouping into 6-bit chunks 
# and mapping to ALPHABET (240 bits → 40 chars) 
def bits_to_text(bits: np.ndarray) -> str:
    assert len(bits) == N_INFO_BITS
    text = ''
    for i in range(N_CHARS):
        chunk = bits[i*BITS_PER_CHAR : (i+1)*BITS_PER_CHAR]
        idx = 0
        for b in chunk:
            idx = (idx << 1) | int(b)
        text += IDX_TO_CHAR[idx]
    return text

# Applies T_k by pairs on x.
# T_1 is identity, T_2 is 90° rotation, T_3 is 180°, T_4 is 270°.
def apply_T(x: np.ndarray, k: int) -> np.ndarray:
    assert len(x) % 2 == 0
    pairs = x.reshape(-1, 2)
    a, b = pairs[:, 0], pairs[:, 1]
    
    if k == 1:
        out = np.stack([ a,  b], axis=1)
    elif k == 2:
        out = np.stack([-b,  a], axis=1)
    elif k == 3:
        out = np.stack([-a, -b], axis=1)
    elif k == 4:
        out = np.stack([ b, -a], axis=1)
    else:
        raise ValueError(f"k must be in {{1,2,3,4}}, received {k}")
    
    return out.flatten()

# Applies T_k^(-1) by pairs. Note: T_k^(-1) = T_k for k=1,3 and T_2^(-1)=T_4, T_4^(-1)=T_2.
def apply_T_inverse(x: np.ndarray, k: int) -> np.ndarray:
    inverses = {1: 1, 2: 4, 3: 3, 4: 2}
    return apply_T(x, inverses[k])

# Check constraints: n even, n ≤ 500, ‖x‖² ≤ 1200
# Used in transmitter to assert constraints, 
# and in test_local to verify that channel output is valid.
def check_constraints(x: np.ndarray) -> bool:
    from config import N_MAX, ENERGY_MAX
    if len(x) % 2 != 0: return False
    if len(x) > N_MAX: return False
    if np.sum(x**2) > ENERGY_MAX: return False
    return True
