import numpy as np

from config import PREAMBLE_AMPLITUDE, PREAMBLE_LENGTH, N_INFO_BITS, CONV_TAIL_BITS
from utils import apply_T, apply_T_inverse, bits_to_text

# ── Viterbi precomputed tables ────────────────────────────────────────────────
# NASA K=7 rate-1/2 code  (same polynomials as transmitter)
_K        = 7
_N_STATES = 1 << (_K - 1)   # 64
_G1 = [1, 0, 1, 1, 0, 1, 1]
_G2 = [1, 1, 1, 1, 0, 0, 1]

# _OUTPUT[s, u] = (c1, c2)  output bits for state s and input u
# _NEXT[s, u]   = next state
_OUTPUT = np.zeros((_N_STATES, 2, 2), dtype=np.int8)
_NEXT   = np.zeros((_N_STATES, 2),    dtype=np.int32)

for _s in range(_N_STATES):
    for _u in range(2):
        _full = [_u] + [(_s >> (_K - 2 - _j)) & 1 for _j in range(_K - 1)]
        _OUTPUT[_s, _u, 0] = sum(_full[_i] * _G1[_i] for _i in range(_K)) % 2
        _OUTPUT[_s, _u, 1] = sum(_full[_i] * _G2[_i] for _i in range(_K)) % 2
        _NEXT[_s, _u]      = ((_s >> 1) | (_u << (_K - 2))) & (_N_STATES - 1)

# Reverse table: for each next_state, the two (prev_state, input) pairs
# that lead to it — used for vectorised ACS.
_REV     = np.zeros((_N_STATES, 2, 2), dtype=np.int32)
_rev_cnt = np.zeros(_N_STATES, dtype=int)
for _s in range(_N_STATES):
    for _u in range(2):
        _ns             = int(_NEXT[_s, _u])
        _REV[_ns, _rev_cnt[_ns]] = [_s, _u]
        _rev_cnt[_ns]  += 1

# Precomputed ±1 chip values and incoming-path indices for the ACS loop
_X1  = 1.0 - 2.0 * _OUTPUT[:, :, 0].astype(float)  # (64, 2)
_X2  = 1.0 - 2.0 * _OUTPUT[:, :, 1].astype(float)  # (64, 2)
_PS0 = _REV[:, 0, 0]   # (64,) first  incoming prev-state
_U0  = _REV[:, 0, 1]   # (64,) first  incoming input
_PS1 = _REV[:, 1, 0]   # (64,) second incoming prev-state
_U1  = _REV[:, 1, 1]   # (64,) second incoming input


# ── Functions ─────────────────────────────────────────────────────────────────

def estimate_T(y_preamble: np.ndarray, A: float) -> int:
    """
    ML estimation of the rotation index k ∈ {1,2,3,4} from the received preamble.
    Candidates are built dynamically from PREAMBLE_LENGTH.
    """
    base       = np.full(PREAMBLE_LENGTH, A, dtype=float)
    candidates = np.array([apply_T(base, k) for k in range(1, 5)])
    return int(np.argmax(candidates @ y_preamble)) + 1


def viterbi_decode(soft: np.ndarray) -> np.ndarray:
    """
    Soft-decision Viterbi decoder for the NASA K=7 rate-1/2 convolutional code.

    The forward pass is fully vectorised with numpy (no Python inner loops).
    Traceback starts from state 0 because the encoder terminates the register
    with 6 tail bits.

    Parameters:
        soft - 1D array of N_CODED_BITS=492 de-rotated BPSK soft values (≈ ±Aq + noise)
    Returns:
        1D array of N_INFO_BITS=240 decoded info bits in {0, 1}
    """
    n_steps = len(soft) // 2   # 246 (240 info + 6 tail)

    path_metric = np.full(_N_STATES, -np.inf)
    path_metric[0] = 0.0

    survivors = np.zeros((n_steps, _N_STATES), dtype=np.int8)
    prev_st   = np.zeros((n_steps, _N_STATES), dtype=np.int8)

    for t in range(n_steps):
        r1, r2 = soft[2 * t], soft[2 * t + 1]

        # Branch metrics for all (state, input) pairs — shape (64, 2)
        branch = r1 * _X1 + r2 * _X2

        # Accumulated metrics — shape (64, 2)
        total = path_metric[:, None] + branch

        # Vectorised ACS: compare the 2 paths arriving at each next state
        m0       = total[_PS0, _U0]          # (64,) metric via 1st incoming path
        m1       = total[_PS1, _U1]          # (64,) metric via 2nd incoming path
        choose_1 = m1 > m0

        path_metric = np.where(choose_1, m1, m0)
        survivors[t] = np.where(choose_1, _U1,  _U0 ).astype(np.int8)
        prev_st[t]   = np.where(choose_1, _PS1, _PS0).astype(np.int8)

    # Traceback from state 0 (code is terminated — encoder ends in state 0)
    decoded = np.zeros(n_steps, dtype=int)
    state   = 0
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = survivors[t, state]
        state      = int(prev_st[t, state])

    return decoded[:N_INFO_BITS]   # discard the 6 tail bits


def decode_message(y: np.ndarray) -> str:
    """
    Full receiver pipeline:
        y → preamble / data split → estimate T → de-rotate → Viterbi → text
    """
    y_preamble  = y[:PREAMBLE_LENGTH]
    y_data      = y[PREAMBLE_LENGTH:]

    k_hat       = estimate_T(y_preamble, PREAMBLE_AMPLITUDE)
    y_derotated = apply_T_inverse(y_data, k_hat)
    bits        = viterbi_decode(y_derotated)

    return bits_to_text(bits)
