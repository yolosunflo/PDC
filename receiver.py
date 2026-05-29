import numpy as np

from config import PREAMBLE_AMPLITUDE, PREAMBLE_LENGTH, N_INFO_BITS, CONV_TAIL_BITS, K, G1, G2
from utils import apply_T, apply_T_inverse, bits_to_text

N_STATES = 2 ** (K - 1)   # 64

# OUTPUT[s, u] = (c1, c2)  output bits for state s and input u
# NEXT[s, u] = next state
OUTPUT = np.zeros((N_STATES, 2, 2), dtype=np.int8)
NEXT = np.zeros((N_STATES, 2),    dtype=np.int32)

for s in range(N_STATES):
    for u in range(2):
        full = [u] + [(s // 2 ** (K - 2 - j)) % 2 for j in range(K - 1)]
        OUTPUT[s, u, 0] = np.dot(full, G1) % 2
        OUTPUT[s, u, 1] = np.dot(full, G2) % 2
        NEXT[s, u] = ((s // 2) | (u * 2 ** (K - 2))) & (N_STATES - 1)

# Reverse table: for each next_state, the two (prev_state, input) pairs that lead to it ( used for vectorised Add-Compare-Select)
REV = np.zeros((N_STATES, 2, 2), dtype=np.int32)
rev_cnt = np.zeros(N_STATES, dtype=int)
for s in range(N_STATES):
    for u in range(2):
        next_state = int(NEXT[s, u])
        REV[next_state, rev_cnt[next_state]] = [s, u]
        rev_cnt[next_state] += 1

# Precomputed ±1 chip values and incoming-path indices for the Add-Compare-Select loop
CHIPS_1 = 1.0 - 2.0 * OUTPUT[:, :, 0].astype(float)  # (64, 2) expected BPSK chip on branch 1
CHIPS_2 = 1.0 - 2.0 * OUTPUT[:, :, 1].astype(float)  # (64, 2) expected BPSK chip on branch 2
INCOMING_STATE = REV[:, :, 0]   # (64, 2) incoming prev-state for each of the 2 paths
INCOMING_BIT = REV[:, :, 1]   # (64, 2) input bit that caused each transition


 # ML estimation of the rotation index k ∈ {1,2,3,4} from the received preamble
def estimate_T(y_preamble: np.ndarray, A: float) -> int:
    base = np.full(PREAMBLE_LENGTH, A, dtype=float)
    candidates = np.array([apply_T(base, k) for k in range(1, 5)])
    return int(np.argmax(candidates @ y_preamble)) + 1

# Viterbi decoder 
# Parameters: received_symbols, 1D array of N_CODED_BITS=492 derotated BPSK received_symbols values (= ±Aq + noise)
# Returns: 1D array of N_INFO_BITS=240 decoded info bits in {0, 1}
def viterbi_decode(received_symbols: np.ndarray) -> np.ndarray:

    n_steps = len(received_symbols) // 2   # 246 (240 info + 6 tail)
    path_metric = np.full(N_STATES, -np.inf)
    path_metric[0] = 0.0

    survivors  = np.zeros((n_steps, N_STATES), dtype=np.int8)
    prev_state = np.zeros((n_steps, N_STATES), dtype=np.int8)
    idx        = np.arange(N_STATES)

    for t in range(n_steps):
        r1, r2 = received_symbols[2 * t], received_symbols[2 * t + 1]

        # Branch metrics for all (state, input) pairs; shape (64, 2)
        branch = r1 * CHIPS_1 + r2 * CHIPS_2

        # Accumulated metrics for the 2 paths arriving at each next state
        total = path_metric[:, None] + branch
        cand  = total[INCOMING_STATE, INCOMING_BIT]   # (64, 2)

        #Add-Compare-Select pick the best incoming path for each state
        best = np.argmax(cand, axis=1)                # (64,)/ 0 ou 1

        path_metric   = cand[idx, best]
        survivors[t]  = INCOMING_BIT[idx, best]
        prev_state[t] = INCOMING_STATE[idx, best]

    # Traceback from state 0 (code is terminated encoder ends in state 0)
    decoded = np.zeros(n_steps, dtype=int)
    state = 0
    for t in range(n_steps - 1, -1, -1):
        decoded[t] = survivors[t, state]
        state      = int(prev_state[t, state])

    return decoded[:N_INFO_BITS]   # discard the 6 tail bits

#Full receiver pipeline
def decode_message(y: np.ndarray) -> str:
    y_preamble  = y[:PREAMBLE_LENGTH]
    y_data = y[PREAMBLE_LENGTH:]

    k_hat = estimate_T(y_preamble, PREAMBLE_AMPLITUDE)
    y_derotated = apply_T_inverse(y_data, k_hat)
    bits = viterbi_decode(y_derotated)

    return bits_to_text(bits)
