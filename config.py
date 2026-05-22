# Constants

# === PRÉAMBULE ===
PREAMBLE_AMPLITUDE = 3.0
PREAMBLE_LENGTH    = 4

# === MODULATION BPSK ===
ENERGY_PER_QPSK_SYMBOL = 4.70                              # energy per pair of chips
QPSK_AMPLITUDE         = (ENERGY_PER_QPSK_SYMBOL / 2) ** 0.5   # Aq ≈ 1.533

# === CODE CONVOLUTIF (K=7, rate 1/2, NASA standard) ===
CONV_TAIL_BITS = 6   # tail zeros to flush the shift register back to state 0

# === CONTRAINTES ===
N_MAX      = 500
ENERGY_MAX = 1200
SIGMA      = 1.0
N_INFO_BITS  = 240
N_CHARS      = 40
BITS_PER_CHAR = 6

# Derived: (240 + 6) info+tail bits × 2 (rate 1/2) = 492 coded chips
N_CODED_BITS = (N_INFO_BITS + CONV_TAIL_BITS) * 2   # 492

# === ALPHABET ===
ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .'
