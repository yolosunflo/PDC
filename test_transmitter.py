"""
Unit tests for the convolutional transmitter.
Run from terminal: python test_transmitter.py
"""

import numpy as np
from transmitter import conv_encode, map_bpsk, build_preamble, encode_message
from config import N_INFO_BITS, N_CODED_BITS, CONV_TAIL_BITS, QPSK_AMPLITUDE


def test_conv_encode_length():
    bits = np.random.randint(0, 2, N_INFO_BITS)
    out  = conv_encode(bits)
    assert len(out) == N_CODED_BITS, f"Expected {N_CODED_BITS}, got {len(out)}"


def test_conv_encode_binary():
    bits = np.random.randint(0, 2, N_INFO_BITS)
    out  = conv_encode(bits)
    assert set(np.unique(out)).issubset({0, 1})


def test_conv_encode_all_zeros():
    # All-zero input → all-zero output (both generators sum to 0 mod 2)
    bits = np.zeros(N_INFO_BITS, dtype=int)
    out  = conv_encode(bits)
    assert np.all(out == 0), "All-zero input must produce all-zero codeword."


def test_conv_tail_terminates():
    # The last CONV_TAIL_BITS*2 output bits come from the tail zeros.
    # For an all-zero input the tail output must also be all zero.
    bits = np.zeros(N_INFO_BITS, dtype=int)
    out  = conv_encode(bits)
    tail = out[-(CONV_TAIL_BITS * 2):]
    assert np.all(tail == 0)


def test_bpsk_mapping():
    coded    = np.array([0, 1, 0, 1])
    chips    = map_bpsk(coded, Aq=1.0)
    expected = np.array([1.0, -1.0, 1.0, -1.0])
    assert np.allclose(chips, expected)


def test_build_preamble():
    p = build_preamble(3.0, repeat=4)
    assert len(p) == 4
    assert np.allclose(p, 3.0)


def test_encode_constraints():
    msg = "Hello World This Is A Test Message 1234."   # 40 chars
    x   = encode_message(msg)
    assert len(x) <= 500
    assert len(x) % 2 == 0
    assert np.sum(x**2) <= 1200


if __name__ == "__main__":
    test_conv_encode_length()
    print("test conv encode length")
    test_conv_encode_binary()
    print("test conv encode binary")
    test_conv_encode_all_zeros()
    print("test conv encode all zeros")
    test_conv_tail_terminates()
    print("test conv tail terminates")
    test_bpsk_mapping()
    print("test bpsk mapping")
    test_build_preamble()
    print("test build preamble")
    test_encode_constraints()
    print("test encode constraints")
    print("All transmitter tests passed.")
