"""
Unit tests for transmitter only
Run from terminal using: python test_transmitter.py
"""

import numpy as np

from transmitter import (
    build_hadamard_matrix,
    hadamard_encode_block,
    hadamard_encode_all,
    map_to_qpsk,
    build_preamble,
    encode_message
)

def test_hadamard_matrix():

    H = build_hadamard_matrix(3)

    assert H.shape == (8, 8)
    assert np.allclose(H @ H.T, 8 * np.eye(8))

def test_hadamard_encode_consistency():

    bits = np.array([1, 0, 1, 1])
    out = hadamard_encode_block(bits)

    assert out.shape == (8,)
    assert set(out).issubset({-1, 1})

def test_hadamard_encode_all():

    bits = np.random.randint(0, 2, 240)
    out = hadamard_encode_all(bits)

    assert out.shape == (480,)
    assert set(np.unique(out)).issubset({-1, 1})

def test_qpsk_mapping():

    coded = np.array([1, -1, -1, 1])
    Aq = 0.5
    out = map_to_qpsk(coded, Aq)
    expected = np.array([0.5, -0.5, -0.5, 0.5])

    assert np.allclose(out, expected)

def test_build_preamble():

    p = build_preamble(3.0, repeat=6)

    assert len(p) == 6
    assert np.allclose(p, 3.0)

def test_encode_constraints():

    msg = "Hello World This Is A Test Message 1234"
    x = encode_message(msg)

    assert len(x) <= 500
    assert len(x) % 2 == 0
    assert np.sum(x**2) <= 1200

if __name__ == "__main__":

    test_hadamard_matrix()
    print("test matrix");
    test_hadamard_encode_consistency()
    print("test hadamar consistency");
    test_hadamard_encode_all()
    print("test hadamar encode all");
    test_qpsk_mapping()
    print("test qpsk");
    test_build_preamble()
    print("test build preamble");
    test_encode_constraints()

    print("All transmitter tests passed.")