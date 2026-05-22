"""
Test for the whole pipeline
v1 (maybe to modfy)

Run from terminal: python test_local.py
"""

import numpy as np
from transmitter import encode_message
from receiver import decode_message
from channel_sim import channel
from config import ALPHABET, N_CHARS

def test_full_chain():

    msg = "Hello World This Is A Test Message 1234"
    x = encode_message(msg)
    y = channel(x)
    decoded = decode_message(y)
    print(decoded)

    assert decoded == msg

# Pareil mais avec un random message en gros
def random_message():
    return ''.join(np.random.choice(list(ALPHABET), size=N_CHARS))

def test_full_channel():
    char_errors = 0
    perfect = 0
    N = 500
    energies = []
    
    for i in range(N):
        msg = random_message()
        x = encode_message(msg)
        energies.append(np.sum(x**2))
        y = channel(x)
        decoded = decode_message(y)
        errs = sum(a != b for a, b in zip(msg, decoded))
        char_errors += errs
        if errs == 0:
            perfect += 1
        else:
            print(f"[{i:3d}] sent   : '{msg}'")
            print(f"       decoded: '{decoded}'")
            print()
    
    print(f"Taux d'erreur caractère : {char_errors / (N_CHARS*N):.5f}")
    print(f"Transmissions parfaites : {perfect}/{N}")
    print(f"Énergie moyenne : {np.mean(energies):.2f}")

if __name__ == '__main__':
    test_full_channel()
