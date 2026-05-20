"""
Test for the whole pipeline
v1 (maybe to modfy)

Run from terminal: python test_local.py
"""

from transmitter import encode_message
from receiver import decode_message
from channel_sim import channel


def test_full_chain():

    msg = "Hello World This Is A Test Message 1234"
    x = encode_message(msg)
    y = channel(x)
    decoded = decode_message(y)
    print(decoded)

    assert decoded == msg