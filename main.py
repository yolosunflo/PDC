"""
main
first minimalist version (just to understand how the project is built)
Run from terminal : python main.py "Hello World This Is A Test Message 1234"
"""

import sys
import numpy as np

from transmitter import encode_message
from receiver import decode_message

from channel_sim import channel

def main():

    message = sys.argv[1]

    # ENCODE
    x = encode_message(message)

    print("Sent signal:")
    print(x)

    # CHANNEL
    y = channel(x)

    print("Received signal:")
    print(y)

    # DECODE
    decoded = decode_message(y)

    print("Decoded message:")
    print(decoded)


if __name__ == "__main__":
    main()