"""
main
first minimalist version (just to understand how the project is built)
Run from terminal : python main.py "Hello World This Is A Test Message 1234"
"""

import subprocess # ?????????????????????????????????? on a le droit?
import sys
import numpy as np

from transmitter import encode_message
from receiver import decode_message

from channel_sim import channel

def write_input(x, filename='input.txt'):
    np.savetxt(filename, x, fmt='%.10f')

def read_output(filename='output.txt'):
    return np.loadtxt(filename)

def transmit_via_server():
    subprocess.run([
        'python3', 'client.py',
        '--input_file', 'input.txt',
        '--output_file', 'output.txt',
        '--srv_hostname=iscsrv72.epfl.ch',
        '--srv_port=80'
    ], check=True)

# Put '--local' in command line args to test local channel simulation 
# instead of server communication.
def main():
    message = sys.argv[1]
    use_local = '--local' in sys.argv
    
    x = encode_message(message)
    print(f"n = {len(x)}, energy = {np.sum(x**2):.2f}")
    
    if use_local:
        from channel_sim import channel
        y = channel(x)
    else:
        write_input(x)
        transmit_via_server()
        y = read_output()
    
    decoded = decode_message(y)
    errors = sum(a != b for a, b in zip(message, decoded))
    print(f"Sent    : '{message}'")
    print(f"Decoded : '{decoded}'")
    print(f"Errors  : {errors}/{len(message)}")
# def main():

#     message = sys.argv[1]

#     # ENCODE
#     x = encode_message(message)

#     print("Sent signal:")
#     print(x)

#     # CHANNEL
#     y = channel(x)

#     print("Received signal:")
#     print(y)

#     # DECODE
#     decoded = decode_message(y)

#     print("Decoded message:")
#     print(decoded)

if __name__ == "__main__":
    main()
