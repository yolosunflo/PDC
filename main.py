import subprocess
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
        'python', 'client.py',
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

if __name__ == "__main__":
    main()
