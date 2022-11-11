from utils import *

bits = 16
fractional_bits = 8
x = -12.1059

print(f'Bits: {bits} (fractional: {fractional_bits})')

print('\nInput:', x)
binary = float_to_binary(x, bits, fractional_bits)
print('Fixed-point representation:', binary)
print('Back to floating point:', binary_to_float(binary, bits, fractional_bits))
