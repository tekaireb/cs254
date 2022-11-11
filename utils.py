def flatten(l): return [val for sublist in l for val in sublist]


def int_to_binary(n, bits):
    s = bin(n & ((1 << bits) - 1))[2:]
    return f'{s:0>{bits}}'


def binary_to_int(n, bits):
    val = int(n, 2)
    if (val & (1 << (bits - 1))):
        val -= (1 << bits)
    return val


def float_to_fixedpoint(x_float, fractional_bits):
    return int(x_float * (2 ** fractional_bits))


def fixedpoint_to_float(x_fixedpoint, fractional_bits):
    return float(x_fixedpoint * (2 ** (-fractional_bits)))


def float_to_binary(x, bits, fractional_bits):
    fixedpoint_integer = float_to_fixedpoint(x, fractional_bits)
    return int_to_binary(fixedpoint_integer, bits)


def binary_to_float(x, bits, fractional_bits):
    fixedpoint_integer = binary_to_int(x, bits)
    return fixedpoint_to_float(fixedpoint_integer, fractional_bits)
