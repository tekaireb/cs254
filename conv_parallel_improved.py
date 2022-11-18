import pyrtl as rtl
from adders import *


def conv(A, K, rows_a: int, cols_a: int, rows_k: int, cols_k: int, bitwidth: int, fractional_bits=0):
    # Perform shift to adjust for fixed-point multiplication (if necessary)
    def fp_adjust(x):
        if fractional_bits:
            return (rtl.shift_right_arithmetic(x, fractional_bits))
        else:
            return x

    row_a = cols_a * bitwidth
    row_k = cols_k * bitwidth

    result = []

    # Convert flattened A and kernel into 2D array of WireVectors

    # a = [[rtl.WireVector(bitwidth, f'a({r}, {c})')
    #       for c in range(cols_a)] for r in range(rows_a)]

    # k = [[rtl.WireVector(bitwidth, f'k({r}, {c})')
    #       for c in range(cols_k)] for r in range(rows_k)]

    a = [[rtl.WireVector(bitwidth) for c in range(cols_a)]
         for r in range(rows_a)]

    k = [[rtl.WireVector(bitwidth) for c in range(cols_k)]
         for r in range(rows_k)]

    for r in range(rows_a):
        for c in range(cols_a):
            a[r][c] <<= A[row_a * r + bitwidth * c:
                          row_a * r + bitwidth * (c + 1)]

    for r in range(rows_k):
        for c in range(cols_k):
            k[r][c] <<= K[row_k * r + bitwidth * c:
                          row_k * r + bitwidth * (c + 1)]

    # Calculate sum of elementwise products for each output pixel

    for r in range(rows_a - rows_k + 1):
        for c in range(cols_a - cols_k + 1):
            ew_product = [fp_adjust(a[r + i][c + j] * k[i][j])
                          for j in range(len(k[0])) for i in range(len(k))]

            # s = rtl.WireVector(bitwidth, f'res({r}, {c})')
            s = rtl.WireVector(bitwidth)
            s <<= fast_group_adder(ew_product)

            result.append(s)

    # Combine list of output pixels into singular flat WireVector
    return rtl.concat_list(result)
