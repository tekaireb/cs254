import pyrtl as rtl
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
from functools import reduce

from conv_parallel import *
from utils import *

# # Input
# A = [
#     [2, 0, 1, 1],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 3, 0, 0]
# ]

# Input image
A = np.array(Image.open('images/desert_road_input.bmp')).tolist()

# Sobel G_x Kernel
Kx = [
    [+1, +2, +1],
    [+0, +0, +0],
    [-1, -2, -1]
]

# Sobel G_y Kernel
Ky = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]


flat_a = flatten(A)
flat_kx = flatten(Kx)
flat_ky = flatten(Ky)

rows_r = len(A) - len(Kx) + 1
cols_r = len(A[0]) - len(Kx[0]) + 1
bitwidth = 16
fractional_bits = 2
row_r = cols_r * bitwidth

a = rtl.Input(len(A) * len(A[0]) * bitwidth, 'A')
kx = rtl.Input(len(Kx) * len(Kx[0]) * bitwidth, 'Kx')
ky = rtl.Input(len(Ky) * len(Ky[0]) * bitwidth, 'Ky')

kx_res = rtl.WireVector(rows_r * cols_r * bitwidth, 'kx_res')
kx_res <<= conv(a, kx, len(A), len(A[0]), len(
    Kx), len(Kx[0]), bitwidth, fractional_bits)

ky_res = rtl.WireVector(rows_r * cols_r * bitwidth, 'ky_res')
ky_res <<= conv(a, ky, len(A), len(A[0]), len(
    Ky), len(Ky[0]), bitwidth, fractional_bits)

sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

sim_inputs = {
    'A': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_a]), 2),
    'Kx': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_kx]), 2),
    'Ky': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_ky]), 2)
}

# # Show initial image
# plt.imshow(A, interpolation='nearest', cmap='gray')
# plt.savefig('input.png')
# plt.show()

for cycle in range(1):
    sim.step(sim_inputs)

    # Extract result (as matrix)
    raw_kx = sim.value[kx_res]
    raw_ky = sim.value[ky_res]
    raw_kx_binary = f'{raw_kx:b}'
    raw_ky_binary = f'{raw_ky:b}'

    # Fill missing space with zeros if necessary
    dif = (rows_r * cols_r * bitwidth) - len(raw_kx_binary)
    raw_kx_binary = ('0' * dif) + raw_kx_binary
    dif = (rows_r * cols_r * bitwidth) - len(raw_ky_binary)
    raw_ky_binary = ('0' * dif) + raw_ky_binary

    kx_res_matrix = np.zeros((rows_r, cols_r))
    ky_res_matrix = np.zeros((rows_r, cols_r))

    for r in range(rows_r):
        for c in range(cols_r):
            # Kx
            binary_value = raw_kx_binary[row_r * r + bitwidth * c:
                                         row_r * r + bitwidth * (c + 1)]
            kx_res_matrix[r][c] = binary_to_float(
                binary_value, bitwidth, fractional_bits)
            # Ky
            binary_value = raw_ky_binary[row_r * r + bitwidth * c:
                                         row_r * r + bitwidth * (c + 1)]
            ky_res_matrix[r][c] = binary_to_float(
                binary_value, bitwidth, fractional_bits)

    # G = sqrt(G_x ^ 2 + G_y ^ 2)
    combined = np.sqrt(kx_res_matrix ** 2 + ky_res_matrix ** 2)

    # Normalize
    max_pixel = np.max(combined)
    normalization_factor = 255.0 / max_pixel
    result_matrix = combined * normalization_factor

    # Show final image
    plt.imshow(result_matrix, interpolation='nearest', cmap='gray')
    plt.savefig('sobel_result.png')
    plt.show()

# sim_trace.render_trace(trace_list=['A', 'K', 'result'])
