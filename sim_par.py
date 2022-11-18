import pyrtl as rtl
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

import conv_parallel
import conv_parallel_improved
from utils import *

# Test inputs
# A = np.ones((4, 4)).tolist()
# A = np.ones((8, 8)).tolist()
# A = np.ones((16, 16)).tolist()
# A = np.ones((32, 32)).tolist()
A = np.ones((64, 64)).tolist()
# A = np.ones((128, 128)).tolist()

# # Input
# A = [
#     [2, 0, 1, 1],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 3, 0, 0]
# ]

# A = np.array(Image.open('images/desert_road_input.bmp')).tolist()

# # Kernel
# K = [
#     [1, 0, 1],
#     [0, 0, 0],
#     [0, 1, 0]
# ]

# # Sobel Kernel
# K = [
#     [-1/2, 0, 1/2],
#     [-1, 0, 1],
#     [-1/2, 0, 1/2]
# ]

# # Gaussian Blur Kernel
# K = [
#     [1/16, 2/16, 1/16],
#     [2/16, 4/16, 2/16],
#     [1/16, 2/16, 1/16]
# ]

# Test kernels
K = np.ones((3, 3))
# K = np.ones((4, 4))
# K = np.ones((5, 5))
# K = np.ones((6, 6))

flat_a = flatten(A)
flat_k = flatten(K)

rows_r = len(A) - len(K) + 1
cols_r = len(A[0]) - len(K[0]) + 1
bitwidth = 16
fractional_bits = 2
row_r = cols_r * bitwidth

a = rtl.Input(len(A) * len(A[0]) * bitwidth, 'A')
k = rtl.Input(len(K) * len(K[0]) * bitwidth, 'K')

result = rtl.WireVector(rows_r * cols_r * bitwidth, 'result')
result <<= conv_parallel_improved.conv(a, k, len(A), len(A[0]), len(
    K), len(K[0]), bitwidth, fractional_bits)


sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

# sim_inputs = {
#     'A': int(''.join([int_to_binary(i, bitwidth) for i in flat_a]), 2),
#     'K': int(''.join([int_to_binary(i, bitwidth) for i in flat_k]), 2)
# }

sim_inputs = {
    'A': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_a]), 2),
    'K': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_k]), 2)
}

# # Show initial image
# plt.imshow(A, interpolation='nearest', cmap='gray')
# plt.savefig('input.png')
# plt.show()

for cycle in range(1):
    sim.step(sim_inputs)

    # Extract result (as matrix)
    raw = sim.value[result]
    # print(raw)
    raw_binary = f'{raw:b}'

    # Fill missing space with zeros if necessary
    dif = (rows_r * cols_r * bitwidth) - len(raw_binary)
    raw_binary = ('0' * dif) + raw_binary

    result_matrix = [[None for c in range(cols_r)] for r in range(rows_r)]

    for r in range(rows_r):
        for c in range(cols_r):
            binary_value = raw_binary[row_r * r + bitwidth * c:
                                      row_r * r + bitwidth * (c + 1)]
            # result_matrix[r][c] = binary_to_int(binary_value, bitwidth)
            result_matrix[r][c] = binary_to_float(
                binary_value, bitwidth, fractional_bits)

    # # Print result matrix
    # print('[')
    # for row in result_matrix:
    #     print(' ', row)
    # print(']')

    # # Show final image
    # plt.imshow(result_matrix, interpolation='nearest', cmap='gray')
    # plt.savefig('output.png')
    # plt.show()

# sim_trace.render_trace(trace_list=['A', 'K', 'result'])

# with open('vis.txt', 'w') as f:
#     rtl.output_to_graphviz(f)


# Analysis

ta = rtl.TimingAnalysis()
print(f'Max timing delay: {ta.max_length()} ps')

print(f'Area: {sum(rtl.area_estimation())} mm^2')
