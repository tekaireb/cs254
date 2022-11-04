import pyrtl as rtl
from matplotlib import pyplot as plt

from conv_parallel import *


def flatten(l): return [val for sublist in l for val in sublist]


# Input
A = [
    [2, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 3, 0, 0]
]

# Kernel
K = [
    [1, 0, 1],
    [0, 0, 0],
    [0, 1, 0]
]

# TODO: Change binary conversion to use 2's complement to support negative numbers

# # Sobel Kernel
# K = [
#     [-1, 0, 1],
#     [-2, 0, 2],
#     [-1, 0, 1]
# ]

flat_a = flatten(A)
flat_k = flatten(K)

rows_r = len(A) - len(K) + 1
cols_r = len(A[0]) - len(K[0]) + 1
bitwidth = 8
row_r = cols_r * bitwidth

a = rtl.Input(len(A) * len(A[0]) * bitwidth, 'A')
k = rtl.Input(len(K) * len(K[0]) * bitwidth, 'K')

result = rtl.WireVector(rows_r * cols_r * bitwidth, 'result')
result <<= conv(a, k, len(A), len(A[0]), len(K), len(K[0]), bitwidth)


sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

# sim_inputs = {
#     'A': [int(''.join([f'{i:08b}' for i in flat_a]), 2)] * 10,
#     'B': [int(''.join([f'{i:08b}' for i in flat_k]), 2)] * 10
# }

# sim.step_multiple(sim_inputs)

sim_inputs = {
    'A': int(''.join([f'{i:08b}' for i in flat_a]), 2),
    'K': int(''.join([f'{i:08b}' for i in flat_k]), 2)
}

# Show initial image
plt.imshow(A, interpolation='nearest', cmap='gray')
plt.savefig('input.png')
plt.show()

for cycle in range(1):
    sim.step(sim_inputs)

    # Extract result (as matrix)
    raw = sim.value[result]
    raw_binary = f'{raw:b}'
    mod = len(raw_binary) % bitwidth
    if mod:
        raw_binary = '0' * (bitwidth - mod) + raw_binary

    result_matrix = [[None for c in range(cols_r)] for r in range(rows_r)]

    for r in range(rows_r):
        for c in range(cols_r):
            binary_value = raw_binary[row_r * r + bitwidth * c:
                                      row_r * r + bitwidth * (c + 1)]
            result_matrix[r][c] = int(binary_value, 2)

    print('[')
    for row in result_matrix:
        print(' ', row)
    print(']')

    # Show final image
    plt.imshow(result_matrix, interpolation='nearest', cmap='gray')
    plt.savefig('output.png')
    plt.show()

sim_trace.render_trace(trace_list=['A', 'K', 'result'])

# with open('vis.txt', 'w') as f:
#     rtl.output_to_graphviz(f)


# Analysis

# ta = rtl.TimingAnalysis()
# print(f'Max timing delay: {ta.max_length()} ps')

# print(f'Area: {sum(rtl.area_estimation())} mm^2')
