import pyrtl as rtl

from test import *


def flatten(l): return [val for sublist in l for val in sublist]

## 4x4 ##


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

flat_a = flatten(A)
flat_k = flatten(K)

row_r, col_r, bitwidth = len(A), len(K[0]), 8

a = rtl.Input(len(A) * len(A[0]) * bitwidth, 'A')
k = rtl.Input(len(K) * len(K[0]) * bitwidth, 'K')

result = rtl.WireVector(row_r * col_r * bitwidth, 'result')
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

for cycle in range(5):
    sim.step(sim_inputs)

sim_trace.render_trace()

# with open('vis.txt', 'w') as f:
#     rtl.output_to_graphviz(f)


# Analysis

# ta = rtl.TimingAnalysis()
# print(f'Max timing delay: {ta.max_length()} ps')

# print(f'Area: {sum(rtl.area_estimation())} mm^2')
