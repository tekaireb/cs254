import pyrtl as rtl


def flatten(l): return [val for sublist in l for val in sublist]


def kernel_mul(A, K, rows_a: int, cols_a: int, rows_k: int, cols_k: int, bitwidth: int):
    counter = rtl.Register(bitwidth=3, name='counter')
    counter.next <<= counter + 1

    val = rtl.Register(len(A), 'val')
    res = rtl.Register(1, 'res')

    res.next <<= val[0]

    with rtl.conditional_assignment:
        with counter == 0:
            val.next |= A
        with counter > 0:
            val.next |= rtl.shift_right_logical(val, rtl.Const(1))

    return res


# Input
A = [
    [2, 0, 1],
    [0, 1, 0],
    [0, 0, 1]
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
result <<= kernel_mul(a, k, len(A), len(A[0]), len(K), len(K[0]), bitwidth)

sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

# sim_inputs = {
#     'A': int(''.join([f'{i:08b}' for i in flat_a]), 2),
#     'K': int(''.join([f'{i:08b}' for i in flat_k]), 2)
# }

sim_inputs = {
    'A': 0b10101010101010,
    'K': 0b01010101010101
}

for cycle in range(10):
    sim.step(sim_inputs)

sim_trace.render_trace()
