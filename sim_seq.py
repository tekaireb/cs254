import pyrtl as rtl
from matplotlib import pyplot as plt

from conv_sequential import conv


def flatten(l): return [val for sublist in l for val in sublist]


def int_to_binary(n, bits):
    s = bin(n & ((1 << bits) - 1))[2:]
    return f'{s:0>{bits}}'


def binary_to_int(n, bits):
    val = int(n, 2)
    if (val & (1 << (bits - 1))):
        val -= (1 << bits)
    return val


# Input
A = [
    [2, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 3, 0, 0]
]

a_len = len(A) * len(A[0])


# Kernel
K = [
    [1, 1, 1],
    [1, 3, 1],
    [1, 1, 1]
]

flat_a = flatten(A)
flat_k = flatten(K)
bitwidth = 8

mem_i = rtl.MemBlock(bitwidth=bitwidth, addrwidth=32,
                     name="input_img")
mem_k = rtl.MemBlock(bitwidth=bitwidth, addrwidth=32,
                     name="input_kernel")

rows_r = len(A) - len(K) + 1
cols_r = len(A[0]) - len(K[0]) + 1
row_r = cols_r * bitwidth

reset = rtl.Input(bitwidth=1, name="reset")
done = rtl.Output(bitwidth=1, name="done")

output_memory, complete = conv(mem_i, mem_k, reset, len(
    A), len(A[0]), len(K), len(K[0]), bitwidth)

done <<= complete

# Show initial image
plt.imshow(A, interpolation='nearest', cmap='gray')
plt.savefig('input.png')

print("IMAGE")
print(A)
print()
print("KERNEL")
print(K)
print()

# At first we load the kernle and images to their respective memories.
writing_image = rtl.Input(bitwidth=1, name="writing_image")
addr = rtl.Input(bitwidth=32, name="pixel_addr")
pixel = rtl.Input(bitwidth=bitwidth, name="pixel")
mem_i[addr] <<= rtl.MemBlock.EnabledWrite(pixel, writing_image)

writing_kernel = rtl.Input(bitwidth=1, name="writing_kernel")
addr = rtl.Input(bitwidth=32, name="kernel_addr")
pixel = rtl.Input(bitwidth=bitwidth, name="kernel_value")
mem_k[addr] <<= rtl.MemBlock.EnabledWrite(pixel, writing_kernel)


sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

for i in range(len(flat_a)):
    sim.step({"reset": 1,
              "pixel_addr": i,
              "pixel": flat_a[i],
              "kernel_addr": 0,
              "kernel_value": 0,
              "writing_image": True,
              "writing_kernel": False,
              })

for i in range(len(flat_k)):
    sim.step({"reset": 1,
              "kernel_addr": i,
              "kernel_value": flat_k[i],
              "pixel_addr": 0,
              "pixel": 0,
              "writing_image": False,
              "writing_kernel": True,
              })

while sim.inspect("done") == 0:
    sim.step({
        "reset": 0,
        "kernel_addr": 0,
        "kernel_value": 0,
        "pixel_addr": 0,
        "pixel": 0,
        "writing_image": False,
        "writing_kernel": False,
    })

sim_trace.render_trace(trace_list=["reset", "done", "a_row", "a_col", "focused_pixel_idx"])

output = sim.inspect_mem(output_memory)
convolution = [[0 for _ in range(len(A[0]))] for _ in range(len(A))]

for (pixel, value) in output.items():
    convolution[pixel // len(convolution[0])][pixel % len(convolution[0])] = value

print("CONVOLUTION")
print(convolution)

plt.imshow(convolution, interpolation="nearest", cmap="gray")
plt.savefig("output.png")
