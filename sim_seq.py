import pyrtl as rtl
from matplotlib import pyplot as plt
from utils import *
from PIL import Image
import numpy as np

from conv_sequential import conv

# Input
# A = [
#     [2, 0, 1, 1],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [0, 3, 0, 0]
# ]

A = np.array(Image.open('images/desert_road_input.bmp')).tolist()

a_len = len(A) * len(A[0])

# Kernel
# K = [
#     [1, 1, 1],
#     [1, 3, 1],
#     [1, 1, 1]
# ]

# Sobel Kernel
K = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

flat_a = flatten(A)
flat_k = flatten(K)
bitwidth = 16
fractional_bits = 2

mem_i = rtl.MemBlock(bitwidth=bitwidth, addrwidth=32,
                     name="input_img")
mem_k = rtl.MemBlock(bitwidth=bitwidth, addrwidth=32,
                     name="input_kernel")

reset = rtl.Input(bitwidth=1, name="reset")
done = rtl.Output(bitwidth=1, name="done")

output_memory, complete = conv(mem_i, mem_k, reset, len(
    A), len(A[0]), len(K), len(K[0]), bitwidth, fractional_bits)

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

# At first we load the kernel and images to their respective memories.
writing_image = rtl.Input(bitwidth=1, name="writing_image")
addr = rtl.Input(bitwidth=32, name="pixel_addr")
pixel = rtl.Input(bitwidth=bitwidth, name="pixel")
mem_i[addr] <<= rtl.MemBlock.EnabledWrite(pixel, writing_image)

writing_kernel = rtl.Input(bitwidth=1, name="writing_kernel")
addr = rtl.Input(bitwidth=32, name="kernel_addr")
pixel = rtl.Input(bitwidth=bitwidth, name="kernel_value")
mem_k[addr] <<= rtl.MemBlock.EnabledWrite(pixel, writing_kernel)

sim_inputs = {
    'A': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_a]), 2),
    'K': int(''.join([float_to_binary(i, bitwidth, fractional_bits) for i in flat_k]), 2)
}

sim_trace = rtl.SimulationTrace()
sim = rtl.Simulation(tracer=sim_trace)

for i in range(len(flat_a)):
    sim.step({"reset": 1,
              "pixel_addr": i,
              "pixel": int(float_to_binary(flat_a[i], bitwidth, fractional_bits),2),
              "kernel_addr": 0,
              "kernel_value": 0,
              "writing_image": True,
              "writing_kernel": False,
              })

for i in range(len(flat_k)):
    sim.step({"reset": 1,
              "kernel_addr": i,
              "kernel_value": int(float_to_binary(flat_k[i], bitwidth, fractional_bits),2),
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
    convolution[pixel // len(convolution[0])][pixel % len(convolution[0])] = binary_to_float(f'{value:b}', bitwidth, fractional_bits)

print("CONVOLUTION")
print(convolution)

plt.imshow(convolution, interpolation="nearest", cmap="gray")
plt.savefig("output_seq.png")
