import pyrtl as rtl


def conv(A,
         K,
         reset: rtl.WireVector,
         rows_a: int,
         cols_a: int,
         rows_k: int,
         cols_k: int,
         bitwidth: int,
         fractional_bits: int):

    # ／(^ㅅ^)＼ adjust floating-point bitshift
    def fp_adjust(x):
        if fractional_bits:
            return (rtl.shift_right_arithmetic(x, fractional_bits))
        else:
            return x

    # ／(^ㅅ^)＼ Define a memory block for the output image
    output_img = rtl.MemBlock(bitwidth=bitwidth,
                              addrwidth=A.addrwidth, name='output_img')

    # ／(^ㅅ^)＼ Define registers to keep track of rows and columns
    a_row = rtl.Register(bitwidth=16, name='a_row', reset_value=rows_k // 2)
    a_col = rtl.Register(bitwidth=16, name='a_col', reset_value=cols_k // 2)
    k_row = rtl.Register(bitwidth=16, name='k_row')
    k_col = rtl.Register(bitwidth=16, name='k_col')

    # ／(^ㅅ^)＼ Define registers to hold the result of the multiplication, the result of the addition,
    # and the completed value
    mult_res = rtl.Register(bitwidth=bitwidth, name='mult_res')
    aggregator = rtl.Register(bitwidth=bitwidth, name='aggregator')
    complete = rtl.Register(bitwidth=1, name='complete')

    # ／(^ㅅ^)＼ Define value of last row
    last_row = rows_a - rows_k // 2
    last_col = cols_a - cols_k // 2

    # ／(^ㅅ^)＼ Get next row
    next_a_row = rtl.WireVector(bitwidth=16, name='next_a_row')
    next_a_col = rtl.WireVector(bitwidth=16, name='next_a_col')
    next_k_row = rtl.WireVector(bitwidth=16, name='next_k_row')
    next_k_col = rtl.WireVector(bitwidth=16, name='next_k_col')

    # ／(^ㅅ^)＼ Get focused pixel/kernel from the register
    focused_pixel_idx = rtl.Register(bitwidth=A.addrwidth, name='focused_pixel_idx')
    focused_kernel_idx = rtl.Register(bitwidth=K.addrwidth, name='focused_kernel_idx')

    # ／(^ㅅ^)＼ Calculate the next focused pixel/kernel index
    focused_pixel_idx.next <<= (
                (next_a_row + next_k_row - rows_k // 2) * cols_a + (next_a_col + next_k_col - cols_k // 2))
    focused_kernel_idx.next <<= (next_k_row * cols_k + next_k_col)

    # ／(^ㅅ^)＼ Assign focused pixel/kernel
    focused_pixel = A[focused_pixel_idx]
    focused_kernel = K[focused_kernel_idx]

    # ／(^ㅅ^)＼ Assign next values of rows and columns
    k_row.next <<= next_k_row
    k_col.next <<= next_k_col
    a_row.next <<= next_a_row
    a_col.next <<= next_a_col

    with rtl.conditional_assignment:
        # ／(^ㅅ^)＼ Initial values
        with reset:
            next_k_row |= 0
            next_k_col |= 0
            next_a_row |= a_row.reset_value
            next_a_col |= a_col.reset_value
            mult_res.next |= 0
            aggregator.next |= 0
            complete.next |= 0
        # ／(^ㅅ^)＼ For every other cycle
        with rtl.otherwise:
            # ／(^ㅅ^)＼ If we're at the final value in the kernel
            with (k_row == rows_k - 1) & (k_col == cols_k - 1):
                next_k_row |= 0
                next_k_col |= 0
                next_a_row |= rtl.select(a_col == last_col - 1, a_row + 1, a_row)
                next_a_col |= rtl.select(a_col == last_col - 1, a_col.reset_value, a_col + 1)
                complete.next |= rtl.select((a_row == last_row - 1) &
                                        (a_col == last_col - 1),
                                        True, complete)
                mult_res.next |= 0
                aggregator.next |= 0
                # ／(^ㅅ^)＼ mult_res must be added here in the final cycle for each kernel
                output_img[rtl.truncate(
                    a_row * cols_a + a_col, output_img.addrwidth
                    )] |= rtl.truncate(aggregator + fp_adjust(focused_pixel * focused_kernel) + mult_res, output_img.bitwidth)
            # ／(^ㅅ^)＼ If we're not at the final value in the kernel
            with rtl.otherwise:
                next_a_row |= a_row
                next_a_col |= a_col
                next_k_row |= rtl.select(k_col == cols_k - 1, k_row + 1, k_row)
                next_k_col |= rtl.select(k_col == cols_k - 1, 0, k_col + 1)
                # ／(^ㅅ^)＼ Multiplication result
                mult_res.next |= fp_adjust(focused_pixel * focused_kernel)
                # ／(^ㅅ^)＼ Addition result
                aggregator.next |= aggregator + mult_res
                complete.next |= complete

    return output_img, complete
