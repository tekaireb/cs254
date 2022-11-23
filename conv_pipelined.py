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

    def fp_adjust(x):
        if fractional_bits:
            return (rtl.shift_right_arithmetic(x, fractional_bits))
        else:
            return x

    output_img = rtl.MemBlock(bitwidth=bitwidth,
                              addrwidth=A.addrwidth, name='output_img')

    a_row = rtl.Register(bitwidth=16, name='a_row', reset_value=rows_k // 2)
    a_col = rtl.Register(bitwidth=16, name='a_col', reset_value=cols_k // 2)
    k_row = rtl.Register(bitwidth=16, name='k_row')
    k_col = rtl.Register(bitwidth=16, name='k_col')

    mult_res = rtl.Register(bitwidth=bitwidth, name='mult_res')
    aggregator = rtl.Register(bitwidth=bitwidth, name='aggregator')
    complete = rtl.Register(bitwidth=1, name='complete')

    last_row = rows_a - rows_k // 2
    last_col = cols_a - cols_k // 2

    next_a_row = rtl.WireVector(bitwidth=16, name='next_a_row')
    next_a_col = rtl.WireVector(bitwidth=16, name='next_a_col')
    next_k_row = rtl.WireVector(bitwidth=16, name='next_k_row')
    next_k_col = rtl.WireVector(bitwidth=16, name='next_k_col')

    focused_pixel_idx = rtl.Register(bitwidth=A.addrwidth, name='focused_pixel_idx')
    focused_kernel_idx = rtl.Register(bitwidth=K.addrwidth, name='focused_kernel_idx')

    focused_pixel_idx.next <<= (
                (next_a_row + next_k_row - rows_k // 2) * cols_a + (next_a_col + next_k_col - cols_k // 2))
    focused_kernel_idx.next <<= (next_k_row * cols_k + next_k_col)

    focused_pixel = A[focused_pixel_idx]
    focused_kernel = K[focused_kernel_idx]

    k_row.next <<= next_k_row
    k_col.next <<= next_k_col
    a_row.next <<= next_a_row
    a_col.next <<= next_a_col

    # add_once = rtl.Register(bitwidth=1, name="add_once")

    with rtl.conditional_assignment:
        with reset:
            next_k_row |= 0
            next_k_col |= 0
            next_a_row |= a_row.reset_value
            next_a_col |= a_col.reset_value
            # add_once.next |= 1
            mult_res.next |= 0
            aggregator.next |= 0
            complete.next |= 0
        with rtl.otherwise:
            with (k_row == rows_k - 1) & (k_col == cols_k - 1):
                # with add_once:
                #     aggregator.next |= aggregator + mult_res
                #     add_once.next |= 0
                # with rtl.otherwise:
                next_k_row |= 0
                next_k_col |= 0
                next_a_row |= rtl.select(a_col == last_col - 1, a_row + 1, a_row)
                next_a_col |= rtl.select(a_col == last_col - 1, a_col.reset_value, a_col + 1)
                complete.next |= rtl.select((a_row == last_row - 1) &
                                        (a_col == last_col - 1),
                                        True, complete)
                mult_res.next |= 0
                aggregator.next |= 0
                # add_once.next |= 1
                output_img[rtl.truncate(
                    a_row * cols_a + a_col, output_img.addrwidth
                    )] |= rtl.truncate(aggregator + fp_adjust(focused_pixel * focused_kernel), output_img.bitwidth)
            with rtl.otherwise:
                next_a_row |= a_row
                next_a_col |= a_col
                next_k_row |= rtl.select(k_col == cols_k - 1, k_row + 1, k_row)
                next_k_col |= rtl.select(k_col == cols_k - 1, 0, k_col + 1)
                mult_res.next |= fp_adjust(focused_pixel * focused_kernel)
                aggregator.next |= aggregator + mult_res
                complete.next |= complete

    return output_img, complete
