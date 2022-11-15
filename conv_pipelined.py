import pyrtl as rtl

def convPar(A,
            K,
            rows_a: int,
            cols_a: int,
            rows_k: int,
            cols_k: int,
            bitwidth: int,
            reset):
    output_img = rtl.MemBlock(bitwidth = 2 * bitwidth + 1,
                              addrwidth = 32, name = 'output_img')

    row_a = rtl.Register(bitwidth=16, name='row_a', reset_value=rows_k//2)
    col_a = rtl.Register(bitwidth=16, name='col_a', reset_value=cols_k//2)
    row_k = rtl.Register(bitwidth=16, name='row_k')
    col_k = rtl.Register(bitwidth=16, name='col_k')

    mult_res = rtl.Register(bitwidth=2*bitwidth+1, name='mult_res')
    aggregator = rtl.Register(bitwidth=2*bitwidth+1, name='aggregator')
    complete = rtl.Register(bitwidth=1, name='complete')

    last_row = rows_a - rows_k // 2
    last_col = cols_a - cols_k // 2

    next_a_row = rtl.WireVector(bitwidth=16, name='next_a_row')
    next_a_col = rtl.WireVector(bitwidth=16, name='next_a_col')
    next_k_row = rtl.WireVector(bitwidth=16, name='next_k_row')
    next_k_col = rtl.WireVector(bitwidth=16, name='next_k_col')

    focused_pixel_idx = rtl.Register(bitwidth=A.addrwidth, name='focused_pixel_idx')
    focused_kernel_idx = rtl.Register(bitwidth=K.addrwidth, name='focused_kernel_idx')

    focused_pixel_idx.next <<= ((next_a_row + next_k_row - rows_k // 2) * cols_a + (next_a_col + next_k_col - cols_k // 2))
    focused_kernel_idx.next <<= (next_k_row * cols_k + next_k_col)

    focused_pixel = A[focused_pixel_idx]
    focused_kernel = K[focused_kernel_idx]

    row_k.next <<= next_k_row
    col_k.next <<= next_k_col
    row_a.next <<= next_a_row
    col_a.next <<= next_a_col

    with rtl.conditional_assignment:
        with reset:
            next_k_row |= 0
            next_k_col |= 0
            next_a_row |= row_a.reset_value
            next_a_col |= col_a.reset_value
            aggregator.next |= 0
            complete.next |= 0
        with rtl.otherwise:
            with (row_k == rows_k - 1) & (col_k == cols_k - 1):
                next_k_row |= 0
                next_k_col |= 0
                next_a_row |= rtl.select(col_a == last_col - 1, row_a + 1, row_a)
                next_a_col |= rtl.select(col_a == last_col - 2, col_a.reset_value, col_a+1)
                complete.next |= rtl.select((row_a == last_row - 1) &
                                            (col_a == last_col - 1),
                                            True, complete)
                aggregator.next |= 0
                output_img[rtl.truncate(row_a * cols_a + col_a, output_img.addrwidth)] |= aggregator
            with rtl.otherwise:
                next_a_row |= row_a
                next_a_col |= col_a
                next_k_row |= rtl.select(col_k == cols_k -1, row_k + 1, row_k)
                next_k_col |= rtl.select(col_k == cols_k - 1, 0, col_k + 1)
                mult_res.next |= focused_pixel * focused_kernel
                aggregator.next |= aggregator + mult_res
                complete.next |= complete

    return output_img, complete


