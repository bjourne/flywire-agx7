# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from flysim import (
    EXT_BUF_ROWS,
    SimResult,
    cache_to_file,
    compute_abg
)
from myopencl.objs import Context
from myopencl.utils import format_opts

import ctypes
import myopencl as cl
import numpy as np

########################################################################
# OpenCL utils
########################################################################
def write_np_array(ctx, qname, bname, x):
    assert x.flags["C_CONTIGUOUS"]
    assert x.data.contiguous
    if x.ndim > 1:
        assert x.strides[0] == x.shape[-1] * x.itemsize
    ptr = x.ctypes.data_as(ctypes.c_void_p)
    return ctx.write_buffer(qname, bname, x.nbytes, ptr)

def read_np_array(ctx, qname, bname, x):
    ptr = x.ctypes.data_as(ctypes.c_void_p)
    return ctx.read_buffer(qname, bname, x.nbytes, ptr)

class cl_half(ctypes.Structure):
    _fields_ = [("bits", ctypes.c_uint16)]
    def __init__(self, value):
        self.bits = unpack('H', pack('e', float(value)))[0]

    def to_float(self):
        return unpack('e', pack('H', self.bits))[0]

    def __float__(self):
        return self.to_float()

    def __repr__(self):
        return f"float16({self.to_float()})"

########################################################################
# Block alignment
########################################################################
def block_count(x, y):
    return (x + y - 1) // y

def align_to(x, y):
    return block_count(x, y) * y

def n_neu_blocks(n_neurons, neu_align):
    return block_count(n_neurons + neu_align, neu_align)

########################################################################
# Misc utils
########################################################################
@cache_to_file(lambda edges, k, m: [len(edges), k, m])
def create_aligned_csr(edges, k, m):
    assert len(edges) % k == 0
    dummy_blk = (len(edges) - k) // k

    n_neurons = len(edges)
    index = [0] * (n_neurons + 1)
    syns = []
    n_old = sum(len(e) for e in edges)
    for ix, es in enumerate(edges):
        es = pad_vertices(es, k, m, dummy_blk)
        es = [(cnt << 18) | dst for (dst, cnt) in es]
        syns.extend(es)
        index[ix + 1] = index[ix] + len(es) // (k * m)

    n_new = k*m*index[n_neurons]
    pct = 100 * (n_new / n_old)
    print("== Index Expansion ==")
    print("  From %d to %d synapes (%d%% growth)" % (n_old, n_new, pct))
    print("  k = %d, m = %d" % (k, m))

    # OpenCL dislikes empty arrays
    if not syns:
        syns.append(0)

    index = np.array(index, dtype = np.uint32)
    syns = np.array(syns, dtype = np.int32)
    return index, syns

def get_preprocessor_defines(
        ps, neu_align, syn_align, syn_grp_align, n_lanes,
        is_aoc
):
    blocks = n_neu_blocks(ps.n_neurons, neu_align)
    return [
        f"EXT_BUF_ROWS={EXT_BUF_ROWS}",
        f"N_NEU_BLOCKS={blocks}",
        f"NEU_ALIGN={neu_align}",
        f"SYN_ALIGN={syn_align}",
        f"SYN_GRP_ALIGN={syn_grp_align}",
        f"N_LANES={n_lanes}",
        f"T_REF={ps.t_ref}",
        f"T_DLY={ps.t_dly}",
        f"FP_BITS={ps.fp_bits}",
        f"IS_AOC={1 if is_aoc else 0}"
    ]

########################################################################
# Main entry point
########################################################################
def sim(
        platform_index, ocl_path,
        neu_align, syn_align, syn_grp_align, n_lanes,
        ps
):
    # FP type
    tps = {
        16 : (np.float16, cl_half),
        32 : (np.float32, cl.c_float),
        64 : (np.float64, cl.c_double)
    }

    np_fp_t, ct_fp_t = tps[ps.fp_bits]

    alpha, beta, gamma = compute_abg(ps)

    # Padded neuron count and edges
    n_padded = n_neu_blocks(ps.n_neurons, neu_align) * neu_align
    n_pad = n_padded - ps.n_neurons
    edges = ps.edges + [[]] * (n_padded - ps.n_neurons)

    assert len(edges) == n_padded

    # Store edges in CSR format
    src_ix, src_syn = create_aligned_csr(edges, syn_align, syn_grp_align)

    left_pad = [0, n_pad]

    vs_init = np.pad(ps.v_init, left_pad).astype(np_fp_t)
    gs_init = np.pad(ps.g_init, left_pad).astype(np_fp_t)
    assert len(vs_init) == n_padded

    ret = np.zeros(1 + n_padded, dtype = np.uint32)

    # External
    xs, ys = ps.ext_spikes
    ext_buf = np.zeros((EXT_BUF_ROWS, n_padded), dtype = np.int8)
    ext_buf[ys, xs] = 1

    ctx = Context.from_indexes(platform_index, 0)

    # Setup buffer args
    props = [
        cl.CommandQueueInfo.CL_QUEUE_PROPERTIES,
        cl.CommandQueueProperties.CL_QUEUE_PROFILING_ENABLE
    ]
    ctx.register_queue("main", props)

    rw_flag = cl.MemFlags.CL_MEM_READ_WRITE
    ro_flag = cl.MemFlags.CL_MEM_READ_ONLY
    bufs = [
        ("gl_vs_init", vs_init, False),
        ("gl_gs_init", gs_init, False),

        ("gl_ext", ext_buf, False),

        ("gl_ix", src_ix, False),
        ("gl_syn", src_syn, False),

        ("gl_ret", ret, True)
    ]
    for name, arr, writable in bufs:
        buf_size = arr.nbytes
        flag = rw_flag if writable else ro_flag
        ctx.register_buffer(name, buf_size, flag)
        ev = write_np_array(ctx, "main", name, arr)
        cl.wait_for_events([ev])

    # We pass constant integer parameters as pre-processor defines
    defines = get_preprocessor_defines(
        ps, neu_align, syn_align, syn_grp_align, n_lanes, False
    )

    # Automatically include the kernel's own directory.
    includes = [ocl_path.parent]

    src = [ocl_path]
    opts = format_opts([
        "-cl-unsafe-math-optimizations",
        "-cl-std=CL2.0"
    ], includes, defines)

    print("== Kernel Compilation ==")
    print("  Compiling with %s" % opts)

    ctx.register_program("sim", src, opts)
    ev = ctx.run_kernel("main", "sim", "sim", [1], None, [
        (cl.c_uint, ps.n_ticks),
        (ct_fp_t, alpha),
        (ct_fp_t, beta),
        (ct_fp_t, gamma),
        (ct_fp_t, ps.v_th),
        (ct_fp_t, ps.w_int),
        (ct_fp_t, ps.w_ext),

        "gl_vs_init", "gl_gs_init",
        "gl_ext",
        "gl_ix", "gl_syn",
        "gl_ret"
    ])
    cl.wait_for_events([ev])

    attr_start = cl.ProfilingInfo.CL_PROFILING_COMMAND_START
    attr_end = cl.ProfilingInfo.CL_PROFILING_COMMAND_END
    start = cl.get_info(attr_start, ev)
    end = cl.get_info(attr_end, ev)
    secs = (end - start) * 1.0e-9
    cl.wait_for_events([read_np_array(ctx, "main", "gl_ret", ret)])

    return SimResult(
        n_real_secs = secs,
        n_spikes = ret[0],
        spikes_per_neuron = list(ret[1:1+ps.n_neurons])
    )
