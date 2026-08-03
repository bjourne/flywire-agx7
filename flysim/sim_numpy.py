# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from flysim import EXT_BUF_ROWS, SimResult, compute_abg
from time import time

import numpy as np

def index_edges(edges):
    n_neurons = len(edges)
    index = np.empty(n_neurons + 1, dtype = np.int32)
    at = 0
    data = []
    for ix, es in enumerate(edges):
        index[ix] = at
        for dst, cnt in es:
            data.append(dst)
            data.append(cnt)
        at += len(es)
    index[n_neurons] = at
    assert 2 * at == len(data)

    assert data
    return index, np.array(data, dtype = np.int32)


def sim(ps):
    n_neurons = len(ps.v_init)
    alpha, beta, gamma = compute_abg(ps)

    fp_t = {
        16 : np.float16,
        32 : np.float32,
        64 : np.float64
    }[ps.fp_bits]

    alpha = fp_t(alpha)
    beta = fp_t(beta)
    gamma = fp_t(gamma)
    w_ext = fp_t(ps.w_ext)
    w_int = fp_t(ps.w_int)

    xs, ys = ps.ext_spikes
    ext_buf = np.zeros((EXT_BUF_ROWS, n_neurons), dtype = np.int8)
    ext_buf[ys, xs] = 1

    assert len(ps.edges) == ps.n_neurons

    index, data = index_edges(ps.edges)
    index = list(index)
    assert len(index) == ps.n_neurons + 1

    # State variables
    v = np.array(ps.v_init, dtype = fp_t)
    g = np.array(ps.g_init, dtype = fp_t)
    s = np.empty(n_neurons, dtype = np.int32)
    r = np.zeros(n_neurons, dtype = np.int32)

    int_buf = np.zeros((2 * ps.t_dly, n_neurons), dtype = np.int16)
    n_spikes = 0
    all_spikes = []
    spikes_per_neuron = np.zeros(n_neurons)
    start = time()
    for i in range(ps.n_ticks):
        int_row = i % ps.t_dly
        int_in = int_buf[int_row]

        # Aggregate incoming current
        ext_in = ext_buf[i % EXT_BUF_ROWS]

        inp = ext_in * w_ext + int_in * w_int

        vp = alpha * v + gamma * g
        gp = beta * g + inp
        not_ref = (r == 0)
        sub_v_th = (vp <= ps.v_th) & not_ref

        s[:] = (vp > ps.v_th) & not_ref
        v[:] = sub_v_th * vp
        g[:] = sub_v_th * gp

        r[:] = np.maximum(np.where(s, ps.t_ref, r) - 1, 0)

        spiking = np.nonzero(s)[0]
        all_spikes.append(list(spiking))
        n_spikes += len(spiking)
        spikes_per_neuron += s

        int_buf[int_row] = 0
        for v1 in spiking:
            i0 = index[v1]
            i1 = index[v1 + 1]

            dsts = data[2*i0:2*i1:2]
            cnts = data[2*i0+1:2*i1+1:2]
            np.add.at(int_buf[int_row], dsts, cnts)
    secs = time() - start
    return SimResult(
        n_real_secs = secs,
        n_spikes = n_spikes,
        spikes_per_neuron = spikes_per_neuron
    )
