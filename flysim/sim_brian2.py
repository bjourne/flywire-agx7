# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from flysim import EXT_BUF_ROWS, SimResult
from math import ceil
from time import time

import numpy as np

def sim(ps):
    #  I don't like imports that take a long time.
    from brian2 import (
        Hz, NeuronGroup, Network, PoissonInput,
        SpikeGeneratorGroup, SpikeMonitor, StateMonitor, Synapses,
        mV, ms
    )
    n_neurons = len(ps.v_init)

    EQS = """
    dv/dt = (g - v) / tau_m  : volt (unless refractory)
    dg/dt = -g / tau_g       : volt (unless refractory)
    rfc                      : second
    """
    assert n_neurons == len(ps.g_init)
    ng = NeuronGroup(
        N = n_neurons,
        # Fucker don't change this!
        method = "exact",
        model = EQS,
        refractory = "rfc",
        reset = "v = 0 * mV; g = 0 * mV",
        threshold = "v > v_th"
    )
    ng.v = ps.v_init * mV
    ng.g = ps.g_init * mV
    ng.rfc = ps.t_ref * ps.dt * ms

    syn_int = Synapses(
        ng, ng,
        "w : volt",
        on_pre="g += w",
        delay = ps.t_dly * ps.dt * ms
    )

    syn_i = []
    syn_j = []
    syn_c = []

    for v1, v2s in enumerate(ps.edges):
        for v2, c in v2s:
            syn_i.append(v1)
            syn_j.append(v2)
            syn_c.append(c)

    if syn_i:
        syn_int.connect(i = syn_i, j = syn_j)
    else:
        syn_int.connect(False)
    syn_int.w = np.array(syn_c) * ps.w_int * mV

    ng_spike_mon = SpikeMonitor(ng)

    # May be a better way to do this
    n_reps = ceil(ps.n_ticks / EXT_BUF_ROWS)
    indices, times = ps.ext_spikes
    tot_indices, tot_times = [], []
    for b in range(n_reps):
        tot_indices.extend(indices)
        tot_times.extend([b * EXT_BUF_ROWS + t for t in times])
    tot_times = np.array(tot_times) * ps.dt * ms

    ext = SpikeGeneratorGroup(n_neurons, tot_indices, tot_times)
    syn_ext = Synapses(
        ext, ng,
        "w : volt",
        on_pre = "g += w"
    )
    syn_ext.connect(j = "i")
    syn_ext.w = ps.w_ext * mV


    net = Network([
        ng,
        ng_spike_mon,
        ext,
        syn_ext,
        syn_int
    ])
    start = time()
    net.run(ps.n_ticks * ps.dt * ms, namespace = dict(
        tau_g = ps.tau_g * ms,
        v_th = ps.v_th * mV,
        tau_m = ps.tau_m * ms
    ))
    secs = time() - start

    spikes_per_neuron = [len(ts) for ts in ng_spike_mon.spike_trains().values()]
    return SimResult(
        n_real_secs = secs,
        n_spikes = sum(spikes_per_neuron),
        spikes_per_neuron = spikes_per_neuron
    )
