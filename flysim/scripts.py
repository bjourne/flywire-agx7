# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from collections import namedtuple
from flysim import (
    EXT_BUF_ROWS,
    cache_to_file,
    sim_brian2,
    sim_numpy,
    sim_opencl
)
from pathlib import Path
from random import choice, randrange, sample, seed as rseed, uniform

import click
import numpy as np
import pandas as pd

SimParams = namedtuple("SimParams", [
    # General
    "n_ticks", "n_neurons",

    # Neural dynamics
    "v_th", "tau_m", "tau_g",
    "t_ref", "t_dly", "dt",

    # Internal and external spike weights
    "w_int", "w_ext",

    # Initial state
    "v_init", "g_init",

    # Edges and external spikes
    "edges", "ext_spikes",

    # Floating point precision
    "fp_bits"
])

@cache_to_file(lambda p: [p.stem,])
def read_edges(path):

    df = pd.read_parquet(path)

    scol = "Presynaptic_Index"
    dcol = "Postsynaptic_Index"
    ccol = "Excitatory x Connectivity"

    df = df[[scol, dcol, ccol]]
    df = df.sort_values(by=[scol, dcol])

    # We assume the connectome references all neurons.
    n_neurons = max(df[scol].max(), df[dcol].max()) + 1

    edges = [[] for i in range(n_neurons)]
    for s, d, c in zip(df[scol], df[dcol], df[ccol]):
        edges[s].append((d, c))
    return edges

def print_result(ps, n_real_secs, n_spikes, sim_seed):
    args = n_spikes, n_real_secs, sim_seed
    fmt = "%8d spikes in %12.5f seconds (seed: %5d)"
    print(fmt % args)

    n_ticks = ps.n_ticks
    n_neurons = ps.n_neurons
    n_sim_secs = n_ticks / 10_000

    spike_freq = n_spikes / (n_sim_secs * n_neurons)
    rtf = n_real_secs / n_sim_secs
    print("RTF: %12.5f, Hz: %12.5f" % (rtf, spike_freq))

def get_sim_params(obj, sim_seed):
    rseed(sim_seed)
    np.random.seed(sim_seed)

    edges = read_edges(obj["connectome"])

    # Maybe prune some edges
    max_n_neurons = obj["max_n_neurons"]
    if max_n_neurons is not None:
        edges = [[(d, c) for (d, c) in es if d < max_n_neurons]
                 for es in edges[:max_n_neurons]]
    n_neurons = len(edges)

    # Dynamics
    v_th = 7.0
    tau_m = 20
    tau_g = 5
    t_ref = 22
    t_dly = 18
    dt = 0.1

    # Weights of spikes
    w_int = 0.275
    w_ext = 250 * w_int

    # Initial state
    is_sugar = obj["sugar_exp"]
    spike_frac = obj["spike_frac"]
    v_init = [0] * n_neurons

    if spike_frac is None and not is_sugar:
        g_init = [randrange(50) for _ in range(n_neurons)]
    else:
        g_init = [0] * n_neurons

    # Number of ticks
    n_ticks = obj["n_ticks"]

    ext_spikes = {}
    if is_sugar:
        ext_spikes = sugar_stimuli(n_ticks, obj["connectome"])
    elif spike_frac:
        n_range = min(EXT_BUF_ROWS, n_ticks) * n_neurons
        k = round(n_range * spike_frac)
        print("%d external spikes." % k)
        data = sorted(sample(list(range(n_range)), k = k))

        indices = [ix % n_neurons for ix in data]
        times = [ix // n_neurons for ix in data]
    else:
        indices, times = [], []
    return SimParams(
        n_ticks, n_neurons,
        v_th, tau_m, tau_g,
        t_ref, t_dly, dt,
        w_int, w_ext,
        v_init, g_init,
        edges, (indices, times), obj["fp_bits"]
    )

def run_sim(obj, sim_fun, sim_name):
    spike_frac = obj["spike_frac"]
    for i in range(obj["n_loops"]):
        sim_seed = obj["sim_seed"] + i
        ps = get_sim_params(obj, sim_seed)
        res = sim_fun(ps)
        print_result(ps, res.n_real_secs, res.n_spikes, sim_seed)
        df = pd.DataFrame(dict(
            neuron_id = list(range(ps.n_neurons)),
            n_spikes = res.spikes_per_neuron,
            n_ticks = ps.n_ticks,
            sim_seed = sim_seed,
            spike_frac = spike_frac
        ))
        path = Path(f"{sim_name}.csv")
        df.to_csv(
            path,
            mode="a",
            header = not path.exists(),
            index = False
        )

########################################################################
# Click commands below this point
########################################################################

@click.group(
    invoke_without_command = True,
    no_args_is_help=True
)
@click.version_option("0.0.1")
def cli():
    pass

@cli.group(help = "Simulate FlyWire with Brian2, NumPy, or OpenCL")
@click.pass_context
@click.option(
    "--connectome",
    type = click.Path(exists = True, path_type = Path),
    required = True
)
@click.option(
    "--sugar-exp", is_flag = True,
    help = "Whether to run the sugar experiment"
)
@click.option(
    "--n-ticks",
    type = int,
    required = True
)
@click.option(
    "--max-n-neurons",
    type = int,
    default = None,
    help = "Maximum number of neurons"
)
@click.option(
    "--fp-bits",
    type = int,
    help = "Number of FP bits",
    required = True
)
@click.option(
    "--spike-frac",
    type = float,
    help = "External spike prob",
    required = False
)
@click.option(
    "--sim-seed",
    type = int,
    help = "Simulation seed",
    default = 1234
)
@click.option(
    "--n-loops",
    type = int,
    required = True
)
def simulate(
        ctx,
        connectome, sugar_exp, n_ticks,
        max_n_neurons, fp_bits, spike_frac,
        sim_seed, n_loops
):
    ctx.obj = {}
    ctx.obj["connectome"] = connectome
    ctx.obj["fp_bits"] = fp_bits
    ctx.obj["max_n_neurons"] = max_n_neurons
    ctx.obj["n_ticks"] = n_ticks
    ctx.obj["sim_seed"] = sim_seed
    ctx.obj["spike_frac"] = spike_frac
    ctx.obj["sugar_exp"] = sugar_exp
    ctx.obj["n_loops"] = n_loops


@simulate.command(
    help = "Simulate the FlyWire connectome with NumPy"
)
@click.pass_context
def numpy(ctx):
    run_sim(ctx.obj, sim_numpy.sim, "numpy")

@simulate.command(
    help = "Simulate the FlyWire connectome with OpenCL"
)
@click.pass_context
@click.option(
    "--platform-index",
    help = "Index of platform to use",
    required = True,
    type = int
)
@click.option(
    "--ocl-path",
    type = click.Path(exists = True, path_type = Path),
    required = True
)
@click.option(
    "--neu-align",
    type = int,
    help = "Neuron alignment",
    required = True
)
@click.option(
    "--syn-align",
    type = int,
    help = "Synapse alignment",
    required = True
)
@click.option(
    "--syn-grp-align",
    help = "Alignment of synapse groups",
    type = int,
    required = True
)
@click.option(
    "--n-lanes",
    help = "Number of lanes privatization lanes",
    required = True
)
def opencl(
        ctx,
        platform_index, ocl_path,
        neu_align, syn_align, syn_grp_align, n_lanes
):
    def sim_fun(ps):
        return sim_opencl.sim(
            platform_index, ocl_path,
            neu_align, syn_align, syn_grp_align, n_lanes,
            ps
        )
    run_sim(ctx.obj, sim_fun, "opencl")

@simulate.command(
    help = "Simulate the FlyWire connectome with Brian2"
)
@click.pass_context
def brian2(ctx):
    run_sim(ctx.obj, sim_brian2.sim, "brian2")

def main():
    cli()

main()
