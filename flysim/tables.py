# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from flysim.data import (
    AGILEX7_SIMS,
    AGILEX7_SYNTH,
    BMARK_SPECS,
    INDEX_GROWTH, SIM_SPECS, SPEEDS
)
from math import isnan
from pathlib import Path

import pandas as pd

def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

def read_power_data(path):
    names = [
        "Time",
        "FPGA Temp.",
        "Board Temp.",
        "Board2 Temp.",
        "XCVR Temp.",
        "Fan RPM",
        "12V V",
        "12V A",
        "Core V",
        "Core A",
        "col1",
        "col2",
        "col3",
        "col4"
    ]
    paths = path.glob("power-*.csv")
    paths = [(p.stem.split("-")[1], p) for p in paths]
    df = pd.concat([
        pd.read_csv(p, names = names, header = 0).assign(freq = f)
        for (f, p) in paths
        if is_num(f)
    ])

    df["pwr_board"] = df["Core V"] * df["Core A"]
    df["pwr_fpga"] = df["col1"] * df["col2"] + df["col3"] * df["col4"]
    df["time"] = df["Time"]
    df["freq"] = df["freq"].astype(float)

    # Shift measurements so that they align on the time axis
    cutoffs = df[df.groupby("freq")["pwr_board"].diff() > 0.5] \
        .groupby("freq", as_index=False).first()[["freq", "time"]]

    df = df.merge(cutoffs, on="freq")
    df["time"] = df["time_x"] - df["time_y"]
    df = df[(df["time"] > -2.5) & (df["time"] < 50)]
    df = df[["freq", "time", "pwr_board", "pwr_fpga"]]
    df = pd.melt(
        df,
        id_vars = ["freq", "time"],
        value_vars = ["pwr_board", "pwr_fpga"],
        var_name = "pwr_type",
        value_name = "pwr"
    )
    return df

def pretty_print_df(df, for_typst, n_dec):
    fmt = "%%.%df" % n_dec
    def val2str(v):
        if isinstance(v, float):
            v = "-" if isnan(v) else fmt % v
        if for_typst:
            v = f"[{v}],"
        return v

    df = df.map(val2str)
    print(df.to_string(index = False))
    print()

def pivoted_benchmarks():
    # Take mins for each measurement
    df = SPEEDS.groupby(["freq", "sim"])["rtf"].min()
    df = df.reset_index()
    df = df.pivot(index = "sim", columns = "freq", values = "rtf")
    df = df.reset_index()
    return df.merge(SIM_SPECS, on = "sim")

def munge_power(df):
    # Integrate
    key = ["freq", "pwr_type"]

    cutoffs = df[df.groupby(key)["pwr"].diff() < -0.5] \
        .groupby(key, as_index=False).last()[["freq", "pwr_type", "time"]]

    df = df.merge(cutoffs, on = ["freq", "pwr_type"])
    df = df[(df["time_x"] > 0) & (df["time_x"] < df["time_y"])]
    df = df.groupby(["freq", "pwr_type"]).agg(
        time_max = ("time_x", "max"),
        time_min = ("time_x", "min"),
        pwr_mean = ("pwr", "mean")
    ).reset_index()
    df["energy"] = round((df["time_max"] - df["time_min"]) * df["pwr_mean"])
    return df


def print_tables(for_typst):
    sims = [
        "stacs", "loihi2-1.0ms", "loihi2-0.1ms", "brian2-wang",
        "brian2-ours", "numpy", "agx7g"
    ]
    freqs = [0.00, 0.50, 1.00, 2.00, 3.50, 5.00, 10, 20, 40, 60]

    piv_df = pivoted_benchmarks()

    df = piv_df.set_index("sim").reindex(sims).reset_index()
    df = df[["name"] + freqs]
    pretty_print_df(df, for_typst, 2)

    df = piv_df.merge(AGILEX7_SIMS, on = "sim")
    df = df[["t_fmax", "fp", "syn_para", "grp_para", "neu_para"] + freqs]
    pretty_print_df(df, for_typst, 2)

    df = INDEX_GROWTH
    pretty_print_df(df, for_typst, 2)

    df = AGILEX7_SYNTH
    df["lut"] = df["lut"] / 487200
    df["dsp"] = df["dsp"] / 4510
    df["ram"] = df["ram"] / 7110
    pretty_print_df(df, for_typst, 2)

    df = BMARK_SPECS
    df["frac"] *= 10_000
    pretty_print_df(df, for_typst, 2)

    df = read_power_data(Path("powerdata/"))
    pretty_print_df(munge_power(df), for_typst, 2)
