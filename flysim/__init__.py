# Copyright (C) 2026 Björn A. Lindqvist <bjourne@gmail.com>
from dataclasses import dataclass
from math import exp
from functools import wraps
from platformdirs import PlatformDirs

import pickle

# Max number of time steps to generate random data for.
EXT_BUF_ROWS = 1024

@dataclass
class SimResult:
    n_real_secs: float
    n_spikes: int
    spikes_per_neuron: list = None

def compute_abg(ps):
    alpha = exp(-ps.dt/ps.tau_m)
    beta = exp(-ps.dt/ps.tau_g)
    gamma = ps.tau_g * (beta - alpha) / (ps.tau_g - ps.tau_m)
    return alpha, beta, gamma

def cache_to_file(kfun):
    def decor(fun):
        @wraps(fun)
        def wrapper(*args, **kwargs):
            name = [str(x) for x in [fun.__name__]
                    + kfun(*args, **kwargs)]
            dirs = PlatformDirs("flysim", ensure_exists = True)
            path = dirs.user_cache_path / ("-".join(name) + ".pickle")
            if path.exists():
                print("Loading %s" % path)
                with path.open("rb") as f:
                    return pickle.load(f)
            ret = fun(*args, **kwargs)
            print("Writing %s" % path)
            with path.open("wb") as f:
                pickle.dump(ret, f)
            return ret
        return wrapper
    return decor
