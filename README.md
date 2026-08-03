# SNN Simulation of FlyWire on the Agilex 7 FPGA

This repository contains a high-performance SNN simulator of FlyWire
designed for the Agilex 7 FPGA. It serves as the online compendium for
the article "High-Performance SNN Simulation of FlyWire on the Agilex
7 FPGA" (unpublished).

The simulator depends on a number of easily installable Python
libraries. It also depends on the non-standard OpenCL wrapper
[ml-stuff](https://github.com/bjourne/ml-stuff), which must be
installed from source. Intel FPGA SDK for OpenCL 21.2 and Quartus 21.2
is required for hardware synthesis.

If you find this work useful, then:

* Please star this repository!
* Consider citing us:
```
bibtex here
```

## Usage

Running the simulator requires the FlyWire connectome which can be
downloaded from [this
repository](https://github.com/philshiu/Drosophila_brain_model/blob/main/Connectivity_783.parquet).

To run the simulator on OpenCL:

```
PYTHONPATH=. python flysim/scripts.py simulate \
    --connectome=/path/to/Connectivity_783.parquet \
    --n-ticks=100 --fp-bits=32 --n-loops=10 \
	--spike-frac=0.00205000 --sim-seed=2000 \
    opencl \
    --ocl-path=kernel.cl \
    --neu-align=16 --syn-align=16 --syn-grp-align=2 --n-lanes=4 \
    --platform-index=0
```

`--platform-index` selects the OpenCL platform to use.

To run the simulator on NumPy or Brian 2:

```
PYTHONPATH=. python flysim/scripts.py simulate \
    --connectome=/path/to/Connectivity_783.parquet \
    --n-ticks=100 --fp-bits=32 --n-loops=1 --spike-frac=0.00205000 --sim-seed=2000 \
    numpy/brian2
```

To synthesize the kernel:

```
PYTHONPATH=. python flysim/scripts.py synthesize \
    --connectome=/path/to/Connectivity_783.parquet  \
    --seed=9988 --fp-bits=32 --output-path /tmp \
    --ocl-path=kernel.cl \
    --neu-align=64 --syn-align=16 --syn-grp-align=1 --n-lanes=4 \
        --board-package=/path/to/board_package --board=BOARD_ID --only-rtl
```

The optional switches `--board` and `--board-package` specifies the
FPGA model.

To print experimental data:

```
PYTHONPATH=. python flysim/scripts.py print-tables
```
