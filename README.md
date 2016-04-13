# opencl-fdtd

**opencl-fdtd** is a python application for running 3D time-domain
electromagnetic simulations on parallel compute hardware (mainly GPUs).

**Performance** highly depends on what hardware you have available:
* A 395x345x73 cell simulation (~10 million points, 8-cell absorbing boundaries)
 runs at around 42 iterations/sec. on my Nvidia GTX 580.
* On my laptop (Nvidia 940M) the same simulation achieves ~8 iterations/sec.
* An L3 photonic crystal cavity ringdown simulation (1550nm source, 40nm
 discretization, 8000 steps) takes about 5 minutes on my laptop.

**Capabilities** are currently pretty minimal:
* Absorbing boundaries (CPML)
* Conducting boundaries (PMC)
* Anisotropic media (eps_xx, eps_yy, eps_zz)
* Direct access to fields (eg., you can trivially add a soft or hard
 current source with just sim.E[1] += sin(f0 * t), or save any portion
 of a field to a file)

## Installation

**Requirements:**
* python 3 (written and tested with 3.5)
* numpy
* pyopencl
* h5py (for file output)
* [gridlock](https://mpxd.net/gogs/jan/gridlock)
* [masque](https://mpxd.net/gogs/jan/masque)

To get the code, just clone this repository:
```bash
git clone https://mpxd.net/gogs/jan/opencl-fdtd.git
```

You can install the requirements and their dependencies easily with
```bash
pip install -r requirements.txt
```

## Running
The root directory contains ``fdtd.py``, which sets up and runs a sample simulation
 (cavity ringdown).

```bash
python3 fdtd.py
```
