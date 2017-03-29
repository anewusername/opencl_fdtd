# opencl_fdtd

**opencl_fdtd** is a python application for running 3D time-domain
electromagnetic simulations on parallel compute hardware (mainly GPUs).

**Performance** highly depends on what hardware you have available:
* A 395x345x73 cell simulation (~10 million points, 8-cell absorbing boundaries)
 runs at around 91 iterations/sec. on my AMD RX480.
* On an Nvidia GTX 580, it runs at 66 iterations/sec
* On my laptop (Nvidia 940M) the same simulation achieves ~12 iterations/sec.
* An L3 photonic crystal cavity ringdown simulation (1550nm source, 40nm
 discretization, 8000 steps) takes about 3 minutes on my laptop.

**Capabilities** are currently pretty minimal:
* Absorbing boundaries (CPML)
* Perfect electrical conductors (PECs; to use set epsilon to inf)
* Anisotropic media (eps_xx, eps_yy, eps_zz, mu_xx, ...)
* Direct access to fields (eg., you can trivially add a soft or hard
 current source with just sim.E[ind] += sin(f0 * t), or save any portion
 of a field to a file)


## Installation

**Requirements:**
* python 3 (written and tested with 3.5)
* numpy
* pyopencl
* jinja2
* dill (for file output)
* [gridlock](https://mpxd.net/gogs/jan/gridlock)
* [masque](https://mpxd.net/gogs/jan/masque)
* [fdfd_tools](https://mpxd.net/gogs/jan/fdfd_tools)

To get the code, just clone this repository:
```bash
git clone https://mpxd.net/gogs/jan/opencl_fdtd.git
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
