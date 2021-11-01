"""
Example code for running an OpenCL FDTD simulation

See main() for simulation setup.
"""

import sys
import time
import logging
import pyopencl

import numpy
import lzma
import dill

from opencl_fdtd import Simulation
from masque import Pattern, shapes
import gridlock
import pcgen
import meanas


__author__ = 'Jan Petykiewicz'

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def perturbed_l3(a: float, radius: float, **kwargs) -> Pattern:
    """
    Generate a masque.Pattern object containing a perturbed L3 cavity.

    :param a: Lattice constant.
    :param radius: Hole radius, in units of a (lattice constant).
    :param kwargs: Keyword arguments:
        hole_dose, trench_dose, hole_layer, trench_layer: Shape properties for Pattern.
                Defaults *_dose=1, hole_layer=0, trench_layer=1.
        shifts_a, shifts_r: passed to pcgen.l3_shift; specifies lattice constant (1 -
                multiplicative factor) and radius (multiplicative factor) for shifting
                holes adjacent to the defect (same row). Defaults are 0.15 shift for
                first hole, 0.075 shift for third hole, and no radius change.
        xy_size: [x, y] number of mirror periods in each direction; total size is
                2 * n + 1 holes in each direction. Default [10, 10].
        perturbed_radius: radius of holes perturbed to form an upwards-driected beam
                (multiplicative factor). Default 1.1.
        trench width: Width of the undercut trenches. Default 1.2e3.
    :return: masque.Pattern object containing the L3 design
    """

    default_args = {'hole_dose':    1,
                    'trench_dose':  1,
                    'hole_layer':   0,
                    'trench_layer': 1,
                    'shifts_a':     (0.15, 0, 0.075),
                    'shifts_r':     (1.0, 1.0, 1.0),
                    'xy_size':      (10, 10),
                    'perturbed_radius': 1.1,
                    'trench_width': 1.2e3,
                    }
    kwargs = {**default_args, **kwargs}

    xyr = pcgen.l3_shift_perturbed_defect(mirror_dims=kwargs['xy_size'],
                                          perturbed_radius=kwargs['perturbed_radius'],
                                          shifts_a=kwargs['shifts_a'],
                                          shifts_r=kwargs['shifts_r'])
    xyr *= a
    xyr[:, 2] *= radius

    pat = Pattern()
    pat.name = 'L3p-a{:g}r{:g}rp{:g}'.format(a, radius, kwargs['perturbed_radius'])
    pat.shapes += [shapes.Circle(radius=r, offset=(x, y),
                                 dose=kwargs['hole_dose'],
                                 layer=kwargs['hole_layer'])
                   for x, y, r in xyr]

    maxes = numpy.max(numpy.fabs(xyr), axis=0)
    pat.shapes += [shapes.Polygon.rectangle(
        lx=(2 * maxes[0]), ly=kwargs['trench_width'],
        offset=(0, s * (maxes[1] + a + kwargs['trench_width'] / 2)),
        dose=kwargs['trench_dose'], layer=kwargs['trench_layer'])
                   for s in (-1, 1)]
    return pat


def main():
    max_t = 4000            # number of timesteps

    dx = 25                 # discretization (nm/cell)
    pml_thickness = 8       # (number of cells)

    wl = 1550               # Excitation wavelength and fwhm
    dwl = 200

    # Device design parameters
    xy_size = numpy.array([10, 10])
    a = 430
    r = 0.285
    th = 170

    # refractive indices
    n_slab = 3.408  # InGaAsP(80, 50) @ 1550nm
    n_air = 1.0   # air

    # Half-dimensions of the simulation grid
#    xy_max = (xy_size + 1) * a * [1, numpy.sqrt(3)/2]
#    z_max = 1.6 * a
#    xyz_max = numpy.hstack((xy_max, z_max)) + pml_thickness * dx
#
#    # Coordinates of the edges of the cells.
#    #  The fdtd package can only do square grids at the moment.
#    half_edge_coords = [numpy.arange(dx/2, m + dx, step=dx) for m in xyz_max]
#    edge_coords = [numpy.hstack((-h[::-1], h)) for h in half_edge_coords]
    edge_coords = [numpy.arange(-100.5, 101), numpy.arange(-1, 1), numpy.arange(-100.5, 101)]
#    edge_coords = [numpy.arange(-100.5, 101), numpy.arange(-100.5, 101), numpy.arange(-1, 1)]

    # #### Create the grid, mask, and draw the device ####
    grid = gridlock.Grid(edge_coords)
    epsilon = grid.allocate(n_air**2)
#    grid.draw_slab(epsilon,
#                   surface_normal=2,
#                   center=[0, 0, 0],
#                   thickness=th,
#                   eps=n_slab**2)
#    mask = perturbed_l3(a, r)
#
#    grid.draw_polygons(epsilon,
#                       surface_normal=2,
#                       center=[0, 0, 0],
#                       thickness=2 * th,
#                       eps=n_air**2,
#                       polygons=mask.as_polygons())

    logger.info('grid shape: {}'.format(grid.shape))
    # #### Create the simulation grid ####
#    pmls = [{'axis': a, 'polarity': p, 'thickness': pml_thickness}
#            for a in 'xyz' for p in 'np']
    pmls = [{'axis': a, 'polarity': p, 'thickness': pml_thickness}
            for a in 'xz' for p in 'np']
    #bloch = [{'axis': a, 'real': 1, 'imag': 0} for a in 'x']
    bloch = []
    sim = Simulation(epsilon, do_poynting=True, pmls=pmls, bloch_boundaries=bloch)

    # Source parameters and function
    w = 2 * numpy.pi * dx / wl
    fwhm = dwl * w * w / (2 * numpy.pi * dx)
    alpha = (fwhm ** 2) / 8 * numpy.log(2)
    delay = 7/numpy.sqrt(2 * alpha)

    def field_source(i):
        t0 = i * sim.dt - delay
        return numpy.sin(w * t0) * numpy.exp(-alpha * t0**2)

    with open('sources.c', 'w') as f:
        f.write(sim.sources['E'])
        f.write('\n====================H======================\n')
        f.write(sim.sources['H'])
        if sim.update_S:
            f.write('\n=====================S=====================\n')
            f.write(sim.sources['S'])
        if bloch:
            f.write('\n=====================F=====================\n')
            f.write(sim.sources['F'])
            f.write('\n=====================G=====================\n')
            f.write(sim.sources['G'])


    planes = numpy.empty((max_t, 4))
    planes2 = numpy.empty((max_t, 4))
    Ectr = numpy.empty(max_t)
    u = numpy.empty(max_t)
    ui = numpy.empty(max_t)
    # #### Run a bunch of iterations ####
    # event = sim.whatever([prev_event]) indicates that sim.whatever should be queued
    #  immediately and run once prev_event is finished.
    start = time.perf_counter()
    for t in range(max_t):
        e = sim.update_E([])
#        if bloch:
#            e = sim.update_F([e])
        e.wait()

        ind = numpy.ravel_multi_index(tuple(grid.shape//2), dims=grid.shape, order='C') + numpy.prod(grid.shape)
#        sim.E[ind] += field_source(t)
        if t == 2:
            sim.E[ind] += 1e6

        h_old = sim.H.copy()

        e = sim.update_H([])
#        if bloch:
#            e = sim.update_G([e])
        e.wait()

        S = sim.S.get().reshape(epsilon.shape) * sim.dt * dx * dx *dx
        m = 30
        planes[t] = (
            S[0][+pml_thickness+2, :, pml_thickness+3:-pml_thickness-3].sum(),
            S[0][-pml_thickness-2, :, pml_thickness+3:-pml_thickness-3].sum(),
            S[2][pml_thickness+2:-pml_thickness-2, :, +pml_thickness+2].sum(),
            S[2][pml_thickness+2:-pml_thickness-2, :, -pml_thickness-2].sum(),
            )
        planes2[t] = (
            S[0][grid.shape[0]//2-1, 0, grid.shape[2]//2].sum(),
            S[0][grid.shape[0]//2  , 0, grid.shape[2]//2].sum(),
            S[2][grid.shape[0]//2  , 0, grid.shape[2]//2-1].sum(),
            S[2][grid.shape[0]//2  , 0, grid.shape[2]//2].sum(),
            )

#        planes[t] = (
#            S[0][+pml_thickness+m, pml_thickness+m+1:-pml_thickness-m, :].sum(),
#            S[0][-pml_thickness-m, pml_thickness+m+1:-pml_thickness-m, :].sum(),
#            S[1][pml_thickness+1+m:-pml_thickness-m, +pml_thickness+m, :].sum(),
#            S[1][pml_thickness+1+m:-pml_thickness-m, -pml_thickness-m, :].sum(),
#            )
#        planes2[t] = (
#            S[0][grid.shape[0]//2-1, grid.shape[1]//2  , 0].sum(),
#            S[0][grid.shape[0]//2  , grid.shape[1]//2  , 0].sum(),
#            S[1][grid.shape[0]//2  , grid.shape[1]//2-1, 0].sum(),
#            S[1][grid.shape[0]//2  , grid.shape[1]//2  , 0].sum(),
#            )
        Ectr[t] = sim.E[ind].get()
        u[t] = pyopencl.array.sum(sim.E * sim.E * sim.eps + h_old * sim.H).get() * dx * dx * dx
        ui[t] = (sim.E * sim.E * sim.eps + h_old * sim.H).reshape(epsilon.shape).get()[:, pml_thickness+m:-pml_thickness-m, :,
                                                                                            pml_thickness+m:-pml_thickness-m].sum() * dx * dx * dx
#        ui[t] = (sim.E * sim.E * sim.eps + h_old * sim.H).reshape(epsilon.shape).get()[:, pml_thickness+m:-pml_thickness-m,
#                                                                                            pml_thickness+m:-pml_thickness-m, :].sum() * dx * dx * dx

        if t % 100 == 0:
            logger.info('iteration {}: average {} iterations per sec'.format(t, (t+1)/(time.perf_counter()-start)))
            sys.stdout.flush()

    with lzma.open('saved_simulation', 'wb') as f:
        def unvec(f):
            return meanas.fdmath.unvec(f, grid.shape)
        d = {
            'grid': grid,
            'epsilon': epsilon,
            'E': unvec(sim.E.get()),
            'H': unvec(sim.H.get()),
            'dt': sim.dt,
            'dx': dx,
            'planes': planes,
            'planes2': planes2,
            'Ectr': Ectr,
            'u': u,
            'ui': ui,
            }
        if sim.S is not None:
            d['S'] = unvec(sim.S.get())
        dill.dump(d, f)


if __name__ == '__main__':
    main()
