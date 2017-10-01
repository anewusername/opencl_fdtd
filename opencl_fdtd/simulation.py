"""
Class for constructing and holding the basic FDTD operations and fields
"""

from typing import List, Dict, Callable
from collections import OrderedDict
import numpy
import jinja2
import warnings

import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

from fdfd_tools import vec


__author__ = 'Jan Petykiewicz'


# Create jinja2 env on module load
jinja_env = jinja2.Environment(loader=jinja2.PackageLoader(__name__, 'kernels'))


class Simulation(object):
    """
    Constructs and holds the basic FDTD operations and related fields

    After constructing this object, call the (update_E, update_H, update_S) members
     to perform FDTD updates on the stored (E, H, S) fields:

        pmls = [{'axis': a, 'polarity': p} for a in 'xyz' for p in 'np']
        sim = Simulation(grid.grids, do_poynting=True, pmls=pmls)
        with open('sources.c', 'w') as f:
            f.write('{}'.format(sim.sources))

        for t in range(max_t):
            sim.update_E([]).wait()

            # Find the linear index for the center point, for Ey
            ind = numpy.ravel_multi_index(tuple(grid.shape//2), dims=grid.shape, order='C') + \
                    numpy.prod(grid.shape) * 1
            # Perturb the field (i.e., add a soft current source)
            sim.E[ind] += numpy.sin(omega * t * sim.dt)
            event = sim.update_H([])
            if sim.update_S:
                event = sim.update_S([event])
            event.wait()

            with lzma.open('saved_simulation', 'wb') as f:
                dill.dump(fdfd_tools.unvec(sim.E.get(), grid.shape), f)

    Code in the form
        event2 = sim.update_H([event0, event1])
     indicates that the update_H operation should be prepared immediately, but wait for
     event0 and event1 to occur (i.e. previous operations to finish) before starting execution.
     event2 can then be used to prepare further operations to be run after update_H.
    """
    E = None    # type: List[pyopencl.array.Array]
    H = None    # type: List[pyopencl.array.Array]
    S = None    # type: List[pyopencl.array.Array]
    eps = None  # type: List[pyopencl.array.Array]
    dt = None   # type: float

    arg_type = None     # type: numpy.float32 or numpy.float64

    context = None      # type: pyopencl.Context
    queue = None        # type: pyopencl.CommandQueue

    update_E = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    update_H = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    update_S = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    sources = None      # type: Dict[str, str]

    def __init__(self,
                 epsilon: List[numpy.ndarray],
                 pmls: List[Dict[str, int or float]],
                 dt: float = .99/numpy.sqrt(3),
                 initial_E: List[numpy.ndarray] = None,
                 initial_H: List[numpy.ndarray] = None,
                 context: pyopencl.Context = None,
                 queue: pyopencl.CommandQueue = None,
                 float_type: numpy.float32 or numpy.float64 = numpy.float32,
                 do_poynting: bool = True):
        """
        Initialize the simulation.

        :param epsilon: List containing [eps_r,xx, eps_r,yy, eps_r,zz], where each element is a Yee-shifted ndarray
                spanning the simulation domain. Relative epsilon is used.
        :param pmls: List of dicts with keys:
            'axis': One of 'x', 'y', 'z'.
            'direction': One of 'n', 'p'.
            'thickness': Number of layers, default 8.
            'epsilon_eff': Effective epsilon to match to. Default 1.0.
            'mu_eff': Effective mu to match to. Default 1.0.
            'ln_R_per_layer': Desired (ln(R) / thickness) value. Default -1.6.
            'm': Polynomial grading exponent. Default 3.5.
            'ma': Exponent for alpha. Default 1.
        :param dt: Time step. Default is .99/sqrt(3).
        :param initial_E: Initial E-field (default is 0 everywhere). Same format as epsilon.
        :param initial_H: Initial H-field (default is 0 everywhere). Same format as epsilon.
        :param context: pyOpenCL context. If not given, pyopencl.create_some_context(False) is called.
        :param queue: pyOpenCL command queue. If not given, pyopencl.CommandQueue(context) is called.
        :param float_type: numpy.float32 or numpy.float64. Default numpy.float32.
        :param do_poynting: If true, enables calculation of the poynting vector, S.
                Poynting vector calculation adds the following computational burdens:
                    * During update_H, ~6 extra additions/cell are performed in order to spatially
                        average E and temporally average H. These quantities are multiplied
                        (6 multiplications/cell) and then stored (6 writes/cell, cache-friendly).
                    * update_S performs a discrete cross product using the precalculated products
                        from update_H. This is not nice to the cache and similar to e.g. update_E
                        in complexity.
                    * GPU memory requirements are approximately doubled, since S and the intermediate
                        products must be stored.
        """

        if len(epsilon) != 3:
            Exception('Epsilon must be a list with length of 3')
        if not all((e.shape == epsilon[0].shape for e in epsilon[1:])):
            Exception('All epsilon grids must have the same shape. Shapes are {}', [e.shape for e in epsilon])

        if context is None:
            self.context = pyopencl.create_some_context()
        else:
            self.context = context

        if queue is None:
            self.queue = pyopencl.CommandQueue(self.context)
        else:
            self.queue = queue

        if dt > .99/numpy.sqrt(3):
            warnings.warn('Warning: unstable dt: {}'.format(dt))
        elif dt <= 0:
            raise Exception('Invalid dt: {}'.format(dt))
        else:
            self.dt = dt

        self.arg_type = float_type
        self.sources = {}
        self.eps = pyopencl.array.to_device(self.queue, vec(epsilon).astype(float_type))

        if initial_E is None:
            self.E = pyopencl.array.zeros_like(self.eps)
        else:
            if len(initial_E) != 3:
                Exception('Initial_E must be a list of length 3')
            if not all((E.shape == epsilon[0].shape for E in initial_E)):
                Exception('Initial_E list elements must have same shape as epsilon elements')
            self.E = pyopencl.array.to_device(self.queue, vec(E).astype(float_type))

        if initial_H is None:
            self.H = pyopencl.array.zeros_like(self.eps)
        else:
            if len(initial_H) != 3:
                Exception('Initial_H must be a list of length 3')
            if not all((H.shape == epsilon[0].shape for H in initial_H)):
                Exception('Initial_H list elements must have same shape as epsilon elements')
            self.H = pyopencl.array.to_device(self.queue, vec(H).astype(float_type))

        for pml in pmls:
            pml.setdefault('thickness', 8)
            pml.setdefault('epsilon_eff', 1.0)
            pml.setdefault('mu_eff', 1.0)
            pml.setdefault('ln_R_per_layer', -1.6)
            pml.setdefault('m', 3.5)
            pml.setdefault('ma', 1)

        ctype = type_to_C(self.arg_type)

        def ptr(arg: str) -> str:
            return ctype + ' *' + arg

        base_fields = OrderedDict()
        base_fields[ptr('E')] = self.E
        base_fields[ptr('H')] = self.H
        base_fields[ctype + ' dt'] = self.dt

        eps_field = OrderedDict()
        eps_field[ptr('eps')] = self.eps

        common_source = jinja_env.get_template('common.cl').render(
                ftype=ctype,
                shape=epsilon[0].shape,
                )
        jinja_args = {
                'common_header': common_source,
                'pmls': pmls,
                'do_poynting': do_poynting,
                }
        E_source = jinja_env.get_template('update_e.cl').render(**jinja_args)
        H_source = jinja_env.get_template('update_h.cl').render(**jinja_args)

        self.sources['E'] = E_source
        self.sources['H'] = H_source

        if do_poynting:
            S_source = jinja_env.get_template('update_s.cl').render(**jinja_args)
            self.sources['S'] = S_source

            self.oS = pyopencl.array.zeros(self.queue, self.E.shape + (2,), dtype=float_type)
            self.S = pyopencl.array.zeros_like(self.E)
            S_fields = OrderedDict()
            S_fields[ptr('oS')] = self.oS
            S_fields[ptr('S')] = self.S
        else:
            S_fields = OrderedDict()

        '''
        PML
        '''
        pml_e_fields = OrderedDict()
        pml_h_fields = OrderedDict()
        for pml in pmls:
            a = 'xyz'.find(pml['axis'])

            sigma_max = -pml['ln_R_per_layer'] / 2 * (pml['m'] + 1) / \
                    numpy.sqrt(pml['epsilon_eff'] * pml['mu_eff'])
            alpha_max = 0           # TODO: Nonzero alpha

            def par(x):
                sigma = ((x / pml['thickness']) ** pml['m']) * sigma_max
                alpha = ((1 - x / pml['thickness']) ** pml['ma']) * alpha_max
                p0 = numpy.exp(-(sigma + alpha) * dt)
                p1 = sigma / (sigma + alpha) * (p0 - 1)
                return p0, p1

            xe, xh = (numpy.arange(1, pml['thickness'] + 1, dtype=float_type)[::-1] for _ in range(2))
            if pml['polarity'] == 'p':
                xe -= 0.5
            elif pml['polarity'] == 'n':
                xh -= 0.5

            pml_p_names = [['p' + pml['axis'] + i + eh + pml['polarity'] for i in '01'] for eh in 'eh']
            for name_e, name_h, pe, ph in zip(pml_p_names[0], pml_p_names[1], par(xe), par(xh)):
                pml_e_fields[ptr(name_e)] = pyopencl.array.to_device(self.queue, pe)
                pml_h_fields[ptr(name_h)] = pyopencl.array.to_device(self.queue, ph)

            uv = 'xyz'.replace(pml['axis'], '')
            psi_base = 'Psi_' + pml['axis'] + pml['polarity'] + '_'
            psi_names = [[psi_base + eh + c for c in uv] for eh in 'EH']

            psi_shape = list(epsilon[0].shape)
            psi_shape[a] = pml['thickness']

            for ne, nh in zip(*psi_names):
                pml_e_fields[ptr(ne)] = pyopencl.array.zeros(self.queue, tuple(psi_shape), dtype=self.arg_type)
                pml_h_fields[ptr(nh)] = pyopencl.array.zeros(self.queue, tuple(psi_shape), dtype=self.arg_type)

        self.pml_e_fields = pml_e_fields
        self.pml_h_fields = pml_h_fields


        '''
        Create operations
        '''
        E_args = OrderedDict()
        [E_args.update(d) for d in (base_fields, eps_field, pml_e_fields)]
        E_update = ElementwiseKernel(self.context, operation=E_source,
                                     arguments=', '.join(E_args.keys()))

        H_args = OrderedDict()
        [H_args.update(d) for d in (base_fields, pml_h_fields, S_fields)]
        H_update = ElementwiseKernel(self.context, operation=H_source,
                                     arguments=', '.join(H_args.keys()))
        self.update_E = lambda e: E_update(*E_args.values(), wait_for=e)
        self.update_H = lambda e: H_update(*H_args.values(), wait_for=e)

        if do_poynting:
            S_args = OrderedDict()
            [S_args.update(d) for d in (base_fields, S_fields)]
            S_update = ElementwiseKernel(self.context, operation=S_source,
                                         arguments=', '.join(S_args.keys()))

            self.update_S = lambda e: S_update(*S_args.values(), wait_for=e)


def type_to_C(float_type) -> str:
    """
    Returns a string corresponding to the C equivalent of a numpy type.
    Only works for float16, float32, float64.

    :param float_type: e.g. numpy.float32
    :return: string containing the corresponding C type (eg. 'double')
    """
    if float_type == numpy.float16:
        arg_type = 'half'
    elif float_type == numpy.float32:
        arg_type = 'float'
    elif float_type == numpy.float64:
        arg_type = 'double'
    else:
        raise Exception('Unsupported type')
    return arg_type
