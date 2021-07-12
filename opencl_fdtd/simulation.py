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

from meanas import vec


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
        with open('sources.c', 'wt') as f:
            f.write(repr(sim.sources))

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
    E = None            # type: pyopencl.array.Array
    H = None            # type: pyopencl.array.Array
    S = None            # type: pyopencl.array.Array
    eps = None          # type: pyopencl.array.Array
    dt = None           # type: float
    inv_dxes = None     # type: List[pyopencl.array.Array]

    arg_type = None     # type: numpy.float32 or numpy.float64

    context = None      # type: pyopencl.Context
    queue = None        # type: pyopencl.CommandQueue

    update_E = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    update_H = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    update_S = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    update_J = None     # type: Callable[[List[pyopencl.Event]], pyopencl.Event]
    sources = None      # type: Dict[str, str]

    def __init__(self,
                 epsilon: List[numpy.ndarray],
                 pmls: List[Dict[str, int or float]],
                 bloch_boundaries: List[Dict[str, int or float]] = (),
                 dxes: List[List[numpy.ndarray]] or float = None,
                 dt: float = None,
                 initial_fields: Dict[str, List[numpy.ndarray]] = None,
                 context: pyopencl.Context = None,
                 queue: pyopencl.CommandQueue = None,
                 float_type: numpy.float32 or numpy.float64 = numpy.float32,
                 do_poynting: bool = True,
                 do_poynting_halves: bool = False,
                 do_fieldsrc: bool = False):
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
        :param bloch_boundaries: List of dicts with keys:
            'axis': One of 'x', 'y', 'z'.
            'real': Real part of bloch phase factor (i.e. real(exp(i * phase)))
            'imag': Imaginary part of bloch phase factor (i.e. imag(exp(i * phase)))
        :param dt: Time step. Default is min(dxes) * .99/sqrt(3).
        :param initial_fields: Dict with optional keys ('E', 'H', 'F', 'G') containing initial values for the
            specified fields (default is 0 everywhere). Fields have same format as epsilon.
        :param context: pyOpenCL context. If not given, pyopencl.create_some_context(False) is called.
        :param queue: pyOpenCL command queue. If not given, pyopencl.CommandQueue(context) is called.
        :param float_type: numpy.float32 or numpy.float64. Default numpy.float32.
        :param do_poynting: If true, enables calculation of the poynting vector, S.
                Poynting vector calculation adds the following computational burdens:
                    ****INACCURATE, TODO FIXME*****
                    * During update_H, ~6 extra additions/cell are performed in order to spatially
                        average E and temporally average H. These quantities are multiplied
                        (6 multiplications/cell) and then stored (6 writes/cell, cache-friendly).
                    * update_S performs a discrete cross product using the precalculated products
                        from update_H. This is not nice to the cache and similar to e.g. update_E
                        in complexity.
                    * GPU memory requirements are approximately doubled, since S and the intermediate
                        products must be stored.
        :param do_poynting_halves: TODO DOCUMENT
        """
        if initial_fields is None:
            initial_fields = {}

        self.shape = epsilon[0].shape
        self.arg_type = float_type
        self.sources = {}
        self._create_context(context, queue)
        self._create_eps(epsilon)

        if dxes is None:
            dxes = 1.0

        if isinstance(dxes, (float, int)):
            uniform_dx = dxes
            min_dx = dxes
        else:
            uniform_dx = False
            self.inv_dxes = [self._create_field(1 / dxn) for dxn in dxes[0] + dxes[1]]
            min_dx = min(min(dxn) for dxn in dxes[0] + dxes[1])

        max_dt = min_dx * .99 / numpy.sqrt(3)

        if dt is None:
            self.dt = max_dt
        elif dt > max_dt:
            warnings.warn('Warning: unstable dt: {}'.format(dt))
        elif dt <= 0:
            raise Exception('Invalid dt: {}'.format(dt))
        else:
            self.dt = dt

        self.E = self._create_field(initial_fields.get('E', None))
        self.H = self._create_field(initial_fields.get('H', None))
        if bloch_boundaries:
            self.F = self._create_field(initial_fields.get('F', None))
            self.G = self._create_field(initial_fields.get('G', None))

        for pml in pmls:
            pml.setdefault('thickness', 8)
            pml.setdefault('epsilon_eff', 1.0)
            pml.setdefault('mu_eff', 1.0)
            pml.setdefault('ln_R_per_layer', -1.6)
            pml.setdefault('m', 3.5)
            pml.setdefault('cfs_alpha', 0)
            pml.setdefault('ma', 1)

        ctype = type_to_C(self.arg_type)

        def ptr(arg: str) -> str:
            return ctype + ' *' + arg

        base_fields = OrderedDict()
        base_fields[ptr('E')] = self.E
        base_fields[ptr('H')] = self.H
        base_fields[ctype + ' dt'] = self.dt
        if uniform_dx is False:
            inv_dx_names = ['inv_d' + eh + r for eh in 'eh' for r in 'xyz']
            for name, field in zip(inv_dx_names, self.inv_dxes):
                base_fields[ptr(name)] = field

        eps_field = OrderedDict()
        eps_field[ptr('eps')] = self.eps

        if bloch_boundaries:
            base_fields[ptr('F')] = self.F
            base_fields[ptr('G')] = self.G

            bloch_fields = OrderedDict()
            bloch_fields[ptr('E')] = self.F
            bloch_fields[ptr('H')] = self.G
            bloch_fields[ctype + ' dt'] = self.dt
            bloch_fields[ptr('F')] = self.E
            bloch_fields[ptr('G')] = self.H

        common_source = jinja_env.get_template('common.cl').render(
            ftype=ctype,
            shape=self.shape,
            )
        jinja_args = {
            'common_header': common_source,
            'pmls': pmls,
            'do_poynting': do_poynting,
            'do_poynting_halves': do_poynting_halves,
            'bloch': bloch_boundaries,
            'uniform_dx': uniform_dx,
            }
        E_source = jinja_env.get_template('update_e.cl').render(**jinja_args)
        H_source = jinja_env.get_template('update_h.cl').render(**jinja_args)
        self.sources['E'] = E_source
        self.sources['H'] = H_source

        if bloch_boundaries:
            bloch_args = jinja_args.copy()
            bloch_args['do_poynting'] = False
            bloch_args['bloch'] = [{'axis': b['axis'],
                                    'real': b['imag'],
                                    'imag': b['real']}
                                   for b in bloch_boundaries]
            F_source = jinja_env.get_template('update_e.cl').render(**bloch_args)
            G_source = jinja_env.get_template('update_h.cl').render(**bloch_args)
            self.sources['F'] = F_source
            self.sources['G'] = G_source

        S_fields = OrderedDict()
        if do_poynting:
            self.S = pyopencl.array.zeros_like(self.E)
            S_fields[ptr('S')] = self.S
        if do_poynting_halves:
            self.S0 = pyopencl.array.zeros_like(self.E)
            self.S1 = pyopencl.array.zeros_like(self.E)
            S_fields[ptr('S0')] = self.S0
            S_fields[ptr('S1')] = self.S1

        J_fields = OrderedDict()
        if do_fieldsrc:
            J_source = jinja_env.get_template('update_j.cl').render(**jinja_args)
            self.sources['J'] = J_source

            self.Ji = pyopencl.array.zeros_like(self.E)
            self.Jr = pyopencl.array.zeros_like(self.E)
            J_fields[ptr('Jr')] = self.Jr
            J_fields[ptr('Ji')] = self.Ji

        '''
        PML
        '''
        pml_e_fields, pml_h_fields = self._create_pmls(pmls)
        if bloch_boundaries:
            pml_f_fields, pml_g_fields = self._create_pmls(pmls)

        '''
        Create operations
        '''
        self.update_E = self._create_operation(E_source, (base_fields, eps_field, pml_e_fields))
        self.update_H = self._create_operation(H_source, (base_fields, pml_h_fields, S_fields))
        if bloch_boundaries:
            self.update_F = self._create_operation(F_source, (bloch_fields, eps_field, pml_f_fields))
            self.update_G = self._create_operation(G_source, (bloch_fields, pml_g_fields))
        if do_fieldsrc:
            args = OrderedDict()
            [args.update(d) for d in (base_fields, J_fields)]
            var_args = [ctype + ' ' + v for v in 'cs'] + ['uint ' + r + m for r in 'xyz' for m in ('min', 'max')]
            update = ElementwiseKernel(self.context, operation=J_source,
                                       arguments=', '.join(list(args.keys()) + var_args))
            self.update_J = lambda e, *a: update(*args.values(), *a, wait_for=e)

    def _create_pmls(self, pmls):
        ctype = type_to_C(self.arg_type)

        def ptr(arg: str) -> str:
            return ctype + ' *' + arg

        pml_e_fields = OrderedDict()
        pml_h_fields = OrderedDict()
        for pml in pmls:
            a = 'xyz'.find(pml['axis'])

            sigma_max = -pml['ln_R_per_layer'] / 2 * (pml['m'] + 1)
            kappa_max = numpy.sqrt(pml['mu_eff'] * pml['epsilon_eff'])
            alpha_max = pml['cfs_alpha']

            def par(x):
                scaling = (x / pml['thickness']) ** pml['m']
                sigma = scaling * sigma_max
                kappa = 1 + scaling * (kappa_max - 1)
                alpha = ((1 - x / pml['thickness']) ** pml['ma']) * alpha_max
                p0 = numpy.exp(-(sigma / kappa + alpha) * self.dt)
                p1 = sigma / (sigma + kappa * alpha) * (p0 - 1)
                p2 = 1 / kappa
                return p0, p1, p2

            xe, xh = (numpy.arange(1, pml['thickness'] + 1, dtype=self.arg_type)[::-1] for _ in range(2))
            if pml['polarity'] == 'p':
                xe -= 0.5
            elif pml['polarity'] == 'n':
                xh -= 0.5

            pml_p_names = [['p' + pml['axis'] + i + eh + pml['polarity'] for i in '012'] for eh in 'eh']
            for name_e, name_h, pe, ph in zip(pml_p_names[0], pml_p_names[1], par(xe), par(xh)):
                pml_e_fields[ptr(name_e)] = pyopencl.array.to_device(self.queue, pe)
                pml_h_fields[ptr(name_h)] = pyopencl.array.to_device(self.queue, ph)

            uv = 'xyz'.replace(pml['axis'], '')
            psi_base = 'Psi_' + pml['axis'] + pml['polarity'] + '_'
            psi_names = [[psi_base + eh + c for c in uv] for eh in 'EH']

            psi_shape = list(self.shape)
            psi_shape[a] = pml['thickness']

            for ne, nh in zip(*psi_names):
                pml_e_fields[ptr(ne)] = pyopencl.array.zeros(self.queue, tuple(psi_shape), dtype=self.arg_type)
                pml_h_fields[ptr(nh)] = pyopencl.array.zeros(self.queue, tuple(psi_shape), dtype=self.arg_type)
        return pml_e_fields, pml_h_fields

    def _create_operation(self, source, args_fields):
        args = OrderedDict()
        [args.update(d) for d in args_fields]
        update = ElementwiseKernel(self.context, operation=source,
                                   arguments=', '.join(args.keys()))
        return lambda e: update(*args.values(), wait_for=e)

    def _create_context(self, context: pyopencl.Context = None,
                        queue: pyopencl.CommandQueue = None):
        if context is None:
            self.context = pyopencl.create_some_context()
        else:
            self.context = context

        if queue is None:
            self.queue = pyopencl.CommandQueue(self.context)
        else:
            self.queue = queue

    def _create_eps(self, epsilon: List[numpy.ndarray]):
        if len(epsilon) != 3:
            raise Exception('Epsilon must be a list with length of 3')
        if not all((e.shape == epsilon[0].shape for e in epsilon[1:])):
            raise Exception('All epsilon grids must have the same shape. Shapes are {}', [e.shape for e in epsilon])
        if not epsilon[0].shape == self.shape:
            raise Exception('Epsilon shape mismatch. Expected {}, got {}'.format(self.shape, epsilon[0].shape))
        self.eps = pyopencl.array.to_device(self.queue, vec(epsilon).astype(self.arg_type))

    def _create_field(self, initial_value: List[numpy.ndarray] = None):
        if initial_value is None:
            return pyopencl.array.zeros_like(self.eps)
        else:
            if len(initial_value) != 3:
                Exception('Initial field value must be a list of length 3')
            if not all((f.shape == self.shape for f in initial_value)):
                Exception('Initial field list elements must have same shape as epsilon elements')
            return pyopencl.array.to_device(self.queue, vec(initial_value).astype(self.arg_type))


def type_to_C(float_type: numpy.dtype) -> str:
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

#            def par(x):
#                scaling = ((x / (pml['thickness'])) ** pml['m'])
#                print('scaling', scaling)
#                print('sigma_max * dt / 2', sigma_max * self.dt / 2)
#                print('kappa_max', kappa_max)
#                print('m', pml['m'])
#                sigma = scaling * sigma_max
#                kappa = 1 + scaling * (kappa_max - 1)
#                alpha = ((1 - x / pml['thickness']) ** pml['ma']) * alpha_max
#                p0 = 1/(1 + self.dt * (alpha + sigma / kappa))
#                p1 = self.dt * sigma / kappa * p0
#                p2 = 1/kappa
#                print(p0.min(), p0.max(), p1.min(), p1.max())
#                return p0, p1, p2
#
#
