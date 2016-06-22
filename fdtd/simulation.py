"""
Class for constructing and holding the basic FDTD operations and fields
"""

from typing import List, Dict, Callable
import numpy
import warnings

import pyopencl
import pyopencl.array
from pyopencl.elementwise import ElementwiseKernel

from . import boundary, base
from .base import type_to_C


class Simulation(object):
    """
    Constructs and holds the basic FDTD operations and related fields
    """
    E = None    # type: List[pyopencl.array.Array]
    H = None    # type: List[pyopencl.array.Array]
    eps = None  # type: List[pyopencl.array.Array]
    dt = None   # type: float

    arg_type = None     # type: numpy.float32 or numpy.float64

    context = None      # type: pyopencl.Context
    queue = None        # type: pyopencl.CommandQueue

    update_E = None     # type: Callable[[],pyopencl.Event]
    update_H = None     # type: Callable[[],pyopencl.Event]

    conductor_E = None  # type: Callable[[],pyopencl.Event]
    conductor_H = None  # type: Callable[[],pyopencl.Event]

    cpml_E = None       # type: Callable[[],pyopencl.Event]
    cpml_H = None       # type: Callable[[],pyopencl.Event]

    cpml_psi_E = None   # type: Dict[str, pyopencl.array.Array]
    cpml_psi_H = None   # type: Dict[str, pyopencl.array.Array]

    def __init__(self,
                 epsilon: List[numpy.ndarray],
                 dt: float=.99/numpy.sqrt(3),
                 initial_E: List[numpy.ndarray]=None,
                 initial_H: List[numpy.ndarray]=None,
                 context: pyopencl.Context=None,
                 queue: pyopencl.CommandQueue=None,
                 float_type: numpy.float32 or numpy.float64=numpy.float32):
        """
        Initialize the simulation.

        :param epsilon: List containing [eps_r,xx, eps_r,yy, eps_r,zz], where each element is a Yee-shifted ndarray
                spanning the simulation domain. Relative epsilon is used.
        :param dt: Time step. Default is the Courant factor.
        :param initial_E: Initial E-field (default is 0 everywhere). Same format as epsilon.
        :param initial_H: Initial H-field (default is 0 everywhere). Same format as epsilon.
        :param context: pyOpenCL context. If not given, pyopencl.create_some_context(False) is called.
        :param queue: pyOpenCL command queue. If not given, pyopencl.CommandQueue(context) is called.
        :param float_type: numpy.float32 or numpy.float64. Default numpy.float32.
        """

        if len(epsilon) != 3:
            Exception('Epsilon must be a list with length of 3')
        if not all((e.shape == epsilon[0].shape for e in epsilon[1:])):
            Exception('All epsilon grids must have the same shape. Shapes are {}', [e.shape for e in epsilon])

        if context is None:
            self.context = pyopencl.create_some_context(False)
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

        self.eps = [pyopencl.array.to_device(self.queue, e.astype(float_type)) for e in epsilon]

        if initial_E is None:
            self.E = [pyopencl.array.zeros_like(self.eps[0]) for _ in range(3)]
        else:
            if len(initial_E) != 3:
                Exception('Initial_E must be a list of length 3')
            if not all((E.shape == epsilon[0].shape for E in initial_E)):
                Exception('Initial_E list elements must have same shape as epsilon elements')
            self.E = [pyopencl.array.to_device(self.queue, E.astype(float_type)) for E in initial_E]

        if initial_H is None:
            self.H = [pyopencl.array.zeros_like(self.eps[0]) for _ in range(3)]
        else:
            if len(initial_H) != 3:
                Exception('Initial_H must be a list of length 3')
            if not all((H.shape == epsilon[0].shape for H in initial_H)):
                Exception('Initial_H list elements must have same shape as epsilon elements')
            self.H = [pyopencl.array.to_device(self.queue, H.astype(float_type)) for H in initial_H]

        ctype = type_to_C(self.arg_type)
        E_args = [ctype + ' *E' + c for c in 'xyz']
        H_args = [ctype + ' *H' + c for c in 'xyz']
        eps_args = [ctype + ' *eps' + c for c in 'xyz']
        dt_arg = [ctype + ' dt']

        sxyz = base.shape_source(epsilon[0].shape)
        E_source = sxyz + base.dixyz_source + base.maxwell_E_source
        H_source = sxyz + base.dixyz_source + base.maxwell_H_source

        E_update = ElementwiseKernel(self.context, operation=E_source,
                                     arguments=', '.join(E_args + H_args + dt_arg + eps_args))

        H_update = ElementwiseKernel(self.context, operation=H_source,
                                     arguments=', '.join(E_args + H_args + dt_arg))

        self.update_E = lambda e: E_update(*self.E, *self.H, self.dt, *self.eps, wait_for=e)
        self.update_H = lambda e: H_update(*self.E, *self.H, self.dt, wait_for=e)

    def init_cpml(self, pml_args: List[Dict]):
        """
        Initialize absorbing layers (cpml: complex phase matched layer). PMLs are not actual
         boundary conditions, so you should add a conducting boundary (.init_conductors()) for
         all directions in which you add PMLs.
        Allows use of self.cpml_E(events) and self.cpml_H(events).
        All necessary additional fields are created on the opencl device.

        :param pml_args: A list containing dictionaries which are passed to .boundary.cpml(...).
            The dt argument is set automatically, but the others must be passed in each entry
             of pml_args.
        """
        sxyz = base.shape_source(self.eps[0].shape)

        # Prepare per-iteration constants for later use
        pml_E_source = sxyz + base.dixyz_source + base.xyz_source
        pml_H_source = sxyz + base.dixyz_source + base.xyz_source

        psi_E = []
        psi_H = []
        psi_E_names = []
        psi_H_names = []
        for arg_set in pml_args:
            pml_data = boundary.cpml(dt=self.dt, **arg_set)

            pml_E_source += pml_data['E']
            pml_H_source += pml_data['H']

            ti = numpy.delete(range(3), arg_set['direction'])
            trans = [self.eps[0].shape[i] for i in ti]
            psi_shape = (8, trans[0], trans[1])

            psi_E += [pyopencl.array.zeros(self.queue, psi_shape, dtype=self.arg_type)
                      for _ in pml_data['psi_E']]
            psi_H += [pyopencl.array.zeros(self.queue, psi_shape, dtype=self.arg_type)
                      for _ in pml_data['psi_H']]

            psi_E_names += pml_data['psi_E']
            psi_H_names += pml_data['psi_H']

        ctype = type_to_C(self.arg_type)
        E_args = [ctype + ' *E' + c for c in 'xyz']
        H_args = [ctype + ' *H' + c for c in 'xyz']
        eps_args = [ctype + ' *eps' + c for c in 'xyz']
        dt_arg = [ctype + ' dt']
        arglist_E = [ctype + ' *' + psi for psi in psi_E_names]
        arglist_H = [ctype + ' *' + psi for psi in psi_H_names]
        pml_E_args = ', '.join(E_args + H_args + dt_arg + eps_args + arglist_E)
        pml_H_args = ', '.join(E_args + H_args + dt_arg + arglist_H)

        pml_E = ElementwiseKernel(self.context, arguments=pml_E_args, operation=pml_E_source)
        pml_H = ElementwiseKernel(self.context, arguments=pml_H_args, operation=pml_H_source)

        self.cpml_E = lambda e: pml_E(*self.E, *self.H, self.dt, *self.eps, *psi_E, wait_for=e)
        self.cpml_H = lambda e: pml_H(*self.E, *self.H, self.dt, *psi_H, wait_for=e)
        self.cmpl_psi_E = {k: v for k, v in zip(psi_E_names, psi_E)}
        self.cmpl_psi_H = {k: v for k, v in zip(psi_H_names, psi_H)}

    def init_conductors(self, conductor_args: List[Dict]):
        """
        Initialize reflecting boundary conditions.
        Allows use of self.conductor_E(events) and self.conductor_H(events).

        :param conductor_args: List of dictionaries with which to call .boundary.conductor(...).
        """

        sxyz = base.shape_source(self.eps[0].shape)

        # Prepare per-iteration constants
        bc_E_source = sxyz + base.dixyz_source + base.xyz_source
        bc_H_source = sxyz + base.dixyz_source + base.xyz_source
        for arg_set in conductor_args:
            [e, h] = boundary.conductor(**arg_set)
            bc_E_source += e
            bc_H_source += h

        E_args = [type_to_C(self.arg_type) + ' *E' + c for c in 'xyz']
        H_args = [type_to_C(self.arg_type) + ' *H' + c for c in 'xyz']
        bc_E = ElementwiseKernel(self.context, arguments=E_args, operation=bc_E_source)
        bc_H = ElementwiseKernel(self.context, arguments=H_args, operation=bc_H_source)

        self.conductor_E = lambda e: bc_E(*self.E, wait_for=e)
        self.conductor_H = lambda e: bc_H(*self.H, wait_for=e)
