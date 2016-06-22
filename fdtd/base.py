"""
Basic code snippets for opencl FDTD
"""

from typing import List
import numpy


def shape_source(shape: List[int] or numpy.ndarray) -> str:
    """
    Defines sx, sy, sz C constants specifying the shape of the grid in each of the 3 dimensions.

    :param shape: [sx, sy, sz] values.
    :return: String containing C source.
    """
    sxyz = """
// Field sizes
const int sx = {shape[0]};
const int sy = {shape[1]};
const int sz = {shape[2]};
""".format(shape=shape)
    return sxyz

# Defines dix, diy, diz constants used for stepping in the x, y, z directions in a linear array
#  (ie, given Ex[i] referring to position (x, y, z), Ex[i+diy] will refer to position (x, y+1, z))
dixyz_source = """
// Convert offset in field xyz to linear index offset
const int dix = sz * sy;
const int diy = sz;
const int diz = 1;
"""

# Given a linear index i and shape sx, sy, sz, defines x, y, and z as the 3D indices of the current element (i).
xyz_source = """
// Convert linear index to field index (xyz)
const int x = i / (sz * sy);
const int y = (i - x * sz * sy) / sz;
const int z = (i - y * sz - x * sz * sy);
"""

# Source code for updating the E field; maxes use of dixyz_source.
maxwell_E_source = """
// E update equations
Ex[i] += dt / epsx[i] * ((Hz[i] - Hz[i-diy]) - (Hy[i] - Hy[i-diz]));
Ey[i] += dt / epsy[i] * ((Hx[i] - Hx[i-diz]) - (Hz[i] - Hz[i-dix]));
Ez[i] += dt / epsz[i] * ((Hy[i] - Hy[i-dix]) - (Hx[i] - Hx[i-diy]));
"""

# Source code for updating the H field; maxes use of dixyz_source and assumes mu=0
maxwell_H_source = """
// H update equations
Hx[i] -= dt * ((Ez[i+diy] - Ez[i]) - (Ey[i+diz] - Ey[i]));
Hy[i] -= dt * ((Ex[i+diz] - Ex[i]) - (Ez[i+dix] - Ez[i]));
Hz[i] -= dt * ((Ey[i+dix] - Ey[i]) - (Ex[i+diy] - Ex[i]));
"""


def type_to_C(float_type: numpy.float32 or numpy.float64) -> str:
    """
    Returns a string corresponding to the C equivalent of a numpy type.
    Only works for float32 and float64.

    :param float_type: numpy.float32 or numpy.float64
    :return: string containing the corresponding C type (eg. 'double')
    """
    if float_type == numpy.float32:
        arg_type = 'float'
    elif float_type == numpy.float64:
        arg_type = 'double'
    else:
        raise Exception('Unsupported type')
    return arg_type
