"""
Routines for creating normalized 2D lattices and common photonic crystal
 cavity designs.
"""

from typing import List

import numpy


def triangular_lattice(dims: List[int],
                       asymmetrical: bool=False
                       ) -> numpy.ndarray:
    """
    Return an ndarray of [[x0, y0], [x1, y1], ...] denoting lattice sites for
     a triangular lattice in 2D. The lattice will be centered around (0, 0),
     unless asymmetrical=True in which case there will be extra holes in the +x
     direction.

    :param dims: Number of lattice sites in the [x, y] directions.
    :param asymmetrical: If true, each row in x will contain the same number of
            lattice sites. If false, the structure is symmetrical around (0, 0).
    :return: [[x0, y0], [x1, 1], ...] denoting lattice sites.
    """
    dims = numpy.array(dims, dtype=int)

    if asymmetrical:
        k = 0
    else:
        k = 1

    positions = []
    ymax = (dims[1] - 1)/2
    for j in numpy.linspace(-ymax, ymax, dims[0]):
        j_odd = numpy.floor(j) % 2

        x_offset = j_odd * 0.5
        y_offset = j * numpy.sqrt(3)/2

        num_x = dims[0] - k * j_odd
        xmax = (dims[0] - 1)/2
        xs = numpy.linspace(-xmax, xmax - k * j_odd, num_x) + x_offset
        ys = numpy.full_like(xs, y_offset)

        positions += [numpy.vstack((xs, ys)).T]

    xy = numpy.vstack(tuple(positions))
    return xy[xy[:, 0].argsort(), ]


def square_lattice(dims: List[int]) -> numpy.ndarray:
    """
    Return an ndarray of [[x0, y0], [x1, y1], ...] denoting lattice sites for
     a square lattice in 2D. The lattice will be centered around (0, 0).

    :param dims: Number of lattice sites in the [x, y] directions.
    :return: [[x0, y0], [x1, 1], ...] denoting lattice sites.
    """
    xs, ys = numpy.meshgrid(range(dims[0]), range(dims[1]), 'xy')
    xs -= dims[0]/2
    ys -= dims[1]/2
    xy = numpy.vstack((xs.flatten(), ys.flatten())).T
    return xy[xy[:, 0].argsort(), ]

# ### Photonic crystal functions ###


def nanobeam_holes(a_defect: float,
                   num_defect_holes: int,
                   num_mirror_holes: int
                   ) -> numpy.ndarray:
    """
    Returns a list of [[x0, r0], [x1, r1], ...] of nanobeam hole positions and radii.
     Creates a region in which the lattice constant and radius are progressively
     (linearly) altered over num_defect_holes holes until they reach the value
     specified by a_defect, then symmetrically returned to a lattice constant and
     radius of 1, which is repeated num_mirror_holes times on each side.

    :param a_defect: Minimum lattice constant for the defect, as a fraction of the
            mirror lattice constant (ie., for no defect, a_defect = 1).
    :param num_defect_holes: How many holes form the defect (per-side)
    :param num_mirror_holes: How many holes form the mirror (per-side)
    :return: Ndarray [[x0, r0], [x1, r1], ...] of nanobeam hole positions and radii.
    """
    a_values = numpy.linspace(a_defect, 1, num_defect_holes, endpoint=False)
    xs = a_values.cumsum() - (a_values[0] / 2)  # Later mirroring makes center distance 2x as long
    mirror_xs = numpy.arange(1, num_mirror_holes + 1) + xs[-1]
    mirror_rs = numpy.ones_like(mirror_xs)
    return numpy.vstack((numpy.hstack((-mirror_xs[::-1], -xs[::-1], xs, mirror_xs)),
                         numpy.hstack((mirror_rs[::-1], a_values[::-1], a_values, mirror_rs)))).T


def ln_defect(mirror_dims: List[int], defect_length: int) -> numpy.ndarray:
    """
    N-hole defect in a triangular lattice.

    :param mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
    :param defect_length: Length of defect. Should be an odd number.
    :return: [[x0, y0], [x1, y1], ...] for all the holes
    """
    if defect_length % 2 != 1:
        raise Exception('defect_length must be odd!')
    p = triangular_lattice([2 * d + 1 for d in mirror_dims])
    half_length = numpy.floor(defect_length / 2)
    hole_nums = numpy.arange(-half_length, half_length + 1)
    holes_to_keep = numpy.in1d(p[:, 0], hole_nums, invert=True)
    return p[numpy.logical_or(holes_to_keep, p[:, 1] != 0), ]


def ln_shift_defect(mirror_dims: List[int],
                    defect_length: int,
                    shifts_a: List[float]=(0.15, 0, 0.075),
                    shifts_r: List[float]=(1, 1, 1)
                    ) -> numpy.ndarray:
    """
    N-hole defect with shifted holes (intended to give the mode a gaussian profile
     in real- and k-space so as to improve both Q and confinement). Holes along the
     defect line are shifted and altered according to the shifts_* parameters.

    :param mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
    :param defect_length: Length of defect. Should be an odd number.
    :param shifts_a: Percentage of a to shift (1st, 2nd, 3rd,...) holes along the defect line
    :param shifts_r: Factor to multiply the radius by. Should match length of shifts_a
    :return: [[x0, y0, r0], [x1, y1, r1], ...] for all the holes
    """
    if not hasattr(shifts_a, "__len__") and shifts_a is not None:
        shifts_a = [shifts_a]
    if not hasattr(shifts_r, "__len__") and shifts_r is not None:
        shifts_r = [shifts_r]

    xy = ln_defect(mirror_dims, defect_length)

    # Add column for radius
    xyr = numpy.hstack((xy, numpy.ones((xy.shape[0], 1))))

    # Shift holes
    # Expand shifts as necessary
    n_shifted = max(len(shifts_a), len(shifts_r))

    tmp_a = numpy.array(shifts_a)
    shifts_a = numpy.ones((n_shifted, ))
    shifts_a[:len(tmp_a)] = tmp_a

    tmp_r = numpy.array(shifts_r)
    shifts_r = numpy.ones((n_shifted, ))
    shifts_r[:len(tmp_r)] = tmp_r

    x_removed = numpy.floor(defect_length / 2)

    for ind in range(n_shifted):
        for sign in (-1, 1):
            x_val = sign * (x_removed + ind + 1)
            which = numpy.logical_and(xyr[:, 0] == x_val, xyr[:, 1] == 0)
            xyr[which, ] = (x_val + numpy.sign(x_val) * shifts_a[ind], 0, shifts_r[ind])

    return xyr


def r6_defect(mirror_dims: List[int]) -> numpy.ndarray:
    """
    R6 defect in a triangular lattice.

    :param mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
    :return: [[x0, y0], [x1, y1], ...] specifying hole centers.
    """
    xy = triangular_lattice([2 * d + 1 for d in mirror_dims])

    rem_holes_plus = numpy.array([[1, 0],
                                  [0.5, +numpy.sqrt(3)/2],
                                  [0.5, -numpy.sqrt(3)/2]])
    rem_holes = numpy.vstack((rem_holes_plus, -rem_holes_plus))

    for rem_xy in rem_holes:
        xy = xy[(xy != rem_xy).any(axis=1), ]

    return xy


def l3_shift_perturbed_defect(mirror_dims: List[int],
                              perturbed_radius: float=1.1,
                              shifts_a: List[float]=(),
                              shifts_r: List[float]=()
                              ) -> numpy.ndarray:
    """
    3-hole defect with perturbed hole sizes intended to form an upwards-directed
     beam. Can also include shifted holes along the defect line, intended
     to give the mode a more gaussian profile to improve Q.

    :param mirror_dims: [x, y] mirror lengths (number of holes). Total number of holes
            is 2 * n + 1 in each direction.
    :param perturbed_radius: Amount to perturb the radius of the holes used for beam-forming
    :param shifts_a: Percentage of a to shift (1st, 2nd, 3rd,...) holes along the defect line
    :param shifts_r: Factor to multiply the radius by. Should match length of shifts_a
    :return: [[x0, y0, r0], [x1, y1, r1], ...] for all the holes
    """
    xyr = ln_shift_defect(mirror_dims, 3, shifts_a, shifts_r)

    abs_x, abs_y = (numpy.fabs(xyr[:, i]) for i in (0, 1))

    # Sorted unique xs and ys
    # Ignore row y=0 because it might have shifted holes
    xs = numpy.unique(abs_x[abs_x != 0])
    ys = numpy.unique(abs_y)

    # which holes should be perturbed? (xs[[3, 7]], ys[1]) and (xs[[2, 6]], ys[2])
    perturbed_holes = ((xs[a], ys[b]) for a, b in ((3, 1), (7, 1), (2, 2), (6, 2)))
    for row in xyr:
        if numpy.fabs(row) in perturbed_holes:
            row[2] = perturbed_radius
    return xyr
