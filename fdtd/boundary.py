from typing import List, Dict
import numpy


def conductor(direction: int,
              polarity: int,
              ) -> List[str]:
    """
    Create source code for conducting boundary conditions.

    :param direction: integer in range(3), corresponding to x,y,z.
    :param polarity: -1 or 1, specifying eg. a -x or +x boundary.
    :return: [E_source, H_source] source code for E and H boundary update steps.
    """
    if direction not in range(3):
        raise Exception('Invalid direction: {}'.format(direction))

    if polarity not in (-1, 1):
        raise Exception('Invalid polarity: {}'.format(polarity))

    r = 'xyz'[direction]
    uv = 'xyz'.replace(r, '')

    if polarity < 0:
        bc_E = """
if ({r} == 0) {{
    E{r}[i] = 0;
    E{u}[i] = E{u}[i+di{r}];
    E{v}[i] = E{v}[i+di{r}];
}}
"""
        bc_H = """
if ({r} == 0) {{
    H{r}[i] = H{r}[i+di{r}];
    H{u}[i] = 0;
    H{v}[i] = 0;
}}
"""

    elif polarity > 0:
        bc_E = """
if ({r} == s{r} - 1) {{
    E{r}[i] = -E{r}[i-2*di{r}];
    E{u}[i] = +E{u}[i-di{r}];
    E{v}[i] = +E{v}[i-di{r}];
}} else if ({r} == s{r} - 2) {{
    E{r}[i] = 0;
}}
"""
        bc_H = """
if ({r} == s{r} - 1) {{
    H{r}[i] = +H{r}[i-di{r}];
    H{u}[i] = -H{u}[i-2*di{r}];
    H{v}[i] = -H{v}[i-2*di{r}];
}} else if ({r} == s{r} - 2) {{
    H{u}[i] = 0;
    H{v}[i] = 0;
}}
"""
    else:
        raise Exception()

    replacements = {'r': r, 'u': uv[0], 'v': uv[1]}
    return [s.format(**replacements) for s in (bc_E, bc_H)]


def cpml(direction: int,
         polarity: int,
         dt: float,
         thickness: int=8,
         epsilon_eff: float=1,
         ) -> Dict:
    """
    Generate source code for complex phase matched layer (cpml) absorbing boundaries.
    These are not full boundary conditions and require a conducting boundary to be added
     in the same direction as the pml.

    :param direction: integer in range(3), corresponding to x, y, z directions.
    :param polarity: -1 or 1, corresponding to eg. -x or +x direction.
    :param dt: timestep used by the simulation
    :param thickness: Number of cells used by the pml (the grid is NOT expanded to add these cells). Default 8.
    :param epsilon_eff: Effective epsilon_r of the pml layer. Default 1.
    :return: Dict with entries 'E', 'H' (update equations for E and H) and 'psi_E', 'psi_H' (lists of str,
            specifying the field names of the cpml fields used in the E and H update steps. Eg.,
            Psi_xn_Ex for the complex Ex component for the x- pml.)
    """
    if direction not in range(3):
        raise Exception('Invalid direction: {}'.format(direction))

    if polarity not in (-1, 1):
        raise Exception('Invalid polarity: {}'.format(polarity))

    if thickness <= 2:
        raise Exception('It would be wise to have a pml with 4+ cells of thickness')

    if epsilon_eff <= 0:
        raise Exception('epsilon_eff must be positive')

    m = (3.5, 1)
    sigma_max = 0.8 * (m[0] + 1) / numpy.sqrt(epsilon_eff)
    alpha_max = 0  # TODO: Decide what to do about non-zero alpha
    transverse = numpy.delete(range(3), direction)

    r = 'xyz'[direction]
    np = 'nVp'[numpy.sign(polarity)+1]
    uv = ['xyz'[i] for i in transverse]

    xE = numpy.arange(1, thickness+1, dtype=float)[::-1]
    xH = numpy.arange(1, thickness+1, dtype=float)[::-1]
    if polarity > 0:
        xE -= 0.5
    elif polarity < 0:
        xH -= 0.5

    def par(x):
        sigma = ((x / thickness) ** m[0]) * sigma_max
        alpha = ((1 - x / thickness) ** m[1]) * alpha_max
        p0 = numpy.exp(-(sigma + alpha) * dt)
        p1 = sigma / (sigma + alpha) * (p0 - 1)
        return p0, p1
    p0e, p1e = par(xE)
    p0h, p1h = par(xH)

    vals = {'r': r,
            'u': uv[0],
            'v': uv[1],
            'np':   np,
            'th':   thickness,
            'p0e': ', '.join((str(x) for x in p0e)),
            'p1e': ', '.join((str(x) for x in p1e)),
            'p0h': ', '.join((str(x) for x in p0h)),
            'p1h': ', '.join((str(x) for x in p1h)),
            'se': '-+'[direction % 2],
            'sh': '+-'[direction % 2]}

    if polarity < 0:
        bounds_if = """
if ( 0 < {r} && {r} < {th} + 1 ) {{
    const int ir = {r} - 1;                             // index into pml parameters
    const int ip = {v} + {u} * s{v} + ir * s{v} * s{u};   // linear index into Psi
"""
    elif polarity > 0:
        bounds_if = """
if ( (s{r} - 1) > {r} && {r} > (s{r} - 1) - ({th} + 1) ) {{
    const int ir = (s{r} - 1) - ({r} + 1);              // index into pml parameters
    const int ip = {v} + {u} * s{v} + ir * s{v} * s{u};   // linear index into Psi
"""
    else:
        raise Exception('Bad polarity (=0)')

    code_E = """
    // pml parameters:
    const float p0[{th}] = {{ {p0e} }};
    const float p1[{th}] = {{ {p1e} }};

    Psi_{r}{np}_E{u}[ip] = p0[ir] * Psi_{r}{np}_E{u}[ip] + p1[ir] * (H{v}[i] - H{v}[i-di{r}]);
    Psi_{r}{np}_E{v}[ip] = p0[ir] * Psi_{r}{np}_E{v}[ip] + p1[ir] * (H{u}[i] - H{u}[i-di{r}]);

    E{u}[i] {se}= dt / eps{u}[i] * Psi_{r}{np}_E{u}[ip];
    E{v}[i] {sh}= dt / eps{v}[i] * Psi_{r}{np}_E{v}[ip];
}}
"""
    code_H = """
    // pml parameters:
    const float p0[{th}] = {{ {p0h} }};
    const float p1[{th}] = {{ {p1h} }};

    Psi_{r}{np}_H{u}[ip] = p0[ir] * Psi_{r}{np}_H{u}[ip] + p1[ir] * (E{v}[i+di{r}] - E{v}[i]);
    Psi_{r}{np}_H{v}[ip] = p0[ir] * Psi_{r}{np}_H{v}[ip] + p1[ir] * (E{u}[i+di{r}] - E{u}[i]);

    H{u}[i] {sh}= dt * Psi_{r}{np}_H{u}[ip];
    H{v}[i] {se}= dt * Psi_{r}{np}_H{v}[ip];
}}
"""

    pml_data = {
        'E': (bounds_if + code_E).format(**vals),
        'H': (bounds_if + code_H).format(**vals),
        'psi_E': ['Psi_{r}{np}_E{u}'.format(**vals),
                  'Psi_{r}{np}_E{v}'.format(**vals)],
        'psi_H': ['Psi_{r}{np}_H{u}'.format(**vals),
                  'Psi_{r}{np}_H{v}'.format(**vals)],
    }

    return pml_data
