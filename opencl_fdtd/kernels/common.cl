{#
/* Common code for E, H updates
 *
 * Template parameters:
 *  ftype       type name (e.g. float or double)
 *  shape       list of 3 ints specifying shape of fields
 */
#}

typedef {{ftype}} ftype;


/*
 * Field size info
 */
const size_t sx = {{shape[0]}};
const size_t sy = {{shape[1]}};
const size_t sz = {{shape[2]}};
const size_t field_size = sx * sy * sz;

//Since we use i to index into Ex[], Ey[], ... rather than E[], do nothing if
// i is outside the bounds of Ex[].
if (i >= field_size) {
    PYOPENCL_ELWISE_CONTINUE;
}


/*
 * Array indexing
 */
// Given a linear index i and shape (sx, sy, sz), defines x, y, and z
//  as the 3D indices of the current element (i).
// (ie, converts linear index [i] to field indices (x, y, z)
const size_t x = i / (sz * sy);
const size_t y = (i - x * sz * sy) / sz;
const size_t z = (i - y * sz - x * sz * sy);

// Calculate linear index offsets corresponding to offsets in 3D
// (ie, if E[i] <-> E(x, y, z), then E[i + diy] <-> E(x, y + 1, z)
const size_t dix = sz * sy;
const size_t diy = sz;
const size_t diz = 1;


/*
 * Pointer math
 */
//Pointer offsets into the components of a linearized vector-field
// (eg. Hx = H + XX, where H and Hx are pointers)
const size_t XX = 0;
const size_t YY = field_size;
const size_t ZZ = field_size * 2;

//Define pointers to vector components of each field (eg. Hx = H + XX)
__global ftype *Ex = E + XX;
__global ftype *Ey = E + YY;
__global ftype *Ez = E + ZZ;

__global ftype *Hx = H + XX;
__global ftype *Hy = H + YY;
__global ftype *Hz = H + ZZ;


/*
 * Implement periodic boundary conditions
 *
 * mx ([m]inus [x]) gives the index offset of the adjacent cell in the minus-x direction.
 * In the event that we start at x == 0, we actually want to wrap around and grab the cell
 * x_{-1} == (sx - 1) instead, ie. mx = (sx - 1) * dix .
 *
 * px ([p]lus [x]) gives the index offset of the adjacent cell in the plus-x direction.
 * In the event that we start at x == (sx - 1), we actually want to wrap around and grab
 * the cell x_{+1} == 0 instead, ie. px = -(sx - 1) * dix .
 */
{% for r in 'xyz' %}
int m{{r}} = - (int) di{{r}};
int p{{r}} = + (int) di{{r}};
int wrap_{{r}} = (s{{r}} - 1) * (int) di{{r}};
if ( {{r}} == 0 ) {
  m{{r}} = wrap_{{r}};
}
if ( {{r}} == s{{r}} - 1 ) {
  p{{r}} = -wrap_{{r}};
}
{% endfor %}
