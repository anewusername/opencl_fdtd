/*
 *  Update E-field from J field
 *
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *
 *  OpenCL args:
 *   E, Jr, Ji, c, s, xmin, xmax, ymin, ymax, zmin, zmax
 */

{{common_header}}

////////////////////////////////////////////////////////////////////////////

__global ftype *Jrx = Jr + XX;
__global ftype *Jry = Jr + YY;
__global ftype *Jrz = Jr + ZZ;
__global ftype *Jix = Ji + XX;
__global ftype *Jiy = Ji + YY;
__global ftype *Jiz = Ji + ZZ;


if (x < xmin || y < ymin || z < zmin) {
   PYOPENCL_ELWISE_CONTINUE;
}
if (x >= xmax || y >= ymax || z >= zmax) {
   PYOPENCL_ELWISE_CONTINUE;
}

Ex[i] += c * Jrx[i] + s * Jix[i];
Ey[i] += c * Jry[i] + s * Jiy[i];
Ez[i] += c * Jrz[i] + s * Jiz[i];
