/*
 *  Update E-field, including any PMLs.
 *  
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *   pmls: [('x', 'n'), ('z', 'p'),...] list of pml axes and polarities
 *   pml_thickness: Number of cells (integer)
 *   
 *  OpenCL args:
 *   E, H, dt, S, oS
 */

{{common_header}}

//////////////////////////////////////////////////////////////////////


/*
 * Calculate S from oS (pre-calculated components)
 */ 
__global ftype *Sx = S + XX;
__global ftype *Sy = S + YY;
__global ftype *Sz = S + ZZ;

// Use unscaled S components from H locations 
__global ftype *oSxy = oS + 0 * field_size;
__global ftype *oSyz = oS + 1 * field_size;
__global ftype *oSzx = oS + 2 * field_size;
__global ftype *oSxz = oS + 3 * field_size;
__global ftype *oSyx = oS + 4 * field_size;
__global ftype *oSzy = oS + 5 * field_size;

ftype s_factor = dt * 0.125;
Sx[i] = (oSxy[i] + oSxz[i] + oSxy[i + my] + oSxz[i + mz]) * s_factor;
Sy[i] = (oSyz[i] + oSyx[i] + oSyz[i + mz] + oSyx[i + mx]) * s_factor;
Sz[i] = (oSzx[i] + oSzy[i] + oSzx[i + mx] + oSzy[i + my]) * s_factor;
