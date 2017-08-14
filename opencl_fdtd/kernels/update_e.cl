/*
 *  Update E-field, including any PMLs.
 *  
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *   pmls: [('x', 'n'), ('z', 'p'),...] list of pml axes and polarities
 *   pml_thickness: Number of cells (integer)
 *   
 *  OpenCL args:
 *   E, H, dt, eps, [p{01}e{np}, Psi_{xyz}{np}_E]
 */

{{common_header}}

////////////////////////////////////////////////////////////////////////////

__global ftype *epsx = eps + XX;
__global ftype *epsy = eps + YY;
__global ftype *epsz = eps + ZZ;

{% if pmls -%}
const int pml_thickness = {{pml_thickness}};
{%- endif %}

/*
 *   Precalclate derivatives
 */
ftype dHxy = Hx[i] - Hx[i + my];
ftype dHxz = Hx[i] - Hx[i + mz];

ftype dHyx = Hy[i] - Hy[i + mx];
ftype dHyz = Hy[i] - Hy[i + mz];

ftype dHzx = Hz[i] - Hz[i + mx];
ftype dHzy = Hz[i] - Hz[i + my];

/*
 *   PML Update
 */
// PML effects on E
ftype pExi = 0;
ftype pEyi = 0;
ftype pEzi = 0;

{% for r, p in pmls -%}
    {%- set u, v = ['x', 'y', 'z'] | reject('equalto', r) -%}
    {%- set psi = 'Psi_' ~ r ~ p ~ '_E' -%}
    {%- if r != 'y' -%}
        {%- set se, sh = '-', '+' -%}
    {%- else -%}
        {%- set se, sh = '+', '-' -%}
    {%- endif -%}

    {%- if p == 'n' %}

if ( {{r}} < pml_thickness ) {
    const size_t ir = {{r}};                          // index into pml parameters

    {%- elif p == 'p' %}

if ( s{{r}} > {{r}} && {{r}} >= s{{r}} - pml_thickness ) {
    const size_t ir = (s{{r}} - 1) - {{r}};                     // index into pml parameters

    {%- endif %}
    const size_t ip = {{v}} + {{u}} * s{{v}} + ir * s{{v}} * s{{u}};  // linear index into Psi
    {{psi ~ u}}[ip] = p0e{{p}}[ir] * {{psi ~ u}}[ip] + p1e{{p}}[ir] * dH{{v ~ r}};
    {{psi ~ v}}[ip] = p0e{{p}}[ir] * {{psi ~ v}}[ip] + p1e{{p}}[ir] * dH{{u ~ r}};
    pE{{u}}i {{se}}= {{psi ~ u}}[ip];
    pE{{v}}i {{sh}}= {{psi ~ v}}[ip];
}
{%- endfor %}

/*
 *  Update E
 */
Ex[i] += dt / epsx[i] * (dHzy - dHyz + pExi);
Ey[i] += dt / epsy[i] * (dHxz - dHzx + pEyi);
Ez[i] += dt / epsz[i] * (dHyx - dHxy + pEzi);
