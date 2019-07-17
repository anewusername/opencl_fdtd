/*
 *  Update H-field, including any PMLs.
 *   Also precalculate values for poynting vector if necessary.
 *
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *   pmls: [{'axis': 'x', 'polarity': 'n', 'thickness': 8}, ...] list of pml dicts containing
 *      axes, polarities, and thicknesses.
 *   do_poynting: Whether to calculate poynting vector (boolean)
 *   do_poynting_halves: Whether to calculate half-step poynting vectors (boolean)
 *   uniform_dx: If grid is uniform, uniform_dx should be the grid spacing.
 *      Otherwise, uniform_dx should be False and [inv_de{xyz}] arrays must be supplied as
 *      OpenCL args.
 *
 *  OpenCL args:
 *   E, H, dt, [inv_de{xyz}], [p{xyz}{01}h{np}, Psi_{xyz}{np}_H], [S], [S0, S1]
 */

{{common_header}}

////////////////////////////////////////////////////////////////////////////

/*
 *  Precalculate derivatives
 */
{%- if uniform_dx %}
ftype inv_dx = 1.0 / {{uniform_dx}};
ftype inv_dy = 1.0 / {{uniform_dx}};
ftype inv_dz = 1.0 / {{uniform_dx}};
{%- else %}
ftype inv_dx = inv_dex[x];
ftype inv_dy = inv_dey[y];
ftype inv_dz = inv_dez[z];
{%- endif %}


ftype dExy = (Ex[i + py] - Ex[i]) * inv_dy;
ftype dExz = (Ex[i + pz] - Ex[i]) * inv_dz;

ftype dEyx = (Ey[i + px] - Ey[i]) * inv_dx;
ftype dEyz = (Ey[i + pz] - Ey[i]) * inv_dz;

ftype dEzx = (Ez[i + px] - Ez[i]) * inv_dx;
ftype dEzy = (Ez[i + py] - Ez[i]) * inv_dy;


{% for bloch in bloch_boundaries -%}
    {%- set r = bloch['axis'] -%}
    {%- set u, v = ['x', 'y', 'z'] | reject('equalto', r) -%}
if ({{r}} == s{{r}} - 1) {
    ftype bloch_re = {{bloch['real']}};
    ftype bloch_im = {{bloch['imag']}};
    dE{{u ~ r}} = bloch_re * dE{{u ~ r}} + bloch_im * (F{{u}}[i + p{{u}}] - F{{u}}[i]);
    dE{{v ~ r}} = bloch_re * dE{{v ~ r}} + bloch_im * (F{{v}}[i + p{{v}}] - F{{v}}[i]);
}
{%- endfor %}



/*
 *  PML Update
 */
// PML contributions to H
ftype pHxi = 0;
ftype pHyi = 0;
ftype pHzi = 0;

{% for pml in pmls -%}
    {%- set r = pml['axis'] -%}
    {%- set p = pml['polarity'] -%}
    {%- set u, v = ['x', 'y', 'z'] | reject('equalto', r) -%}
    {%- set psi = 'Psi_' ~ r ~ p ~ '_H' -%}
    {%- if r != 'y' -%}
        {%- set se, sh = '-', '+' -%}
    {%- else -%}
        {%- set se, sh = '+', '-' -%}
    {%- endif %}

int pml_{{r ~ p}}_thickness = {{pml['thickness']}};

    {%- if p == 'n' %}

if ( {{r}} < pml_{{r ~ p}}_thickness ) {
    const size_t ir = {{r}};                          // index into pml parameters

    {%- elif p == 'p' %}

if ( s{{r}} > {{r}} && {{r}} >= s{{r}} - pml_{{r ~ p}}_thickness ) {
    const size_t ir = (s{{r}} - 1) - {{r}};                     // index into pml parameters

    {%- endif %}
    const size_t ip = {{v}} + {{u}} * s{{v}} + ir * s{{v}} * s{{u}};  // linear index into Psi
    dE{{v ~ r}} *= p{{r}}2h{{p}}[ir];
    dE{{u ~ r}} *= p{{r}}2h{{p}}[ir];
    {{psi ~ u}}[ip] = p{{r}}0h{{p}}[ir] * {{psi ~ u}}[ip] + p{{r}}1h{{p}}[ir] * dE{{v ~ r}};
    {{psi ~ v}}[ip] = p{{r}}0h{{p}}[ir] * {{psi ~ v}}[ip] + p{{r}}1h{{p}}[ir] * dE{{u ~ r}};
    pH{{u}}i {{sh}}= {{psi ~ u}}[ip];
    pH{{v}}i {{se}}= {{psi ~ v}}[ip];
}
{%- endfor %}

/*
 *  Update H
 */
{% if do_poynting or do_poynting_halves -%}
// Save old H for averaging
ftype Hx_old = Hx[i];
ftype Hy_old = Hy[i];
ftype Hz_old = Hz[i];
{%- endif %}

// H update equations
Hx[i] -= dt * (dEzy - dEyz - pHxi);
Hy[i] -= dt * (dExz - dEzx - pHyi);
Hz[i] -= dt * (dEyx - dExy - pHzi);

{% if do_poynting -%}
/*
 *  Calculate unscaled S components at ??? locations
 *  //TODO: document S locations and types
 */
__global ftype *Sx = S + XX;
__global ftype *Sy = S + YY;
__global ftype *Sz = S + ZZ;

// Average H across timesteps
ftype aHxt = Hx[i] + Hx_old;
ftype aHyt = Hy[i] + Hy_old;
ftype aHzt = Hz[i] + Hz_old;

Sx[i] = Ey[i + px] * aHzt - Ez[i + px] * aHyt;
Sy[i] = Ez[i + py] * aHxt - Ex[i + py] * aHzt;
Sz[i] = Ex[i + pz] * aHyt - Ey[i + pz] * aHxt;
{%- endif -%}

{% if do_poynting_halves -%}
/*
 *  Calculate unscaled S components at ??? locations
 *  //TODO: document S locations and types
 */
__global ftype *Sx0 = S0 + XX;
__global ftype *Sy0 = S0 + YY;
__global ftype *Sz0 = S0 + ZZ;
__global ftype *Sx1 = S1 + XX;
__global ftype *Sy1 = S1 + YY;
__global ftype *Sz1 = S1 + ZZ;

Sx0[i] = Ey[i + px] * Hz_old - Ez[i + px] * Hy_old;
Sy0[i] = Ez[i + py] * Hx_old - Ex[i + py] * Hz_old;
Sz0[i] = Ex[i + pz] * Hy_old - Ey[i + pz] * Hx_old;
Sx1[i] = Ey[i + px] * Hz[i] - Ez[i + px] * Hy[i];
Sy1[i] = Ez[i + py] * Hx[i] - Ex[i + py] * Hz[i];
Sz1[i] = Ex[i + pz] * Hy[i] - Ey[i + pz] * Hx[i];
{%- endif -%}
