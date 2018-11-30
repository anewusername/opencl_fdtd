/*
 *  Update H-field, including any PMLs.
 *   Also precalculate values for poynting vector if necessary.
 *
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *   pmls: [{'axis': 'x', 'polarity': 'n', 'thickness': 8}, ...] list of pml dicts containing
 *      axes, polarities, and thicknesses.
 *   do_poynting: Whether to precalculate poynting vector components (boolean)
 *
 *  OpenCL args:
 *   E, H, dt, [p{xyz}{01}h{np}, Psi_{xyz}{np}_H], [oS]
 */

{{common_header}}

////////////////////////////////////////////////////////////////////////////

/*
 *  Precalculate derivatives
 */
ftype dExy = Ex[i + py] - Ex[i];
ftype dExz = Ex[i + pz] - Ex[i];

ftype dEyx = Ey[i + px] - Ey[i];
ftype dEyz = Ey[i + pz] - Ey[i];

ftype dEzx = Ez[i + px] - Ez[i];
ftype dEzy = Ez[i + py] - Ez[i];


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




{%- if do_poynting %}


/*
 *  Precalculate averaged E
 */
ftype aExy = Ex[i + py] + Ex[i];
ftype aExz = Ex[i + pz] + Ex[i];

ftype aEyx = Ey[i + px] + Ey[i];
ftype aEyz = Ey[i + pz] + Ey[i];

ftype aEzx = Ez[i + px] + Ez[i];
ftype aEzy = Ez[i + py] + Ez[i];
{%- endif %}




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
    {%- endif -%}

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
{% if do_poynting -%}
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
// Average H across timesteps
ftype aHxt = Hx[i] + Hx_old;
ftype aHyt = Hy[i] + Hy_old;
ftype aHzt = Hz[i] + Hz_old;

/*
 *  Calculate unscaled S components at H locations
 */
__global ftype *oSxy = oS + 0 * field_size;
__global ftype *oSyz = oS + 1 * field_size;
__global ftype *oSzx = oS + 2 * field_size;
__global ftype *oSxz = oS + 3 * field_size;
__global ftype *oSyx = oS + 4 * field_size;
__global ftype *oSzy = oS + 5 * field_size;

oSxy[i] = aEyx * aHzt;
oSxz[i] = -aEzx * aHyt;
oSyz[i] = aEzy * aHxt;
oSyx[i] = -aExy * aHzt;
oSzx[i] = aExz * aHyt;
oSzy[i] = -aEyz * aHxt;
{%- endif -%}
