/*
 *  Update E-field, including any PMLs.
 *
 *  Template parameters:
 *   common_header: Rendered contents of common.cl
 *   pmls: [{'axis': 'x', 'polarity': 'n', 'thickness': 8}, ...] list of pml dicts containing
 *      axes, polarities, and thicknesses.
 *   uniform_dx: If grid is uniform, uniform_dx should be the grid spacing.
 *      Otherwise, uniform_dx should be False and [inv_dh{xyz}] arrays must be supplied as
 *      OpenCL args.
 *
 *  OpenCL args:
 *   E, H, dt, eps, [p{012}e{np}, Psi_{xyz}{np}_E], [inv_dh{xyz}]
 */

{{common_header}}

////////////////////////////////////////////////////////////////////////////

__global ftype *epsx = eps + XX;
__global ftype *epsy = eps + YY;
__global ftype *epsz = eps + ZZ;


{%- if uniform_dx %}
ftype inv_dx = 1.0 / {{uniform_dx}};
ftype inv_dy = 1.0 / {{uniform_dx}};
ftype inv_dz = 1.0 / {{uniform_dx}};
{%- else %}
ftype inv_dx = inv_dhx[x];
ftype inv_dy = inv_dhy[y];
ftype inv_dz = inv_dhz[z];
{%- endif %}


/*
 *   Precalculate derivatives
 */
ftype dHxy = (Hx[i] - Hx[i + my]) * inv_dy;
ftype dHxz = (Hx[i] - Hx[i + mz]) * inv_dz;

ftype dHyx = (Hy[i] - Hy[i + mx]) * inv_dx;
ftype dHyz = (Hy[i] - Hy[i + mz]) * inv_dz;

ftype dHzx = (Hz[i] - Hz[i + mx]) * inv_dx;
ftype dHzy = (Hz[i] - Hz[i + my]) * inv_dy;

{% for bloch in bloch_boundaries -%}
    {%- set r = bloch['axis'] -%}
    {%- set u, v = ['x', 'y', 'z'] | reject('equalto', r) -%}
if ({{r}} == 0) {
    ftype bloch_im = {{bloch['real']}};
    ftype bloch_re = {{bloch['imag']}};
    dH{{u ~ r}} = bloch_re * dH{{v ~ r}} + bloch_im * (G{{u}}[i] - G{{u}}[i + m{{u}}]);
    dH{{v ~ r}} = bloch_re * dH{{v ~ r}} + bloch_im * (G{{v}}[i] - G{{v}}[i + m{{v}}]);
}
{%- endfor %}


/*
 *   PML Update
 */
// PML effects on E
ftype pExi = 0;
ftype pEyi = 0;
ftype pEzi = 0;

{% for pml in pmls -%}
    {%- set r = pml['axis'] -%}
    {%- set p = pml['polarity'] -%}
    {%- set u, v = ['x', 'y', 'z'] | reject('equalto', r) -%}
    {%- set psi = 'Psi_' ~ r ~ p ~ '_E' -%}
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
    dH{{v ~ r}} *= p{{r}}2e{{p}}[ir];
    dH{{u ~ r}} *= p{{r}}2e{{p}}[ir];
    {{psi ~ u}}[ip] = p{{r}}0e{{p}}[ir] * {{psi ~ u}}[ip] + p{{r}}1e{{p}}[ir] * dH{{v ~ r}};
    {{psi ~ v}}[ip] = p{{r}}0e{{p}}[ir] * {{psi ~ v}}[ip] + p{{r}}1e{{p}}[ir] * dH{{u ~ r}};
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
