import unittest
import numpy

from opencl_fdtd import Simulation
from meanas import fdtd


class BasicTests():
    def test_initial_fields(self):
        # Make sure initial fields didn't change
        e0 = self.es[0]
        h0 = self.hs[0]
        mask = self.src_mask

        self.assertEqual(e0[mask], self.j_mag / self.epsilon[mask])
        self.assertFalse(e0[~mask].any())
        self.assertFalse(h0.any())


    def test_initial_energy(self):
        e0 = self.es[0]
        h0 = self.hs[0]
        h1 = self.hs[1]
        mask = self.src_mask[1]
        dxes = self.dxes if self.dxes is not None else tuple(tuple(numpy.ones(s) for s in e0.shape[1:]) for _ in range(2))
        dV = numpy.prod(numpy.meshgrid(*dxes[0], indexing='ij'), axis=0)
        u0 = self.j_mag * self.j_mag / self.epsilon[self.src_mask] * dV[mask]
        args = {
            'dxes': self.dxes,
            'epsilon': self.epsilon,
            }

        # Make sure initial energy and E dot J are correct
        energy0 = fdtd.energy_estep(h0=h0, e1=e0, h2=self.hs[1], **args)
        e_dot_j_0 = fdtd.delta_energy_j(j0=(e0 - 0) * self.epsilon, e1=e0, dxes=self.dxes)
        self.assertTrue(numpy.allclose(energy0[mask], u0))
        self.assertFalse(energy0[~mask].any(), msg=f'{energy0=}')
        self.assertTrue(numpy.allclose(e_dot_j_0[mask], u0))
        self.assertFalse(e_dot_j_0[~mask].any(), msg=f'{e_dot_j_0=}')


    def test_energy_conservation(self):
        e0 = self.es[0]
        u0 = fdtd.delta_energy_j(j0=(e0 - 0) * self.epsilon, e1=e0, dxes=self.dxes).sum()
        args = {'dxes': self.dxes,
                'epsilon': self.epsilon}

        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args)
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args)
                self.assertTrue(numpy.allclose(u_hstep.sum(), u0), msg=f'u_hstep: {u_hstep.sum()}\n{numpy.moveaxis(u_hstep, -1, 0)}')
                self.assertTrue(numpy.allclose(u_estep.sum(), u0), msg=f'u_estep: {u_estep.sum()}\n{numpy.moveaxis(u_estep, -1, 0)}')


    def test_poynting(self):
        for ii in range(1, 3):
            with self.subTest(i=ii):
                s = fdtd.poynting(e=self.es[ii], h=self.hs[ii+1] + self.hs[ii])
                sf = numpy.moveaxis(s, -1, 0)
                ss = numpy.moveaxis(self.ss[ii], -1, 0)
                self.assertTrue(numpy.allclose(s, self.ss[ii], rtol=1e-4),
                                msg=f'From ExH:\n{sf}\nFrom sim.S:\n{ss}')


    def test_poynting_divergence(self):
        args = {
            'dxes': self.dxes,
            'epsilon': self.epsilon,
            }

        dxes = self.dxes if self.dxes is not None else tuple(tuple(numpy.ones(s) for s in self.epsilon.shape[1:]) for _ in range(2))
        dV = numpy.prod(numpy.meshgrid(*dxes[0], indexing='ij'), axis=0)

        u_eprev = None
        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args)
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args)

                du_half_h2e = u_estep - u_hstep
                div_s_h2e = self.dt * fdtd.poynting_divergence(e=self.es[ii], h=self.hs[ii], dxes=self.dxes) * dV

                du_half_h2e_f = numpy.moveaxis(du_half_h2e, -1, 0)
                div_s_h2e_f = -numpy.moveaxis(div_s_h2e, -1, 0)
                self.assertTrue(numpy.allclose(du_half_h2e, -div_s_h2e, rtol=1e-4),
                                msg=f'du_half_h2e\n{du_half_h2e_f}\ndiv_s_h2e\n{div_s_h2e_f}')

                if u_eprev is None:
                    u_eprev = u_estep
                    continue

                # previous half-step
                du_half_e2h = u_hstep - u_eprev
                div_s_e2h = self.dt * fdtd.poynting_divergence(e=self.es[ii-1], h=self.hs[ii], dxes=self.dxes) * dV
                du_half_e2h_f = numpy.moveaxis(du_half_e2h, -1, 0)
                div_s_e2h_f = -numpy.moveaxis(div_s_e2h, -1, 0)
                self.assertTrue(numpy.allclose(du_half_e2h, -div_s_e2h, rtol=1e-4),
                                msg=f'du_half_e2h\n{du_half_e2h_f}\ndiv_s_e2h\n{div_s_e2h_f}')
                u_eprev = u_estep


    def test_poynting_planes(self):
        args = {
            'dxes': self.dxes,
            'epsilon': self.epsilon,
            }
        dxes = self.dxes if self.dxes is not None else tuple(tuple(numpy.ones(s) for s in self.epsilon.shape[1:]) for _ in range(2))
        dV = numpy.prod(numpy.meshgrid(*dxes[0], indexing='ij'), axis=0)

        mx = numpy.roll(self.src_mask, (-1, -1), axis=(0, 1))
        my = numpy.roll(self.src_mask, -1, axis=2)
        mz = numpy.roll(self.src_mask, (+1, -1), axis=(0, 3))
        px = numpy.roll(self.src_mask, -1, axis=0)
        py = self.src_mask.copy()
        pz = numpy.roll(self.src_mask, +1, axis=0)

        u_eprev = None
        for ii in range(1, 8):
            with self.subTest(i=ii):
                u_hstep = fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args)
                u_estep = fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args)

                s_h2e = -fdtd.poynting(e=self.es[ii], h=self.hs[ii]) * self.dt
                s_h2e[0] *= dxes[0][1][None, :, None] * dxes[0][2][None, None, :]
                s_h2e[1] *= dxes[0][0][:, None, None] * dxes[0][2][None, None, :]
                s_h2e[2] *= dxes[0][0][:, None, None] * dxes[0][1][None, :, None]
                planes = [s_h2e[px].sum(), -s_h2e[mx].sum(),
                          s_h2e[py].sum(), -s_h2e[my].sum(),
                          s_h2e[pz].sum(), -s_h2e[mz].sum()]
                du = (u_estep - u_hstep)[self.src_mask[1]]
                self.assertTrue(numpy.allclose(sum(planes), (u_estep - u_hstep)[self.src_mask[1]]),
                                msg=f'planes: {planes} (sum: {sum(planes)})\n du:\n {du}')

                if u_eprev is None:
                    u_eprev = u_estep
                    continue

                s_e2h = -fdtd.poynting(e=self.es[ii - 1], h=self.hs[ii]) * self.dt
                s_e2h[0] *= dxes[0][1][None, :, None] * dxes[0][2][None, None, :]
                s_e2h[1] *= dxes[0][0][:, None, None] * dxes[0][2][None, None, :]
                s_e2h[2] *= dxes[0][0][:, None, None] * dxes[0][1][None, :, None]
                planes = [s_e2h[px].sum(), -s_e2h[mx].sum(),
                          s_e2h[py].sum(), -s_e2h[my].sum(),
                          s_e2h[pz].sum(), -s_e2h[mz].sum()]
                du = (u_hstep - u_eprev)[self.src_mask[1]]
                self.assertTrue(numpy.allclose(sum(planes), (u_hstep - u_eprev)[self.src_mask[1]]),
                                msg=f'planes: {du} (sum: {sum(planes)})\n du:\n {du}')

                # previous half-step
                u_eprev = u_estep


class Basic2DNoDXOnlyVacuum(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 1]
        self.dt = 0.5
        self.epsilon = numpy.ones(shape, dtype=float)
        self.j_mag = 32
        self.dxes = None

        self.src_mask = numpy.zeros_like(self.epsilon, dtype=bool)
        self.src_mask[1, 2, 2, 0] = True

        e = numpy.zeros_like(self.epsilon)
        h = numpy.zeros_like(self.epsilon)
        e[self.src_mask] = self.j_mag / self.epsilon[self.src_mask]
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True)

        for _ in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()
            self.es.append(sim.E.get().reshape(shape))
            self.hs.append(sim.H.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


class Basic2DUniformDX3(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 1, 5]
        self.dt = 0.5
        self.j_mag = 32
        self.dxes = tuple(tuple(numpy.full(s, 2.0) for s in shape[1:]) for _ in range(2))

        self.src_mask = numpy.zeros(shape, dtype=bool)
        self.src_mask[1, 2, 0, 2] = True

        self.epsilon = numpy.full(shape, 1, dtype=float)
        self.epsilon[self.src_mask] = 2

        e = numpy.zeros_like(self.epsilon)
        h = numpy.zeros_like(self.epsilon)
        e[self.src_mask] = self.j_mag / self.epsilon[self.src_mask]
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True)

        for _ in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()
            self.es.append(sim.E.get().reshape(shape))
            self.hs.append(sim.H.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


class Basic3DUniformDXOnlyVacuum(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 5]
        self.dt = 0.5
        self.epsilon = numpy.ones(shape, dtype=float)
        self.j_mag = 32
        self.dxes = tuple(tuple(numpy.ones(s) for s in shape[1:]) for _ in range(2))

        self.src_mask = numpy.zeros_like(self.epsilon, dtype=bool)
        self.src_mask[1, 2, 2, 2] = True

        e = numpy.zeros_like(self.epsilon)
        h = numpy.zeros_like(self.epsilon)
        e[self.src_mask] = self.j_mag / self.epsilon[self.src_mask]
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True)

        for _ in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()
            self.es.append(sim.E.get().reshape(shape))
            self.hs.append(sim.H.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


class Basic3DUniformDXUniformN(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 5]
        self.dt = 0.5
        self.epsilon = numpy.full(shape, 2.5, dtype=float)
        self.j_mag = 32
        self.dxes = tuple(tuple(numpy.ones(s) for s in shape[1:]) for _ in range(2))

        self.src_mask = numpy.zeros_like(self.epsilon, dtype=bool)
        self.src_mask[1, 2, 2, 2] = True

        e = numpy.zeros_like(self.epsilon)
        h = numpy.zeros_like(self.epsilon)
        e[self.src_mask] = self.j_mag / self.epsilon[self.src_mask]
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True)

        for _ in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()
            self.es.append(sim.E.get().reshape(shape))
            self.hs.append(sim.H.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


class Basic3DUniformDX(unittest.TestCase, BasicTests):
    def setUp(self):
        shape = [3, 5, 5, 5]
        self.dt = 0.33
        self.j_mag = 32
        self.dxes = tuple(tuple(numpy.ones(s) for s in shape[1:]) for _ in range(2))

        self.src_mask = numpy.zeros(shape, dtype=bool)
        self.src_mask[1, 2, 2, 2] = True

        self.epsilon = numpy.full(shape, 1, dtype=float)
        self.epsilon[self.src_mask] = 2

        e = numpy.zeros_like(self.epsilon)
        h = numpy.zeros_like(self.epsilon)
        e[self.src_mask] = self.j_mag / self.epsilon[self.src_mask]
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True)

        for _ in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()
            self.es.append(sim.E.get().reshape(shape))
            self.hs.append(sim.H.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


class JdotE_3DUniformDX(unittest.TestCase):
    def setUp(self):
        shape = [3, 5, 5, 5]
        self.dt = 0.5
        self.j_mag = 32
        self.dxes = tuple(tuple(numpy.full(s, 2.0) for s in shape[1:]) for _ in range(2))

        self.src_mask = numpy.zeros(shape, dtype=bool)
        self.src_mask[1, 2, 2, 2] = True

        self.epsilon = numpy.full(shape, 4, dtype=float)
        self.epsilon[self.src_mask] = 2

        e = numpy.random.randint(-128, 128 + 1, size=shape).astype(float)
        h = numpy.random.randint(-128, 128 + 1, size=shape).astype(float)
        self.es = [e]
        self.hs = [h]
        self.ss = []

        sim = Simulation(epsilon=self.epsilon, pmls=[], dt=self.dt, dxes=self.dxes,
                         initial_fields={'E': e, 'H': h}, do_poynting=True, float_type=numpy.float64)

        eh2h = fdtd.maxwell_h(dt=self.dt, dxes=self.dxes)
        eh2e = fdtd.maxwell_e(dt=self.dt, dxes=self.dxes)
        for ii in range(9):
            e = e.copy()
            h = h.copy()
            sim.update_H([]).wait()
            sim.update_E([]).wait()

            if ii == 1:
                nn = numpy.where(self.src_mask.flat)[0][0]
                self.e_before = sim.E.get().reshape(shape)
                sim.E[nn] += self.j_mag / self.epsilon[self.src_mask][0]
                self.e_after = sim.E.get().reshape(shape)
                self.j_dot_e = self.j_mag * e[self.src_mask]

            self.hs.append(sim.H.get().reshape(shape))
            self.es.append(sim.E.get().reshape(shape))
            self.ss.append(sim.S.get().reshape(shape))


    def test_j_dot_e(self):
        h0 = self.hs[2]
        e0 = self.es[1]
        e1 = self.es[2]
        j0 = numpy.zeros_like(e0)
        j0[self.src_mask] = self.j_mag
        args = {
            'dxes': self.dxes,
            'epsilon': self.epsilon,
            }
        e2h = fdtd.maxwell_h(dt=self.dt, dxes=self.dxes)

        #ee = j0 * (2 * e0 - j0)
        #hh = h0 * e2h(j0, numpy.zeros_like(h0))
        #u0 = fdtd.dxmul(ee, hh, **args)
        u0 = fdtd.delta_energy_j(j0=j0, e1=(e0 + e1), dxes=self.dxes)

        #uh = [fdtd.energy_hstep(e0=self.es[ii-1], h1=self.hs[ii], e2=self.es[ii], **args) for ii in range(1, 9)]
        #ue = [fdtd.energy_estep(h0=self.hs[ii], e1=self.es[ii], h2=self.hs[ii + 1], **args) for ii in range(8)]
        #uht = [uu.sum() for uu in uh]
        #uet = [uu.sum() for uu in ue]

        u_hstep = fdtd.energy_hstep(e0=self.es[0], h1=self.hs[1], e2=self.es[1], **args)
        u_estep = fdtd.energy_estep(h0=self.hs[-2], e1=self.es[-2], h2=self.hs[-1], **args)
        #breakpoint()
        du = (u_estep - u_hstep).sum()
        self.assertTrue(numpy.allclose(u0.sum(), (u_estep - u_hstep).sum()), msg=f'{u0.sum()} != {du}')
