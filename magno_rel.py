import numpy as np

class MagnoRelSim2D:
    """
    2D toy simulator with primary Bz and emergent (Ex, Ey).
    """
    def __init__(self, nx, ny, dx, dy, dt, c=1.0, k_rel=0.0, alpha_emerge=0.0, beta_sat=0.0, sigma_damp=0.0):
        self.nx, self.ny = nx, ny
        self.dx, self.dy = dx, dy
        self.dt = dt
        self.c = c
        self.k_rel = k_rel
        self.alpha_emerge = alpha_emerge
        self.beta_sat = beta_sat
        self.sigma_damp = sigma_damp

        self.Ex = np.zeros((nx+1, ny))
        self.Ey = np.zeros((nx, ny+1))
        self.Bz = np.zeros((nx+1, ny+1))
        self.energy_hist = []

    def c_eff(self):
        if self.k_rel == 0.0:
            return self.c
        return self.c / np.sqrt(1.0 + self.k_rel * (self.Bz**2))

    def curl_E(self):
        dEx_dy = (self.Ex[1:self.nx, 1:self.ny] - self.Ex[1:self.nx, 0:self.ny-1]) / self.dy
        dEy_dx = (self.Ey[1:self.nx, 1:self.ny] - self.Ey[0:self.nx-1, 1:self.ny]) / self.dx
        return dEx_dy - dEy_dx

    def curl_B(self):
        dB_dy_for_Ex = (self.Bz[:, 1:] - self.Bz[:, :-1]) / self.dy
        dB_dx_for_Ey = (self.Bz[1:, :] - self.Bz[:-1, :]) / self.dx
        return dB_dy_for_Ex, -dB_dx_for_Ey

    def step(self, apply_source=None, t=0):
        c_eff = self.c_eff()
        curlE = self.curl_E()
        self.Bz[1:self.nx, 1:self.ny] += self.dt * (c_eff[1:self.nx, 1:self.ny]**2) * curlE

        if apply_source is not None:
            self.Bz += apply_source(t, self.Bz)

        dB_dy_for_Ex, dB_dx_for_Ey = self.curl_B()
        sat_Ex = 1.0 + self.beta_sat * (self.Ex**2)
        sat_Ey = 1.0 + self.beta_sat * (self.Ey**2)
        self.Ex += self.dt * ( self.alpha_emerge * dB_dy_for_Ex / sat_Ex - self.sigma_damp * self.Ex )
        self.Ey += self.dt * ( self.alpha_emerge * dB_dx_for_Ey / sat_Ey - self.sigma_damp * self.Ey )

        self._edge_damp(self.Ex)
        self._edge_damp(self.Ey)
        self._edge_damp(self.Bz)

        energy = (np.mean(self.Bz**2) + np.mean(self.Ex**2) + np.mean(self.Ey**2))
        self.energy_hist.append(energy)

    def _edge_damp(self, A, margin=4, factor=0.9):
        A[:margin, :] *= factor
        A[-margin:, :] *= factor
        A[:, :margin] *= factor
        A[:, -margin:] *= factor
