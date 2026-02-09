"""JAX-friendly planet model for exoverses.

Equinox module wrapping ``orbix.Planets`` for orbital propagation and
an interpax 3-D interpolator for wavelength- and phase-dependent contrast.
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp
from orbix.equations.orbit import mean_anomaly_tp
from orbix.kepler.shortcuts.grid import get_grid_solver
from orbix.system.planets import Planets as OrbixPlanets

from exoverses.jax.star import Star

# Scalar trig solver for Kepler's equation
TRIG_SOLVER = get_grid_solver(level="scalar", E=False, trig=True, jit=True)


class Planet(eqx.Module):
    """JAX-friendly collection of planets with contrast + orbital data.

    Wraps ``orbix.Planets`` for Keplerian propagation and stores a 3-D
    contrast interpolator (wavelength × mean anomaly × planet index).
    """

    star: Star
    orbix_planet: OrbixPlanets
    contrast_interp: interpax.Interpolator3D
    n_planets: int

    def __init__(
        self,
        star: Star,
        orbix_planet: OrbixPlanets,
        contrast_interp: interpax.Interpolator3D,
    ):
        self.star = star
        self.orbix_planet = orbix_planet
        self.contrast_interp = contrast_interp
        self.n_planets = orbix_planet.a.shape[0]

    # ── Orbital propagation ──────────────────────────────────────────

    def mean_anomaly(self, time_jd: float) -> jnp.ndarray:
        """Mean anomalies at *time_jd* [deg], shape ``(n_planets,)``."""
        return jnp.rad2deg(
            mean_anomaly_tp(time_jd, self.orbix_planet.n, self.orbix_planet.tp)
            % (2 * jnp.pi)
        )

    def position(self, time_jd: float) -> jnp.ndarray:
        """On-sky (dRA, dDec) in arcsec, shape ``(2, n_planets)``."""
        ra, dec = self.orbix_planet.prop_ra_dec(
            TRIG_SOLVER, jnp.atleast_1d(time_jd)
        )
        return jnp.stack([ra[:, 0], dec[:, 0]])

    def alpha_dMag(self, time_jd: float):
        """Angular separation [arcsec] and delta-mag, each ``(n_planets,)``."""
        alpha, dMag = self.orbix_planet.alpha_dMag(
            TRIG_SOLVER, jnp.atleast_1d(time_jd)
        )
        return alpha[:, 0], dMag[:, 0]

    # ── Spectral contrast ────────────────────────────────────────────

    def contrast(self, wavelength_nm: float, time_jd: float) -> jnp.ndarray:
        """Planet-to-star contrast at given wavelength and time, ``(n_planets,)``."""
        mean_anomalies_deg = self.mean_anomaly(time_jd)
        planet_indices = jnp.arange(self.n_planets)
        interp = jax.vmap(self.contrast_interp, in_axes=(None, 0, 0))
        return interp(wavelength_nm, mean_anomalies_deg, planet_indices)

    def spec_flux_density(
        self, wavelength_nm: float, time_jd: float
    ) -> jnp.ndarray:
        """Planet flux density [ph/s/m²/nm], shape ``(n_planets,)``."""
        c = self.contrast(wavelength_nm, time_jd)
        star_flux = self.star.spec_flux_density(wavelength_nm, time_jd)
        return c * star_flux
