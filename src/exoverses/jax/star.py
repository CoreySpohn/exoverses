"""JAX-friendly star model for exoverses.

Equinox module holding stellar spectral data with interpax interpolation.
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax
import jax.numpy as jnp


# Physical constants (same values as coronagraphoto/orbix)
_Jy = 1e-26  # W m^-2 Hz^-1
_h = 6.62607015e-34  # J s
_Msun2kg = 1.988409870698051e30
_mas2arcsec = 1e-3
_um2nm = 1e3


def _jy_to_photons_per_nm_per_m2(flux_jy, wavelength_nm):
    """Convert Jy → ph/s/nm/m²."""
    return flux_jy * _Jy / (wavelength_nm * _h)


def _decimal_year_to_jd(decimal_year):
    """Convert decimal year → Julian Date (simplified J2000-based)."""
    year = jnp.floor(decimal_year)
    year_fraction = decimal_year - year

    def _gregorian_to_jd(y, m, d):
        a = jnp.floor((14 - m) / 12)
        y2 = y + 4800 - a
        m2 = m + 12 * a - 3
        jdn = (
            d
            + jnp.floor((153 * m2 + 2) / 5)
            + 365 * y2
            + jnp.floor(y2 / 4)
            - jnp.floor(y2 / 100)
            + jnp.floor(y2 / 400)
            - 32045
        )
        return jdn - 0.5

    jd_start = _gregorian_to_jd(year, 1, 1)
    jd_end = _gregorian_to_jd(year + 1, 1, 1)
    return jd_start + year_fraction * (jd_end - jd_start)


class Star(eqx.Module):
    """JAX-friendly stellar source.

    Stores spectral flux density as ph/s/m²/nm via interpax interpolation
    over wavelength (nm) and time (JD).
    """

    dist_pc: float
    mass_kg: float
    ra_deg: float
    dec_deg: float
    midplane_pa_deg: float
    midplane_i_deg: float
    diameter_arcsec: float
    luminosity_lsun: float

    # Spectral data arrays
    _wavelengths_nm: jnp.ndarray  # (n_wl,)
    _times_jd: jnp.ndarray  # (n_t,)
    _flux_density_phot: jnp.ndarray  # (n_wl, n_t) ph/s/m²/nm

    # Interpolator
    _flux_interp: interpax.Interpolator2D

    def __init__(
        self,
        *,
        dist_pc: float,
        mass_kg: float,
        ra_deg: float = 0.0,
        dec_deg: float = 0.0,
        midplane_pa_deg: float = 0.0,
        midplane_i_deg: float = 0.0,
        diameter_arcsec: float = 0.0,
        luminosity_lsun: float = 1.0,
        wavelengths_nm: jnp.ndarray,
        times_jd: jnp.ndarray,
        flux_density_jy: jnp.ndarray,
    ):
        """Initialize from Jansky flux.

        Args:
            dist_pc: Distance to star [pc].
            mass_kg: Stellar mass [kg].
            ra_deg: Right ascension [deg].
            dec_deg: Declination [deg].
            midplane_pa_deg: System midplane position angle [deg].
            midplane_i_deg: System midplane inclination [deg].
            diameter_arcsec: Angular diameter [arcsec].
            luminosity_lsun: Bolometric luminosity [L_sun].
            wavelengths_nm: 1-D wavelength grid [nm].
            times_jd: 1-D time grid [Julian days].
            flux_density_jy: Flux density in Jy, shape (n_wl, n_t).
        """
        self.dist_pc = dist_pc
        self.mass_kg = mass_kg
        self.ra_deg = ra_deg
        self.dec_deg = dec_deg
        self.midplane_pa_deg = midplane_pa_deg
        self.midplane_i_deg = midplane_i_deg
        self.diameter_arcsec = diameter_arcsec
        self.luminosity_lsun = luminosity_lsun
        self._wavelengths_nm = wavelengths_nm
        self._times_jd = times_jd

        # Jy → ph/s/nm/m² conversion (vectorized over time axis)
        self._flux_density_phot = jax.vmap(
            _jy_to_photons_per_nm_per_m2, in_axes=(1, None), out_axes=1
        )(flux_density_jy, wavelengths_nm)

        self._flux_interp = interpax.Interpolator2D(
            wavelengths_nm, times_jd, self._flux_density_phot, method="cubic"
        )

    def spec_flux_density(self, wavelength_nm: float, time_jd: float) -> float:
        """Scalar spectral flux density [ph/s/m²/nm]."""
        return self._flux_interp(wavelength_nm, time_jd)
