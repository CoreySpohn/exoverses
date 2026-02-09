"""JAX-friendly debris-disk model for exoverses.

Equinox module holding a wavelength-interpolated contrast cube.
"""

from __future__ import annotations

import equinox as eqx
import interpax
import jax.numpy as jnp

from exoverses.jax.star import Star


class Disk(eqx.Module):
    """JAX-friendly debris disk (exozodiacal light).

    Stores a wavelength-interpolated contrast cube relative to the host
    star.  Call :meth:`spec_flux_density` to get actual flux.
    """

    star: Star
    pixel_scale_arcsec: float
    _wavelengths_nm: jnp.ndarray  # (n_wl,)
    _contrast_cube: jnp.ndarray  # (n_wl, ny, nx)
    _contrast_interp: interpax.CubicSpline

    def __init__(
        self,
        star: Star,
        pixel_scale_arcsec: float,
        wavelengths_nm: jnp.ndarray,
        contrast_cube: jnp.ndarray,
    ):
        """
        Args:
            star: Host star.
            pixel_scale_arcsec: Pixel scale of the contrast cube [arcsec/pixel].
            wavelengths_nm: Wavelength grid [nm], shape ``(n_wl,)``.
            contrast_cube: Disk-to-star contrast, shape ``(n_wl, ny, nx)``.
        """
        self.star = star
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self._wavelengths_nm = wavelengths_nm
        self._contrast_cube = contrast_cube
        self._contrast_interp = interpax.CubicSpline(
            wavelengths_nm, contrast_cube, axis=0
        )

    def spec_flux_density(
        self, wavelength_nm: float, time_jd: float
    ) -> jnp.ndarray:
        """Disk flux density [ph/s/m²/nm], shape ``(ny, nx)``."""
        contrast = self._contrast_interp(wavelength_nm)
        star_flux = self.star.spec_flux_density(wavelength_nm, time_jd)
        return contrast * star_flux

    def spatial_extent(self) -> tuple[float, float]:
        """Spatial extent of the disk [arcsec]."""
        ny, nx = self._contrast_cube.shape[-2:]
        return (nx * self.pixel_scale_arcsec, ny * self.pixel_scale_arcsec)
