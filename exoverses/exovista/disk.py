import astropy.units as u
import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d

import exoverses.base as base


class ExovistaDisk(base.disk.Disk):
    def __init__(self, infile, fits_ext, star):
        self.star = star
        with open(infile, "rb") as f:
            obj_data, obj_header = fits.getdata(
                f, ext=fits_ext, header=True, memmap=False
            )
            self.ev_wavelengths = (
                fits.getdata(f, ext=fits_ext - 1, header=False, memmap=False) * u.um
            )

        # The debris disk contrast cube (disk flux divided by star flux),
        # removing the last because it is a 2d map estimating the fractional
        # numerical noise in the contrast calculations
        self.contrast = obj_data[:-1]

        self.disk_contrast_interp = interp1d(
            np.arange(len(self.ev_wavelengths)),
            self.contrast,
            kind="cubic",
            axis=0,
        )

    def spec_flux_density(self, wavelengths, times):
        """
        Calculate the spectral flux density of the disk at the given times and
        wavelengths
        Args:
            wavelengths (astropy Quantity array):
                Wavelengths to calculate the spectral flux density at
            times (astropy Time array):
                Times to calculate the spectral flux density at
        Returns:
            disk_flux_density (astropy Quantity array):
                Spectral flux density of the disk at the given times in Jy. The
                shape of the cube is (Ntimes, Nwavelengths, Npix, Npix)
        """
        # Determine the indices the wavelengths we want to calculate the scene
        # are with respect to the disk wavelengths
        inds = np.searchsorted(self.ev_wavelengths, wavelengths) - 1

        # Determine the fractional index, with respect to the indices of the
        # wavelengths the disk was generated at, of the wavelengths we want to
        # calculate the scene at, since the disk interpolant is over the
        # wavelength index instead of the actual wavelength
        fracinds = inds + (
            np.log(wavelengths.to(u.um).value)
            - np.log(self.ev_wavelengths[inds].to(u.um).value)
        ) / (
            np.log(self.ev_wavelengths[inds + 1].to(u.um).value)
            - np.log(self.ev_wavelengths[inds].to(u.um).value)
        )
        # Interpolate the disk to our desired wavelengths
        disk_contrast = self.disk_contrast_interp(fracinds).T

        # Instantiate the disk flux density cube
        disk_flux_density = (
            np.zeros(
                (
                    len(times),
                    len(wavelengths),
                    self.contrast.shape[1],
                    self.contrast.shape[2],
                )
            )
            * u.Jy
        )
        # Calculate the star's spectral flux density at the desired wavelengths
        star_flux_density = self.star.spec_flux_density(wavelengths, times)
        for i, _ in enumerate(times):
            disk_flux_density[i] = np.multiply(disk_contrast, star_flux_density[i]).T
        return disk_flux_density
