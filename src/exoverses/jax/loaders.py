"""Load ExoVista FITS files into JAX-friendly exoverses objects.

This module migrates the core loading logic from
``coronagraphoto.loaders.exovista`` into exoverses, creating
:class:`~exoverses.jax.system.System` objects directly.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import interpax
import jax.numpy as jnp
import numpy as np
from astropy.io.fits import getdata, getheader
from orbix.equations.orbit import mean_anomaly_tp
from orbix.system.planets import Planets as OrbixPlanets

from exoverses.jax.disk import Disk
from exoverses.jax.planet import Planet
from exoverses.jax.star import Star, _decimal_year_to_jd, _Msun2kg, _mas2arcsec, _um2nm
from exoverses.jax.system import System


# ── Orbital-element extraction ───────────────────────────────────────

# Physical constants for state-vector → Keplerian conversion
_G = 6.67430e-11  # m³ kg⁻¹ s⁻²
_AU2m = 1.495978707e11
_Mearth2kg = 5.972167867791379e24
_au_per_yr_to_m_per_s = _AU2m / (365.25 * 86400.0)


def _state_vector_to_keplerian(r, v, mu):
    """Convert (r, v) → (a, e, i, W, w, M) in SI, all angles in radians.

    Ported from coronagraphoto.transforms.orbital_mechanics.
    """
    r_mag = jnp.linalg.norm(r)
    v_mag = jnp.linalg.norm(v)

    h = jnp.cross(r, v)
    h_mag = jnp.linalg.norm(h)

    i = jnp.arccos(jnp.clip(h[2] / h_mag, -1.0, 1.0))

    k = jnp.array([0.0, 0.0, 1.0])
    n = jnp.cross(k, h)
    n_mag = jnp.linalg.norm(n)

    e_vec = (1 / mu) * ((v_mag**2 - mu / r_mag) * r - jnp.dot(r, v) * v)
    e = jnp.linalg.norm(e_vec)

    E_energy = 0.5 * v_mag**2 - mu / r_mag
    a = jnp.where(jnp.abs(E_energy) > 1e-10, -mu / (2 * E_energy), jnp.inf)

    TOL_E = 1e-9
    TOL_I = 1e-9
    is_circular = e < TOL_E
    is_inclined = n_mag > TOL_I

    W = jnp.where(is_inclined, jnp.arctan2(n[1], n[0]), 0.0)

    cos_w = jnp.dot(n, e_vec) / (n_mag * e)
    w_inclined = jnp.arccos(jnp.clip(cos_w, -1.0, 1.0))
    w_inclined = jnp.where(e_vec[2] < 0, 2 * jnp.pi - w_inclined, w_inclined)

    w_equatorial = jnp.arctan2(e_vec[1], e_vec[0])
    w_equatorial = w_equatorial * jnp.sign(h[2])

    w = jnp.where(is_circular, 0.0, jnp.where(is_inclined, w_inclined, w_equatorial))

    cos_nu = jnp.dot(e_vec, r) / (e * r_mag)
    nu_elliptical = jnp.arccos(jnp.clip(cos_nu, -1.0, 1.0))
    nu_elliptical = jnp.where(
        jnp.dot(r, v) < 0, 2 * jnp.pi - nu_elliptical, nu_elliptical
    )

    cos_u = jnp.dot(n, r) / (n_mag * r_mag)
    u_inclined = jnp.arccos(jnp.clip(cos_u, -1.0, 1.0))
    u_inclined = jnp.where(r[2] < 0, 2 * jnp.pi - u_inclined, u_inclined)

    nu_equatorial = jnp.arctan2(r[1], r[0])
    nu_equatorial = nu_equatorial * jnp.sign(h[2])

    nu = jnp.where(
        is_circular, jnp.where(is_inclined, u_inclined, nu_equatorial), nu_elliptical
    )

    W = W % (2 * jnp.pi)
    w = w % (2 * jnp.pi)
    nu = nu % (2 * jnp.pi)

    E_angle = jnp.arctan2(jnp.sqrt(1 - e**2) * jnp.sin(nu), e + jnp.cos(nu))
    M = E_angle - e * jnp.sin(E_angle)
    M = M % (2 * jnp.pi)
    M = jnp.where(e < 1.0, M, jnp.nan)

    return a, e, i, W, w, M


# ── Loaders ──────────────────────────────────────────────────────────


def _load_star(fits_file: str, fits_ext: int = 4) -> Star:
    """Load star from ExoVista FITS."""
    with open(fits_file, "rb") as f:
        obj_data, obj_header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)

    wavelengths_nm = jnp.asarray(wavelengths_um * _um2nm)
    times_year = jnp.asarray(2000.0 + obj_data[:, 0])
    times_jd = _decimal_year_to_jd(times_year)
    flux_density_jy = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

    diameter_arcsec = obj_header["ANGDIAM"] * _mas2arcsec
    mass_kg = obj_header.get("MASS") * _Msun2kg
    dist_pc = obj_header.get("DIST")
    midplane_pa = obj_header.get("PA", 0.0)
    midplane_i = obj_header.get("I", 0.0)
    ra_deg = obj_header.get("RA", 0.0)
    dec_deg = obj_header.get("DEC", 0.0)
    luminosity_lsun = obj_header.get("LSTAR", 1.0)

    return Star(
        dist_pc=dist_pc,
        mass_kg=mass_kg,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        midplane_pa_deg=midplane_pa,
        midplane_i_deg=midplane_i,
        diameter_arcsec=diameter_arcsec,
        luminosity_lsun=luminosity_lsun,
        wavelengths_nm=wavelengths_nm,
        times_jd=times_jd,
        flux_density_jy=flux_density_jy,
    )


def _load_planets(
    fits_file: str,
    star: Star,
    planet_indices: Sequence[int],
    required_planets: Optional[int] = None,
) -> Planet:
    """Load planets from ExoVista FITS."""
    planet_ext_start = 5
    oe_params: dict[str, list] = {
        "a": [],
        "e": [],
        "i": [],
        "W": [],
        "w": [],
        "M0": [],
        "mass": [],
        "radius": [],
        "p": [],
    }
    contrast_grids: list[jnp.ndarray] = []

    with open(fits_file, "rb") as f:
        wavelengths_um = getdata(f, ext=0, header=False, memmap=False)
    wavelengths_nm = jnp.asarray(wavelengths_um * _um2nm)

    t0 = None

    for idx in planet_indices:
        with open(fits_file, "rb") as f:
            obj_data, obj_header = getdata(
                f, ext=planet_ext_start + idx, header=True, memmap=False
            )

        times_year = jnp.asarray(2000.0 + obj_data[:, 0])
        times_jd = _decimal_year_to_jd(times_year)
        if t0 is None:
            t0 = times_jd[0]

        contrast_data = jnp.asarray(obj_data[:, 16:].T.astype(np.float32))

        # State vectors → orbital elements
        r_sky_au = obj_data[0, 9:12]
        v_sky_au_yr = obj_data[0, 12:15]
        r_sky_m = jnp.array(r_sky_au * _AU2m)
        v_sky_m_s = jnp.array(v_sky_au_yr * _au_per_yr_to_m_per_s)
        mass_earth = obj_header.get("M")
        planet_mass_kg = float(mass_earth) * _Mearth2kg
        total_mass_kg = star.mass_kg + planet_mass_kg
        mu = _G * total_mass_kg
        _a, _e, i_rad, W_rad, w_rad, M_rad = _state_vector_to_keplerian(
            r_sky_m, v_sky_m_s, mu
        )

        oe_params["a"].append(obj_header.get("A"))
        oe_params["e"].append(obj_header.get("E"))
        oe_params["i"].append(float(jnp.degrees(i_rad)))
        oe_params["W"].append(float(jnp.degrees(W_rad)))
        oe_params["w"].append(float(jnp.degrees(w_rad)))
        oe_params["M0"].append(float(jnp.degrees(M_rad)))
        oe_params["mass"].append(obj_header.get("M"))
        oe_params["radius"].append(obj_header.get("R"))
        oe_params["p"].append(obj_header.get("p", 0.2))

        # Mean anomaly → regular grid for contrast interpolation
        temp_planet = OrbixPlanets(
            Ms=jnp.atleast_1d(star.mass_kg),
            dist=jnp.atleast_1d(star.dist_pc),
            a=jnp.atleast_1d(oe_params["a"][-1]),
            e=jnp.atleast_1d(oe_params["e"][-1]),
            W=jnp.atleast_1d(jnp.deg2rad(oe_params["W"][-1])),
            i=jnp.atleast_1d(jnp.deg2rad(oe_params["i"][-1])),
            w=jnp.atleast_1d(jnp.deg2rad(oe_params["w"][-1])),
            M0=jnp.atleast_1d(jnp.deg2rad(oe_params["M0"][-1])),
            t0=jnp.atleast_1d(t0),
            Mp=jnp.atleast_1d(oe_params["mass"][-1]),
            Rp=jnp.atleast_1d(oe_params["radius"][-1]),
            p=jnp.atleast_1d(oe_params["p"][-1]),
        )
        mean_anom_coords = jnp.rad2deg(
            mean_anomaly_tp(times_jd, temp_planet.n, temp_planet.tp) % (2 * jnp.pi)
        )

        # Resample onto regular mean-anomaly grid
        sort_idx = jnp.argsort(mean_anom_coords)
        mean_anom_sorted = mean_anom_coords[sort_idx]
        contrast_sorted = contrast_data[:, sort_idx]

        mean_anomaly_grid = jnp.linspace(0, 360, 100)
        xq, yq = jnp.meshgrid(wavelengths_nm, mean_anomaly_grid, indexing="ij")
        contrast_grid = interpax.interp2d(
            xq.flatten(),
            yq.flatten(),
            wavelengths_nm,
            mean_anom_sorted,
            contrast_sorted,
            method="linear",
            extrap=True,
        ).reshape(xq.shape)
        contrast_grids.append(contrast_grid)

    n_loaded = len(planet_indices)

    # Ghost-planet padding for fixed array sizes
    if required_planets is not None:
        if n_loaded > required_planets:
            warnings.warn(
                f"Loaded {n_loaded} planets, but required_planets is {required_planets}. "
                f"Truncating to first {required_planets} planets.",
                UserWarning,
                stacklevel=2,
            )
            for key in oe_params:
                oe_params[key] = oe_params[key][:required_planets]
            contrast_grids = contrast_grids[:required_planets]
            n_loaded = required_planets

        n_ghosts = required_planets - n_loaded
        if n_ghosts > 0:
            oe_params["a"].extend([1.0] * n_ghosts)
            oe_params["e"].extend([0.0] * n_ghosts)
            oe_params["i"].extend([0.0] * n_ghosts)
            oe_params["W"].extend([0.0] * n_ghosts)
            oe_params["w"].extend([0.0] * n_ghosts)
            oe_params["M0"].extend([0.0] * n_ghosts)
            oe_params["mass"].extend([0.0] * n_ghosts)
            oe_params["radius"].extend([0.0] * n_ghosts)
            oe_params["p"].extend([0.0] * n_ghosts)

            base_shape = (
                contrast_grids[0].shape if n_loaded > 0 else (len(wavelengths_nm), 100)
            )
            zero_grid = jnp.zeros(base_shape, dtype=jnp.float32)
            contrast_grids.extend([zero_grid] * n_ghosts)

    n_total = len(oe_params["a"])

    # Build single OrbixPlanets object
    orbix_planets = OrbixPlanets(
        Ms=jnp.atleast_1d(star.mass_kg),
        dist=jnp.atleast_1d(star.dist_pc),
        a=jnp.array(oe_params["a"]),
        e=jnp.array(oe_params["e"]),
        W=jnp.deg2rad(jnp.array(oe_params["W"])),
        i=jnp.deg2rad(jnp.array(oe_params["i"])),
        w=jnp.deg2rad(jnp.array(oe_params["w"])),
        M0=jnp.deg2rad(jnp.array(oe_params["M0"])),
        t0=jnp.repeat(t0, n_total),
        Mp=jnp.array(oe_params["mass"]),
        Rp=jnp.array(oe_params["radius"]),
        p=jnp.array(oe_params["p"]),
    )

    # Stack contrast grids → 3D interpolator
    if n_total == 1:
        stacked = jnp.stack(contrast_grids * 2, axis=-1)
        interp_indices = jnp.array([0, 1])
    else:
        stacked = jnp.stack(contrast_grids, axis=-1)
        interp_indices = jnp.arange(n_total)

    contrast_interp = interpax.Interpolator3D(
        wavelengths_nm,
        mean_anomaly_grid,
        interp_indices,
        stacked,
        method="linear",
    )

    return Planet(
        star=star,
        orbix_planet=orbix_planets,
        contrast_interp=contrast_interp,
    )


def _load_disk(fits_file: str, fits_ext: int, star: Star) -> Disk:
    """Load debris disk from ExoVista FITS."""
    with open(fits_file, "rb") as f:
        obj_data, header = getdata(f, ext=fits_ext, header=True, memmap=False)
        wavelengths_um = getdata(f, ext=fits_ext - 1, header=False, memmap=False)

    wavelengths_nm = jnp.asarray(wavelengths_um * _um2nm)
    contrast_cube = jnp.asarray(obj_data[:-1].astype(np.float32))
    pixel_scale_arcsec = header["PXSCLMAS"] * _mas2arcsec

    return Disk(
        star=star,
        pixel_scale_arcsec=pixel_scale_arcsec,
        wavelengths_nm=wavelengths_nm,
        contrast_cube=contrast_cube,
    )


def get_earth_like_planet_indices(fits_file: str) -> list[int]:
    """Identify Earth-like planets in an ExoVista FITS file.

    Classification criteria (same as exoverses):
      • Scaled semi-major axis: 0.95 ≤ a / √L_star < 1.67 AU
      • Planet radius: 0.8 / √a_scaled ≤ R < 1.4 R_earth
    """
    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
        _, star_header = getdata(f, ext=4, header=True, memmap=False)

    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4
    star_luminosity_lsun = star_header.get("LSTAR", 1.0)

    earth_indices: list[int] = []
    for i in range(n_planets_total):
        with open(fits_file, "rb") as f:
            _, planet_header = getdata(f, ext=5 + i, header=True, memmap=False)
        a_au = planet_header.get("A", 1.0)
        radius_rearth = planet_header.get("R", 1.0)
        a_scaled = a_au / np.sqrt(star_luminosity_lsun)
        lower_r = 0.8 / np.sqrt(a_scaled)
        is_earth = (0.95 <= a_scaled < 1.67) and (lower_r <= radius_rearth < 1.4)
        if is_earth:
            earth_indices.append(i)

    return earth_indices


def from_exovista(
    fits_file: str,
    planet_indices: Optional[Sequence[int]] = None,
    required_planets: Optional[int] = None,
    only_earths: bool = False,
) -> System:
    """Load an ExoVista FITS file into a JAX-friendly :class:`System`.

    Args:
        fits_file: Path to ExoVista FITS file.
        planet_indices: Planet indices to load (0-based). ``None`` = all.
        required_planets: Pad/truncate to this many planets for fixed shapes.
        only_earths: If True and *planet_indices* is None, auto-filter Earths.

    Returns:
        :class:`System` with star, planets, and disk.
    """
    disk_ext = 2

    with open(fits_file, "rb") as f:
        h = getheader(f, ext=0, memmap=False)
    n_ext = h["N_EXT"]
    n_planets_total = n_ext - 4

    if planet_indices is None:
        if only_earths:
            planet_indices = get_earth_like_planet_indices(fits_file)
        else:
            planet_indices = list(range(n_planets_total))

    star = _load_star(fits_file, fits_ext=4)
    planet = _load_planets(
        fits_file, star, planet_indices, required_planets=required_planets
    )
    disk = _load_disk(fits_file, disk_ext, star)

    return System(star=star, planet=planet, disk=disk)
