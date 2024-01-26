import astropy.units as u
import numpy as np
from astropy import constants as c
from scipy.optimize import root
from scipy.spatial.transform import Rotation as R

"""
Functions from radvel, should either rework or go back to calling radvel
but it wouldn't install and I don't want to keep debugging
"""


def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    from radvel
    Convert Time of Transit to Time of Periastron Passage
    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)
    Returns:
        float: time of periastron passage
    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi / 2 - omega
    ee = 2 * np.arctan(
        np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
    )  # eccentric anomaly
    tp = tc - per / (2 * np.pi) * (ee - ecc * np.sin(ee))  # time of periastron

    return tp


def timeperi_to_timetrans(tp, per, ecc, omega, secondary=False):
    """
    from radvel
    Convert Time of Periastron to Time of Transit
    Args:
        tp (float): time of periastron
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): argument of peri (radians)
        secondary (bool): calculate time of secondary eclipse instead
    Returns:
        float: time of inferior conjunction (time of transit if system is transiting)
    """
    try:
        if ecc >= 1:
            return tp
    except ValueError:
        pass

    if secondary:
        f = 3 * np.pi / 2 - omega  # true anomaly during secondary eclipse
        ee = 2 * np.arctan(
            np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
        )  # eccentric anomaly

        # ensure that ee is between 0 and 2*pi (always the eclipse AFTER tp)
        if isinstance(ee, np.float64):
            ee = ee + 2 * np.pi
        else:
            ee[ee < 0.0] = ee + 2 * np.pi
    else:
        f = np.pi / 2 - omega  # true anomaly during transit
        ee = 2 * np.arctan(
            np.tan(f / 2) * np.sqrt((1 - ecc) / (1 + ecc))
        )  # eccentric anomaly

    tc = tp + per / (2 * np.pi) * (ee - ecc * np.sin(ee))  # time of conjunction

    return tc


def Msini(K, P, Mstar, e, Msini_units="earth"):
    """Calculate Msini
    from radvel
    Calculate Msini for a given K, P, stellar mass, and e
    Args:
        K (float or array: Doppler semi-amplitude [m/s]
        P (float or array): Orbital period [days]
        Mstar (float or array): Mass of star [Msun]
        e (float or array): eccentricity
        Msini_units (Optional[str]): Units of Msini {'earth','jupiter'}
            default: 'earth'
    Returns:
        float or array: Msini [units = Msini_units]
    """

    # convert inputs to array so they work with units
    P = np.array(P)
    Mstar = np.array(Mstar)
    K = np.array(K)
    e = np.array(e)
    G = c.G.value  # added gravitational constant
    Mjup = c.M_jup.value  # added Jupiter's mass
    Msun = c.M_sun.value  # added sun's mass
    Mstar = Mstar * Msun
    Mstar = np.array(Mstar)

    P_year = (P * u.d).to(u.year).value
    P = (P * u.d).to(u.second).value

    # First assume that Mp << Mstar
    K_0 = 28.4329
    Msini = (
        K
        / K_0
        * np.sqrt(1.0 - e**2.0)
        * (Mstar / Msun) ** (2.0 / 3.0)
        * P_year ** (1 / 3.0)
    )

    # Use correct calculation if any elements are >10% of the stellar mass
    if (np.array(((Msini * u.Mjup).to(u.M_sun) / (Mstar / Msun)).value > 0.10)).any():

        a = K * (((2 * (np.pi) * G) / P) ** (-1 / 3.0)) * np.sqrt(1 - (e**2))
        Msini = []
        if isinstance(P, float):
            n_elements = 1
        else:
            assert (
                type(K) == type(P) == type(Mstar) == type(e)
            ), "All input data types must match."
            assert (
                K.size == P.size == Mstar.size == e.size
            ), "All input arrays must have the same length."
            n_elements = len(P)
        for i in range(n_elements):

            def func(x):
                try:
                    return x - a[i] * ((Mstar[i] + x) ** (2 / 3.0))
                except IndexError:
                    return x - a * ((Mstar + x) ** (2 / 3.0))

            sol = root(func, Mjup)
            Msini.append(sol.x[0])

        Msini = np.array(Msini)
        Msini = Msini / Mjup

    if Msini_units.lower() == "jupiter":
        pass
    elif Msini_units.lower() == "earth":
        Msini = (Msini * u.M_jup).to(u.M_earth).value
    else:
        raise Exception("Msini_units must be 'earth', or 'jupiter'")

    return Msini


def rotate_vectors(vectors, axis, angle):
    """
    Rotates a set of Nx3 vectors around a single axis by a single angle
    Args:
        vectors (np.array):
            Nx3 array of vectors
        axis (list):
            3-element array specifying rotation axis (e.g. [0,0,1]
            for z-axis)
        angle (u.Quantity):
            Angle to rotate vectors by
    """
    rot = R.from_rotvec(np.array(axis) * angle.to(u.rad).value)
    return rot.apply(vectors)
