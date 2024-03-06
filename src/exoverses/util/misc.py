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


def gen_rotate_to_sky_coords(
    vector, inclination, position_angle, convention="exovista"
):
    """
    Rotate from barycentric coordinates to plane of the sky, this is set up
    to match the exovista data

    Args:
        vec (np.array):
            Nx3 array of [x,y,z] vectors in barycentric coordinates
        inclination (astropy Quantity):
            Inclination of the system
        position_angle (astropy Quantity):
            Position angle of the system
        convention (str):
            Convention that describes the transormation to be applied

    Returns:
        vec (np.array):
            Nx3 array of vectors rotated to sky coordinates
    """
    if convention == "exovista":
        # Rotate around x axis with midplane inclination
        vector = rotate_vectors(vector, [1, 0, 0], -inclination)
        # Rotate around z axis with midplane position angle
        vector = rotate_vectors(vector, [0, 0, 1], position_angle)
        # Flip around z axis
        vector[:, 2] = -vector[:, 2]
    else:
        raise Exception(
            "Convention must be either 'exovista' or 'radvel', got {}".format(
                convention
            )
        )
    return vector


def gen_rotate_to_local_ecliptic_coords(
    vector, inclination, position_angle, convention="exovista"
):
    """
    Rotate from plane-of-the-sky coordinates to local ecliptic, this is set up
    to match the exovista data

    Args:
        vec (np.array):
            Nx3 array of [x,y,z] vectors in barycentric coordinates
        inclination (astropy Quantity):
            Inclination of the system
        position_angle (astropy Quantity):
            Position angle of the system
        convention (str):
            Convention that describes the transormation to be applied

    Returns:
        vec (np.array):
            Nx3 array of vectors rotated to sky coordinates
    """
    if convention == "exovista":
        # Flip around z axis
        vector[:, 2] = -vector[:, 2]

        # Rotate around z axis with midplane position angle
        vector = rotate_vectors(vector, [0, 0, 1], -position_angle)

        # Rotate around x axis with midplane inclination
        vector = rotate_vectors(vector, [1, 0, 0], inclination)

    else:
        raise Exception("No other conventions currently supported")
    return vector


# def drop_nan_coords(ds):
#     """
#     Removes coordinates from an xarray dataset that contain only NaN values.

#     This function iterates over all coordinates in the provided xarray dataset,
#     checks if each coordinate contains only NaN values, and removes those coordinates.
#     The check is performed using the isnull() method combined with the all() aggregation
#     function. Coordinates that are fully NaN are dropped from the dataset.

#     Args:
#         dataset (xr.Dataset):
#             The input xarray dataset from which NaN-only coordinates will be removed.

#     Returns:
#         xr.Dataset:
#             A new xarray dataset with NaN-only coordinates removed.
#     """
#     coords_to_drop = []
#     for coord in ds.coords:
#         coord_vals = ds[coord].values
#         # Check if all values in the coordinate are NaN
#         if ds[coord].isnull().all():
#             coords_to_drop.append(coord)

#     # Drop identified coordinates
#     for coord in coords_to_drop:
#         ds = ds.drop_vars(coord)

#     return ds


def drop_nan_coord_values(ds):
    """
    Drops coordinate values and their associated data if the data across all
    variables for those coordinate values are NaN.

    This function iterates over each coordinate in the dataset, checks each
    value of the coordinate to determine if all associated data across all
    variables are NaN, and if so, removes those coordinate values and their
    associated data.

    Args:
        ds (xr.Dataset):
            Input xarray dataset.

    Returns:
        xr.Dataset:
            Dataset with NaN-only coordinate values and their data removed.
    """
    for coord in list(ds.coords):
        unique_vals = ds[coord].values
        for val in unique_vals:
            # Mask for the current coordinate value across all data variables
            mask = ds.isel({coord: ds[coord] == val}).notnull()

            # Check if all data for this coordinate value are NaN across all variables
            if not mask.to_array().any():
                # Drop the coordinate value across all variables
                ds = ds.where(ds[coord] != val, drop=True)

    return ds


def add_units(
    ds, new_unit, vars=["x", "y", "z"], distance=None, pixel_scale=None, star_pixel=None
):
    """
    Add units to a dataset by adding a new data variable with the
    desired unit conversion.

    Args:
        ds (xarray.Dataset):
            The original dataset.
        new_unit (astropy.units.Unit):
            The target unit for conversion.
        vars (list of str):
            List of variable names to convert.
        distance (astropy.units.Quantity):
            Distance to system.
        pixel_scale (astropy.units.Quantity):
            Pixel scale of the data in angle/pixel units
        star_pixel (astropy.Quantity array):
            The [x,y] pixel where the star is located under the assumption
            that the star is at the center of the image
    """
    for var in vars:
        # Ensure the variable is in the dataset
        assert var in ds, f"Variable {var} not found in dataset."
        var_data = ds[var].copy()
        base_unit = var_data.unit
        base_data = var_data.data * base_unit
        # Handle different unit conversions
        if new_unit.physical_type == "length":
            converted_data = base_data.to(new_unit)
        elif new_unit.physical_type == "angle":
            assert (
                distance is not None
            ), "Distance to system not provided for angular conversion."
            converted_data = (
                np.arctan(base_data.to(u.m).value / distance.to(u.m).value) * u.rad
            ).to(new_unit)
        elif new_unit == "pixel":
            assert (
                (distance is not None)
                and (pixel_scale is not None)
                and (star_pixel is not None)
            ), "Distance to system and pixel scale must be provided."
            angular_data = (
                np.arctan(base_data.to(u.m).value / distance.to(u.m).value) * u.rad
            )
            converted_data = (angular_data / pixel_scale).to(new_unit) + star_pixel

        # Update the dataset with the converted data
        new_name = f"{var}({new_unit})"
        var_data.data = converted_data.value
        var_data.attrs["unit"] = new_unit

        ds[new_name] = var_data

    return ds
