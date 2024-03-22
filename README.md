# exo(planet uni)verses

A python library to create a (relatively) uniform set of objects between tools that generate planetary systems. Currently supports ExoVista and EXOSIMS universes.

# Install
### Base
`pip install exoverses`
### EXOSIMS support
`pip install exoverses[exosims]`
### ExoVista support
Currently this uses a fork of Exovista, clone it and install it to whatever environment you are using
`git clone git@github.com:CoreySpohn/ExoVista.git`
`cd ExoVista`
`git checkout pip_support`
Load whatever environment you are using (e.g. `source my_project/.venv/bin/activate` for a virtual environment or `conda activate my_project` for conda)
`pip install .`


# Usage
The basic objects are
- `Universe` object which holds all the `System` objects. Each `System` object has a `Star` object, a list of `Planet` objects, and a `Disk` object (if reading an ExoVista system).

## Planet object
The minimum definition of a planet is currently
```
"t0" - Initial epoch
"a" - Semi-major axis
"e" - Eccentricity
"mass" - Mass
"radius" - Radius
"inc" - Inclination
"W" - Longitude of the ascending node
"w" - Argument of periapsis
"M0" - Mean anomaly at initial epoch
"p" - Geometric albedo
```
A planet can be created independently with a dictionary of those values
```
from astropy.time import Time
import astropy.units as u
from exoverses.base.planet import Planet

planet_dict = {
"t0": Time(2000, format='decimalyear'),
"a": 1*u.au,
"e": 0.1,
"mass": 1*u.M_Earth,
"radius": 1*u.R_Earth,
"inc": 90*u.deg,
"W": 1*u.deg,
"w": 1*u.deg,
"M0": 0*u.deg,
"p": 0.3,
}
simple_planet = Planet(planet_dict)
```

## Exovista
```
from exoverses.exovista.universe import ExovistaUniverse

ev_universe_path = "path/to/exovista/fits/files/"
ev_universe = ExovistaUniverse(ev_universe_path)
ev_universe.systems[0]
> HIP -1	dist:10.0 pc	Type:G2V

Planets:
           K             T    secosw  sesinw                 T_c          a  ...         W    w         M0      t0      mass  radius
0   0.007479     87.969367  0.453459     0.0  2451478.3795736698   0.387099  ...  0.843030  0.0  203.98262  2000.0    0.0553   0.383
1   0.076703    224.235443  0.082300     0.0   2451423.336314871   0.722333  ...  1.338331  0.0  105.29904  2000.0    0.8150   0.949
2   0.077484    365.256410  0.129268     0.0   2451341.772314067   1.000000  ... -0.196535  0.0  111.72499  2000.0    1.0000   1.000
3   0.006867    686.959991  0.305634     0.0  2450809.4793391167   1.523662  ...  0.865309  0.0  305.87478  2000.0    0.1070   0.532
4  10.948237   4333.286713  0.219983     0.0  2446990.8777174116   5.203363  ...  1.755036  0.0  293.84823  2000.0  317.8300  11.209
5   2.448455  10756.199527  0.232703     0.0  2440189.9239725103   9.537070  ...  1.984702  0.0  296.22928  2000.0   95.1600   9.449
6   0.259398  30707.489457  0.217181     0.0   2423941.940415186  19.191264  ...  1.295556  0.0  239.00230  2000.0   14.5400   4.007

[7 rows x 14 columns]

# Alternatively you can create a System object independently
from exoverses.exovista.system import ExovistaSystem
ev_system_path = "path/to/exovista/fits/files/system0.fits"
ev_system = ExovistaSystem(ev_system_path)
ev_system
> HIP -1	dist:10.0 pc	Type:G2V

Planets:
           K             T    secosw  sesinw                 T_c          a  ...         W    w         M0      t0      mass  radius
0   0.007479     87.969367  0.453459     0.0  2451478.3795736698   0.387099  ...  0.843030  0.0  203.98262  2000.0    0.0553   0.383
1   0.076703    224.235443  0.082300     0.0   2451423.336314871   0.722333  ...  1.338331  0.0  105.29904  2000.0    0.8150   0.949
2   0.077484    365.256410  0.129268     0.0   2451341.772314067   1.000000  ... -0.196535  0.0  111.72499  2000.0    1.0000   1.000
3   0.006867    686.959991  0.305634     0.0  2450809.4793391167   1.523662  ...  0.865309  0.0  305.87478  2000.0    0.1070   0.532
4  10.948237   4333.286713  0.219983     0.0  2446990.8777174116   5.203363  ...  1.755036  0.0  293.84823  2000.0  317.8300  11.209
5   2.448455  10756.199527  0.232703     0.0  2440189.9239725103   9.537070  ...  1.984702  0.0  296.22928  2000.0   95.1600   9.449
6   0.259398  30707.489457  0.217181     0.0   2423941.940415186  19.191264  ...  1.295556  0.0  239.00230  2000.0   14.5400   4.007

[7 rows x 14 columns]
```

