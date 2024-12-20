import numpy as np
import pandas as pd
from astropy.coordinates import Angle
from numpy.linalg import inv, multi_dot

from astropy.coordinates import SkyCoord  # High-level coordinates
from astropy.coordinates import ICRS, Galactic, FK4, FK5  # Low-level frames
from astropy.time import Time
import astropy.units as u

from uncertainties import ufloat
from uncertainties.umath import *

import sys

def error_propagation(ra, dec):

    # Convert to radians and calculate the cartesian componentes
    x = cos(dec) * cos(ra)
    y = cos(dec) * sin(ra)
    z = sin(dec)

    return np.array([x.std_dev, y.std_dev, z.std_dev])

'''
def error_propagation(ra, dec, ra_err, dec_err):
    # Convert to angles
    ra = Angle(ra).radian
    dec = Angle(dec).radian
    ra_err = Angle(ra_err).radian
    dec_err = Angle(dec_err).radian

    # Use error propagation
    x_err = np.sqrt(
        np.sin(dec) ** 2 * np.cos(ra) ** 2 * dec_err ** 2 + np.cos(dec) ** 2 * np.sin(ra) ** 2 * ra_err ** 2)
    y_err = np.sqrt(
        np.sin(dec) ** 2 * np.sin(ra) ** 2 * dec_err ** 2 + np.cos(dec) ** 2 * np.cos(ra) ** 2 * ra_err ** 2)
    z_err = np.cos(dec) * dec_err

    return np.array([x_err, y_err, z_err])
'''

# Load the VLBI_data
data = pd.read_csv("./data/frame_tie.csv", index_col=0)
PSR_names = ["J0437-4715", "J1713+0747"]
N_pulsars: int = len(PSR_names)

# Define the matrix
D_T = np.empty((1, 3 * N_pulsars))
E_T = np.empty((1, 3 * N_pulsars))
M_T = np.empty((3, 3 * N_pulsars))

# Iterate over the pulsars
for i, PSR in enumerate(PSR_names):

    # Create SkyCoord objects to handle frame, proper motion, etc.
    timing_skycoords = SkyCoord(ra=data.loc[f"{PSR}_timing", "RAJ"], dec=data.loc[f"{PSR}_timing", "DECJ"],
                             frame=ICRS, unit=(u.hourangle, u.deg),
                             equinox=Time(val=data.loc[f"{PSR}_timing", "epoch"], format='mjd', scale='utc'))

    timing_skycoords_err = SkyCoord(ra=data.loc[f"{PSR}_timing", "RAJ_err"], dec=data.loc[f"{PSR}_timing", "DECJ_err"],
                                frame=ICRS, unit=(u.hourangle, u.deg),
                                equinox=Time(val=data.loc[f"{PSR}_timing", "epoch"], format='mjd', scale='utc'))

    # Create uncertainty objects to handle error propagation
    timing_RA = ufloat(timing_skycoords.ra.rad, timing_skycoords_err.ra.rad)
    timing_DEC = ufloat(timing_skycoords.dec.rad, timing_skycoords_err.dec.rad)

    # Do the error propagation automatically
    cartesian_timing_errs = error_propagation(timing_RA, timing_DEC)

    VLBI_skycoords = SkyCoord(ra=data.loc[f"{PSR}_VLBI", "RAJ"], dec=data.loc[f"{PSR}_VLBI", "DECJ"],
                           frame=ICRS, unit=(u.hourangle, u.deg),
                           equinox=Time(val=data.loc[f"{PSR}_VLBI", "epoch"], format='mjd', scale='utc'))

    VLBI_skycoords_err = SkyCoord(ra=data.loc[f"{PSR}_VLBI", "RAJ_err"], dec=data.loc[f"{PSR}_VLBI", "DECJ_err"],
                           frame=ICRS, unit=(u.hourangle, u.deg),
                           equinox=Time(val=data.loc[f"{PSR}_VLBI", "epoch"], format='mjd', scale='utc'))

    # Create uncertainty objects to handle error propagation
    VLBI_RA = ufloat(VLBI_skycoords.ra.rad, VLBI_skycoords_err.ra.rad)
    VLBI_DEC = ufloat(VLBI_skycoords.dec.rad, VLBI_skycoords_err.dec.rad)

    # Do the error propagation automatically
    cartesian_VLBI_errs = error_propagation(VLBI_RA, VLBI_DEC)

    # Calculate the matrices
    D_T[0, 3 * i:3 * (i + 1)] = (timing_skycoords.cartesian - VLBI_skycoords.cartesian).get_xyz()

    E_T[0, 3 * i:3 * (i + 1)] = cartesian_timing_errs - cartesian_VLBI_errs

    x, y, z = VLBI_skycoords.cartesian.get_xyz()
    M_T[:, 3 * i:3 * (i + 1)] = np.transpose(np.matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]]))

# Do the calculations
E = np.transpose(E_T)
D = np.transpose(D_T)
M = np.transpose(M_T)

sigma = np.dot(E, E_T)
sigma = np.diag(np.diag(sigma))  # We have approximated sigma as diagonal
sigma_inv = np.linalg.inv(sigma)

# Do Equation 12 computation
cov = multi_dot([M_T, sigma_inv, M])
Ahat = multi_dot([inv(cov), M_T, sigma_inv, D])

Ax, Ay, Az = Angle(Ahat[0, 0], u.radian), Angle(Ahat[1, 0], u.radian), Angle(Ahat[2, 0], u.radian)

np.save("./results/frame_tie.npy", np.array([Ax.value, Ay.value, Az.value]))

print(Ax.value, Ay.value, Az.value)

print(Ax.arcsec*1000)
print(Ay.arcsec*1000)
print(Az.arcsec*1000)
