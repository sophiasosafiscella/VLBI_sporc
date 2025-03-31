import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord, ICRS, spherical_to_cartesian, cartesian_to_spherical, Angle
import astropy.units as u
from math import sqrt, sin, cos
from uncertainties.unumpy import uarray
import statsmodels.api as sm
import sys

def sec_to_deg(time_series: pd.Series) -> Angle:
    """Convert a pandas Series of seconds of time to an astropy Angle."""
    # Convert seconds of time to degrees (15 degrees per hour)
    degree_series = time_series * (15.0 / 3600)
    return Angle(degree_series, unit=u.deg)


def to_uarray(array):
    nominal_values = [array[i].nominal_value for i in range(len(array))]
    std_devs = [array[i].std_dev for i in range(len(array))]
    return uarray(nominal_values, std_devs)


def diff_pos(ra: float, dec: float):
    return  [[-sin(ra) * cos(dec), -1.0 * cos(ra) * sin(dec)], [cos(ra) * cos(dec), -1.0 * sin(ra) * sin(dec)], [0, cos(dec)]]


#---------------------------------------------
# Read the position of VLBI calibration source
#---------------------------------------------
path: str = './data/Wang_data/'
cal = pd.read_table(path + 'cal.txt', header=0, index_col=0,sep='\s+')

# Original positions
cal1_ra = uarray(Angle(cal["cal1_ra"], unit=u.hourangle).rad, Angle(sec_to_deg(cal["cal1_rae"]), unit=u.deg).rad)
cal1_dec = uarray(Angle(cal["cal1_dec"], unit=u.degree).rad, Angle(cal["cal1_dece"], unit=u.arcsec).rad)

# Positions in ICRF2
cal2_ra = uarray(Angle(cal["cal2_ra"], unit=u.hourangle).rad, Angle(sec_to_deg(cal["cal2_rae"]), unit=u.deg).rad)
cal2_dec = uarray(Angle(cal["cal2_dec"], unit=u.degree).rad, Angle(cal["cal2_dece"], unit=u.arcsec).rad)

#---------------------------------------------
# Read in the pulsar VLBI positions in ICRF 1
#---------------------------------------------
icrf_pos = pd.read_table(path + 'msp_vlbi.txt', header=0, index_col=0, sep='\s+')
psr_names = icrf_pos.index.tolist()
N_pulsars: int = len(psr_names)

ra_v = uarray(Angle(icrf_pos["ra_v"], unit=u.hourangle).rad, Angle(sec_to_deg(icrf_pos["ra_ve"]), unit=u.deg).rad)  # RA from VLBI
dec_v = uarray(Angle(icrf_pos["dec_v"], unit=u.degree).rad, Angle(icrf_pos["dec_ve"], unit=u.arcsec).rad)           # DEC from VLBI

# Convert the VLBI pulsar positions from ICRF1 to ICRF 2
dcal_ra = cal2_ra - cal1_ra
dcal_dec = cal2_dec - cal1_dec

ra_v2 = to_uarray(ra_v + dcal_ra)     # RA from VLBI in ICRF2
dec_v2 = to_uarray(dec_v + dcal_dec)  # DEC from VLBI in ICRF2

# change the error bar of 0437
for i in range(N_pulsars):
   if icrf_pos.index.tolist()[i] == 'J0437-4715':
       # "In this case, we summed the quoted differential uncertainty and the uncertainty in the calibrator source in quadrature"
       ra_v2[i].std_dev = sqrt(ra_v[i].std_dev**2 + cal2_ra[i].std_dev**2 + Angle(0.8 * u.mas).rad**2)
       dec_v2[i].std_dev = sqrt(dec_v[i].std_dev**2 + cal2_dec[i].std_dev**2)
   else:
       ra_v2[i].std_dev = ra_v[i].std_dev       # "We did not correct the published uncertainties to those in ICRF2"
       dec_v2[i].std_dev = dec_v[i].std_dev     # IMPORTANT: I'M VERY SUS OF THIS PART

# Turn the VLBI positions into dictionaries
ra0 = {k: v for k, v in zip(psr_names, ra_v2)}    # This seems to agree with Wang's
dec0 = {k: v for k, v in zip(psr_names, dec_v2)}  # This seems to agree with Wang's

#---------------------------------------------
# Read in timing positions
#---------------------------------------------
timing_pos = pd.read_table(path + 'msp_timing.txt', header=0, index_col=0, sep='\s+')
psr_t_names = timing_pos.index.tolist()
psr_list = timing_pos.index.unique().tolist()
cor = Angle(timing_pos['cor'], unit=u.mas)

# Make sure that the epochs are consistent between VLBI and timing
for i in range(N_pulsars):
    for j in range(len(psr_t_names)):
        if timing_pos.index.tolist()[j] == icrf_pos.index.tolist()[i]:
            if timing_pos['epoch_t'].iloc[j] != icrf_pos['epoch_v'].iloc[i]:
                print('WARNING vlbi and timing epochs are not consistent')

for j, ephem in enumerate(timing_pos['ephem'].unique()):
    B = np.zeros((3 * N_pulsars, 3))
    D = np.zeros((3 * N_pulsars, 2 * N_pulsars))
    Ct, Cv = np.zeros((2 * N_pulsars, 2 * N_pulsars)), np.zeros((2 * N_pulsars, 2 * N_pulsars))

    rat_hms= timing_pos.loc[timing_pos['ephem'] == ephem, 'ra_t']
    rat_dict = {k: v for k, v in zip(rat_hms.index.tolist(), Angle(rat_hms.values, unit=u.hourangle).rad)}
    rat_diff = Angle([rat_dict[psr] - ra0[psr].nominal_value for psr in psr_list], unit=u.rad).to(u.mas).value

    dect_dms = timing_pos.loc[timing_pos['ephem'] == ephem, 'dec_t']
    dect_dict = {k: v for k, v in zip(dect_dms.index.tolist(), Angle(dect_dms.values, unit=u.degree).rad)}
    dect_diff = Angle([dect_dict[psr] - dec0[psr].nominal_value for psr in psr_list], unit=u.rad).to(u.mas).value

    rat_err_sec = timing_pos.loc[timing_pos['ephem'] == ephem, 'ra_te']
    rat_err = {k: v for k, v in zip(rat_err_sec.index.tolist(), Angle(sec_to_deg(rat_err_sec.values), unit=u.degree).to(u.mas).value)}

    dect_err_sec = timing_pos.loc[timing_pos['ephem'] == ephem, 'dec_te']
    dect_err = {k: v for k, v in zip(dect_err_sec.index.tolist(), Angle(dect_err_sec.values, unit=u.arcsec).to(u.mas).value)}

    cor = timing_pos.loc[timing_pos['ephem'] == ephem, 'cor']
    rdcorc = {k: v for k, v in zip(cor.index.tolist(), cor.values)}

    for i, psr in enumerate(psr_names):
        nhat = spherical_to_cartesian(1.0, dec0[psr].nominal_value, ra0[psr].nominal_value)
        B[3 * i: 3 * (i + 1),:] = np.matrix([[0,-nhat[2],nhat[1]], [nhat[2],0,-nhat[0]], [-nhat[1],nhat[0],0]])
        D[3 * i:3 * (i + 1), 2 * i:2 * (i + 1)] = diff_pos(ra0[psr].nominal_value, dec0[psr].nominal_value)

        # This seems to be some sort of covariance matrix. I think it's equivalent to Sigma from Dusty's paper,
        # because it includes rdcorc, which is the "dimensionless normalized cross-correlation between RA and Dec."
        Ct[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.matrix([[rat_err[psr]**2, rat_err[psr] * dect_err[psr] * rdcorc[psr]],
                                                             [rat_err[psr] * dect_err[psr] * rdcorc[psr], dect_err[psr]**2]])

        # VLBI covariance matrix?
        Cv[2 * i:2 * (i + 1), 2 * i:2 * (i + 1)] = np.diag([Angle(ra0[psr].std_dev, unit=u.rad).to(u.mas).value**2,
                                                            Angle(dec0[psr].std_dev, unit=u.rad).to(u.mas).value**2])

    # now set up the LSQ dEq = M A + eps, where M = (Dt*D)^-1*Dt*B
    cov_matrix = Ct + Cv                                                                # This agrees with Wang's
    M = np.linalg.multi_dot([np.linalg.inv(np.matmul(D.T, D)), D.T, B])                 # This agrees with Wang's
    data = np.array([(rat_diff[i], dect_diff[i]) for i in range(N_pulsars)]).flatten()  # Data is not the same as Wang's

    gls_model = sm.GLS(data, M, sigma=cov_matrix).fit()
    print(gls_model.summary())
    sys.exit()
