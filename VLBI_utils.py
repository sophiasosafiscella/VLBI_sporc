import astropy.units as u
import numpy as np
import pandas as pd
import scipy
from astropy.coordinates import Angle
from pandas.core.frame import pandas
from pint.models.timing_model import TimingModel
from scipy.stats import norm, skewnorm
from uncertainties import ufloat, umath
import sys

def parSkewNormal(x0, uL, uR, pX=0.5, pL=0.025, pR=0.975, wX=1, wL=1, wR=1):
    ## INPUTS
    ## x  = Measured value   : x is the 100*pX percentile
    ## VLBI_uL = Left uncertainty : x - VLBI_uL is the 100*pL percentile
    ## VLBI_uR = Right uncertainty: x + VLBI_uR is the 100*pR percentile
    ## wX, wL, wR = Weights for the errors made when attempting to
    ## reproduce x, x-VLBI_uL, and x+VLBI_uR as percentiles of a skew-normal
    ## distribution
    ## OUTPUT
    ## Vector with the values of xi, omega, and alpha for the best
    ## fitting skew-normal distribution

    # xi : vector of location parameters.
    # omega : vector of scale parameters; must be positive.
    # alpha : vector of slant parameter(s); +/- Inf is allowed. For psn, it must be of length 1 if engine="T.Owen". For qsn, it must be of length 1.

    if any(np.array([wX, wL, wR]) < 0):
        raise ValueError("ERROR in parSkewNormal: Weights wL, wX, and wR must all be positive")
    if not ((pL < pX) and (pX < pR)):
        raise ValueError("ERROR in parSkewNormal: Probabilities must be such that pL < pX < pR")

    def fSkewNormal(theta):
        loc, scale, a = theta
        return sum(np.array([wL, wX, wR]) * (
                skewnorm.ppf([pL, pX, pR], loc=loc, scale=scale, a=a) - np.array([x0 - uL, x0, x0 + uR])) ** 2)

    try:
        if abs(pR - pL) < 0.75:
            initial_guess = [x0, (uL + uR) / 2, 2]  # Initial guesses of the parameters of the skew-normal distribution
        else:
            initial_guess = [x0, (uL + uR) / 4, 2]

        res = scipy.optimize.minimize(fSkewNormal, initial_guess, method='Nelder-Mead')
        theta = res.x  # Value of parameters of the skew-normal distribution with which it attains a minimum
        return dict(zip(['loc', 'scale', 'a'], theta))
    except:
        raise ValueError("Optimization failed")


def plot_pdf(x0, uL, uR, num: int = 1000):
    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        x = np.linspace(x0 - 3.5 * uL, x0 + 3.5 * uR, num)
        y = norm.pdf(x, loc=x0, scale=uL)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        x = np.linspace(res['loc'] - 4 * res['scale'], res['loc'] + 4 * res['scale'], num)
        y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

    return x, y


def pdf_value(x, x0, uL, uR):
    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        y = norm.pdf(x, loc=x0, scale=uL)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

    return y


def draw_samples(x0, uL, uR, size=1000):
    # If the error bars are equal, we have a normal distribution
    if uL == uR:
        samples = norm.rvs(loc=x0, scale=uL, size=size)

    # If the error bars are not equal, we have a skew-normal distribution
    if uL != uR:
        res = parSkewNormal(x0=x0, uL=uL, uR=uR)
        samples = skewnorm.rvs(a=res['a'], loc=res['loc'], scale=res['scale'], size=size)

    return samples


def calculate_prior(timing_model, VLBI_data_file, PSR_name: str) -> float:
    VLBI_data = pd.read_csv(VLBI_data_file, index_col=0)

    # ------------------------------Parallax------------------------------
    PX_prior = pdf_value(x=timing_model.PX.quantity.value, x0=VLBI_data.loc[PSR_name, "VLBI_PX"],
                         uL=VLBI_data.loc[PSR_name, "VLBI_PX_uL"], uR=VLBI_data.loc[PSR_name, "VLBI_PX_uR"])

    # ------------------------------Proper Motion------------------------------
    VLBI_DECJ = ufloat(Angle(VLBI_data.loc[PSR_name, "VLBI_DECJ"]).rad,
                       Angle(VLBI_data.loc[PSR_name, "VLBI_DECJ_err"]).rad)

    # For VLBI, sometimes the error bars are asymmetric. In order to propagate errors, we will do this twice, each time
    # assuming a symmetric error equal to either VLBI_uL or VLBI_uR:
    for error_side in ["uL", "uR"]:
        VLBI_PMRA = ufloat(VLBI_data.loc[PSR_name, "VLBI_PMRA"], VLBI_data.loc[PSR_name, "VLBI_PMRA_" + error_side])
        VLBI_PMDEC = ufloat(VLBI_data.loc[PSR_name, "VLBI_PMDEC"], VLBI_data.loc[PSR_name, "VLBI_PMDEC_" + error_side])
        VLBI_PM = umath.sqrt(VLBI_PMDEC ** 2 + VLBI_PMRA ** 2 * (umath.cos(VLBI_DECJ) ** 2))

        if error_side == "uL":
            VLBI_PM_uL = VLBI_PM.std_dev
        elif error_side == "uR":
            VLBI_PM_uR = VLBI_PM.std_dev

    # Calculate the total proper motion from the timing model
    timing_PMRA = ufloat(timing_model.PMRA.value, timing_model.PMRA.uncertainty.value)
    timing_PMDEC = ufloat(timing_model.PMDEC.value, timing_model.PMDEC.uncertainty.value)
    timing_DECJ = ufloat(Angle(timing_model.DECJ.quantity).rad, Angle(timing_model.DECJ.uncertainty).rad)
    timing_PM = umath.sqrt(timing_PMDEC ** 2 + timing_PMRA ** 2 * (umath.cos(timing_DECJ) ** 2))

    # Calculate the prior for the timing value of the PM, given the PDF from the VLBI values
    PM_prior = pdf_value(x=timing_PM.nominal_value, x0=VLBI_PM.nominal_value, uL=VLBI_PM_uL, uR=VLBI_PM_uR)

    # Calculate the joint probability distribution by multiplying the PDFs
    return np.outer(PX_prior, PM_prior)


def replace_params(timing_model: TimingModel, timing_solution: pandas) -> TimingModel:
    # We build a dictionary with a key for each parameter we want to set.
    # The dictionary entries can be either
    #  {'pulsar name': (parameter value, TEMPO_Fit_flag, uncertainty)} akin to a TEMPO par file form
    # or
    # {'pulsar name': (parameter value, )} for parameters that can't be fit
    params = {
        "PMRA": (timing_solution.PMRA, 1, 0.0 * u.mas / u.yr),
        "PMDEC": (timing_solution.PMDEC, 1, 0.0 * u.mas / u.yr),
        "PX": (timing_solution.PX, 1, 0.0 * u.mas)
    }

    # Assign the new parameters
    for name, info in params.items():
        par = getattr(timing_model, name)  # Get parameter object from name
        par.value = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = True  # Frozen means not fit.
            par.uncertainty = info[2]

    # Set up and validate the new model
    timing_model.setup()
    timing_model.validate()

    return timing_model

def add_noise_params(tm: TimingModel, EFAC, EQUAD) -> TimingModel:

    # Add the EFAC, EQUAD components
    from pint.models.noise_model import ScaleToaError
    tm.add_component(ScaleToaError(), validate=False)

    # Add parameter values
    params = {
        "EFAC1": (EFAC, 1, 0),
        "EQUAD1": (EQUAD, 1, 0),
    }

    # Assign the parameters
    for name, info in params.items():
        par = getattr(tm, name)  # Get parameter object from name
        par.quantity = info[0]  # set parameter value
        if len(info) > 1:
            if info[1] == 1:
                par.frozen = False  # Frozen means not fit.
            par.uncertainty = info[2]

    # Set up and validate the model
    tm.setup()
    tm.validate()

    return tm


def unfreeze_noise(mo, verbose=False):
    """
    Unfreeze noise parameters in place in preparation for PINT noise

    Parameters
    ==========
    mo: PINT timing model

    Returns
    =======
    None
    """

    EFAC_EQUAD_components = mo.components['ScaleToaError']
    ECORR_components = mo.components['EcorrNoise']
    # Get the EFAC and EQUAD keys. Ignore TNEQ
    EFAC_keys = EFAC_EQUAD_components.EFACs.keys()
    EQUAD_keys = EFAC_EQUAD_components.EQUADs.keys()
    # Get the ECORR keys
    ECORR_keys = ECORR_components.ECORRs.keys()

    # Iterate over each set and mark unfrozen
    for key in EFAC_keys:
        param = getattr(EFAC_EQUAD_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False
#        param.quantity += 4.0
    for key in EQUAD_keys:
        param = getattr(EFAC_EQUAD_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False
    for key in ECORR_keys:
        param = getattr(ECORR_components, key)
        if verbose:
            print("Unfreezing", key)
        param.frozen = False

    # Unfreeze red noise if present
    # if 'PLRedNoise' in mo.components.keys():
    #    mo.components['PLRedNoise'].RNAMP.frozen = False
    #    mo.components['PLRedNoise'].RNIDX.frozen = False

# x: float = 1.17
# VLBI_uL: float = 0.05
# VLBI_uR: float = 0.04

# res = parSkewNormal(x=x, VLBI_uL=VLBI_uL, VLBI_uR=VLBI_uR)

# x = np.linspace(1.0, 1.5, 1000)
# y = skewnorm.pdf(x, a=res['a'], loc=res['loc'], scale=res['scale'])

# sns.set_style("darkgrid")

# plt.plot(x, y)
# plt.title("SkewNormal PDF for $\Pi=1.17^{+0.04}_{-0.05}$")
# plt.xlabel('$\Pi$')
# plt.ylabel('Probability (unnormalized)')
# plt.show()
