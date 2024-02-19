import os
from glob import glob
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.constants import c, h

import functools
import contextlib
import warnings

from MeanStars import MeanStars

# constant
mirror_diameter = 0.48 * u.m

# paths
# PACKAGEDIR = "." # for testing only
PHOENIXPATH = f"{PACKAGEDIR}/data/phoenix/"



def phoenixcontext():
    """
    Decorator that temporarily sets the `PYSYN_CDBS` environment variable.

    Parameters
    ----------
    phoenixpath : str
        The value to temporarily set for the `PYSYN_CDBS` environment variable.

    Returns
    -------
    function
        A wrapper function that sets `PYSYN_CDBS` to `phoenixpath` before
        executing the decorated function and restores the original environment
        afterwards.

    Examples
    --------
    Using `set_pysyn_cdbs` to temporarily set `PYSYN_CDBS` for a function:

    >>> @set_pysyn_cdbs()
    ... def my_function():
    ...     # Within this function, os.environ["PYSYN_CDBS"] is set
    ...
    >>> my_function()
    >>> 'PYSYN_CDBS' in os.environ
    False
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with modified_environ(PYSYN_CDBS=PHOENIXPATH):
                return func(*args, **kwargs)

        return wrapper

    return decorator

@contextlib.contextmanager
def modified_environ(**update):
    """
    Temporarily updates the `os.environ` dictionary in-place and restores it upon exit.
    """
    env = os.environ
    original_state = env.copy()

    # Apply updates to the environment
    env.update(update)

    try:
        yield
    finally:
        # Restore original environment
        env.clear()
        env.update(original_state)

@phoenixcontext()
def get_phoenix_model(teff, logg=4.5, jmag=None, vmag=None):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Extinction files not found in "
        )
        # Third-party
        import pysynphot
#     print(os.environ["PYSYN_CDBS"])
    logg1 = logg.value if isinstance(logg, u.Quantity) else logg
    star = pysynphot.Icat(
        "phoenix",
        teff.value if isinstance(teff, u.Quantity) else teff,
        0,
        logg1,
    )
    if (jmag is not None) & (vmag is None):
        star_norm = star.renorm(
            jmag, "vegamag", pysynphot.ObsBandpass("johnson,j")
        )
    elif (jmag is None) & (vmag is not None):
        star_norm = star.renorm(
            vmag, "vegamag", pysynphot.ObsBandpass("johnson,V")
        )
    else:
        raise ValueError("Input one of either `jmag` or `vmag`")
    star_norm.convert("Micron")
    star_norm.convert("flam")
    mask = (star_norm.wave >= 0.1) * (star_norm.wave <= 3)
    wavelength = star_norm.wave[mask] * u.micron
    wavelength = wavelength.to(u.angstrom)

    sed = star_norm.flux[mask] * u.erg / u.s / u.cm**2 / u.angstrom
    return wavelength, sed

def calculate_logg_from_teff(teff, s_radius=None):
    """
    this calculation works for dwarf stars.
    calculates logg from teff based on
    "A Modern Mean Dwarf Stellar Color and Effective Temperature Sequence"
    http://www.pas.rochester.edu/~emamajek/EEM_dwarf_UBVIJHK_colors_Teff.txt
    Eric Mamajek
    """
    s_mass = MeanStars().TeffOther("Msun", teff)
    if s_radius is None:
        s_radius = MeanStars().TeffOther("R_Rsun", teff)

    logg = np.log10(s_mass) - 2 * np.log10(s_radius) + 4.437
    return logg

def get_throughput(wavelength):
    throughput = pd.read_csv(f"{PACKAGEDIR}/data/ultrasat_bandpass_center.csv")
    return np.interp(wavelength, throughput.wavelength_nm * u.nm,
                     throughput.throughput, left=0, right=0)

def photon_energy(wavelength):
    """Converts photon wavelength to energy."""
    return ((h * c) / wavelength) * 1 / u.photon

def sensitivity(wavelength):
    sed = (3.631e-20 * u.erg / u.s / u.cm**2 / u.Hz).to(
        u.erg / u.s / u.cm**2 / u.AA,
        equivalencies=u.spectral_density(midpoint()))
    E = photon_energy(wavelength)
    telescope_area = np.pi * (mirror_diameter / 2) ** 2
    photon_flux_density = (
        telescope_area * sed * get_throughput(wavelength) / E
    ).to(u.photon / u.second / u.angstrom)
    sensitivity = photon_flux_density / sed
    return sensitivity

def midpoint():
    """Mid point of the sensitivity function"""
    w = np.arange(2000, 3500, 0.005) * u.AA
    return np.average(w, weights=get_throughput(w))

def magnitude_to_ULTRASAT(teff, logg=None, jmag=None, vmag=None):
    
    teff = teff.value if isinstance(teff, u.Quantity) else teff
    
    if logg is None:
        logg = calculate_logg_from_teff(teff)
        
    wavelength, sed = get_phoenix_model(teff, logg=logg, jmag=jmag, vmag=vmag)
    sens = sensitivity(wavelength)
    
    fd = np.trapz(sens * sed, wavelength) / np.trapz(sens, wavelength)
    mag_uv = -2.5 * np.log10(fd.to(u.erg / u.cm**2 / u.s / u.Hz,
                               equivalencies=u.spectral_density(midpoint())).value)  - 48.594
    return mag_uv
    
def get_transit_SNR(mag, duration, depth):
    # depth must be unscaled depth
    
    SNR = pd.read_csv(f"{PACKAGEDIR}/data/ultrasat_snr_G_5deg.csv")
    
    if isinstance(duration, u.Quantity):
        duration = duration.to(u.hour)
    else:
        raise ValueError("duration requires a unit")
        
    depth = depth * u.dimensionless_unscaled 

    ultrasat_snr = np.interp(mag, SNR.mag, SNR.SNR, left=0, right=0)
    ultrasat_1sigma = (1 / ultrasat_snr)
    
    transit_snr = (depth / ultrasat_1sigma) * np.sqrt(duration/ (900 * u.s).to(u.hour))
    return transit_snr
