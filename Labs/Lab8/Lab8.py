
# # Lab 8 : Star Formation 




from re import S
import numpy as np
from astropy import units as u
from astropy import constants as const

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


# # Part A
# 
# Create a function that returns the SFR for a given luminosity (NUV, FUV, TIR, Halpha)
# 
# $Log( {\rm SFR} (M_\odot/year)) = Log(Lx (erg/s)) - Log(Cx)$ 
# 
# Including corrections for dust absorption 
# 
# Kennicutt & Evans 2012 ARA&A Equation 12 and Table 1, 2

def StarFormationRate(L, Type, TIR=0):
    """
    Calculate the star formation rate (SFR) based on the luminosity of a galaxy.

    Parameters
    ----------
    L : float
        Luminosity of the galaxy in erg/s.
    Type : str
        Type of luminosity ('NUV', 'FUV', 'TIR', 'Halpha').
    TIR : float, optional
        Total infrared luminosity in Lsun (default is 0).

    Returns
    -------
    SFR : float
        Star formation rate in Msun/year.
    """
    
    if Type == 'FUV':
        logCx = 43.35 # calibration from LFUV to SFR
        TIRc = 0.46 # correction factor for dust absorbtion
    elif Type == 'NUV':
        logCx = 43.17 # calibration from LNUV to SFR
        TIRc = 0.27 # correction factor for dust absorbtion
    elif Type == 'Halpha':
        logCx = 41.27
        TIRc = 0.0024
    elif Type == 'TIR':
        logCx = 43.41
        TIRc = 1
    else:
        print('Type not recognized. Expected either "FUV", "NUV", "TIR", "Halpha"')  
    
    # correct the luminosity for dust absorption
    Lcorr = L + TIRc*TIR

    #star formation rate
    SFR = np.log10(Lcorr.value) - logCx

    return SFR


# Let's try to reproduce SFRs derived for the WLM Dwarf Irregular Galaxy using UV luminosities measured with Galex. 
# 
# Compare results to Table 1 from Lee et al. 2009 (who used the older Kennicutt 98 methods)
# https://ui.adsabs.harvard.edu/abs/2009ApJ...706..599L/abstract
# 
# We will use galaxy properties from NED (Photometry and SED):
# https://ned.ipac.caltech.edu/



# First need the Luminosity of the Sun in the right units
print(const.L_sun)
LsunErgS = const.L_sun.to(u.erg/u.s)
print(LsunErgS)


#  WLM Dwarf Irregular Galaxy
NUV_WLM = 1.71e7 * LsunErgS #from NED GALEX data
TIR_WLM = 2.48e6 * LsunErgS + 3.21e5 * LsunErgS + 2.49e6* LsunErgS #from NED IRAS data
print(StarFormationRate(NUV_WLM, 'NUV', TIR_WLM))


# # Part B Star formation main sequence
# 
# 1) Write a function that returns the average SFR of a galaxy at a given redshift, given its stellar mass
# 
# 2) What is the average SFR of a MW mass galaxy today? at z=1?
# 
# 3) Plot the SFR main sequence for a few different redshifts from 1e9 to 1e12 Msun.
# 
# 
# From Whitaker 2012:
# 
# log(SFR) = $\alpha(z)({\rm log}M_\ast - 10.5) + \beta(z)$
# 
# $\alpha(z) = 0.7 - 0.13z$
# 
# $\beta(z) = 0.38 + 1.14z - 0.19z^2$

# # Step 1


def SFRMainSequece(Mstar, z):
    """function that returns the SFR for a given stellar mass of a galaxy
    following the main sequence of star forming galaxies.
    
    Parameters
    ----------
    Mstar: float, stellar mass of the galaxy in Msun
    z: float, redshift of the galaxy

    Returns
    -------
    SFR: float, log of star formation rate in Msun/yr
    """

    # fitting params
    alpha = 0.7 - 0.13 * z
    beta = 0.38 + 1.14 * z - 0.19 * z**2

    SFR = alpha * (np.log10(Mstar) - 10.5) + beta

    return SFR


# # Step 2



# MW at z=0
MW_mass = 7.5e10  # in Msun
# SFR for MW at z=0
SFR_z0 = SFRMainSequece(MW_mass, 0)
print(f"Average SFR for MW at z=0: {10**SFR_z0}")



# MW at z=1
MW_mass_z1 = 7.5e10  # in Msun
SFR_z1 = SFRMainSequece(MW_mass_z1, 1)
print(f"Average SFR for MW at z=1: {10**SFR_z1}")


# # Step 3



# create an array of stellar masses
stellar_masses = np.linspace(1e8, 1e12)




fig = plt.figure(figsize=(8,8), dpi=500)
ax = plt.subplot(111)

# add log log plots
plt.plot(np.log10(stellar_masses), SFRMainSequece(stellar_masses, 0), label='z=0', color='blue', lw=3)
plt.plot(np.log10(stellar_masses), SFRMainSequece(stellar_masses, 1), label='z=1', color='orange', lw=3)
plt.plot(np.log10(stellar_masses), SFRMainSequece(stellar_masses, 2), label='z=2', color='red', lw=3)
plt.plot(np.log10(stellar_masses), SFRMainSequece(stellar_masses, 3), label='z=3', color='pink', lw=3)

# Add axis labels
plt.xlabel('Log(Mstar (M$_\odot$))', fontsize=12)
plt.ylabel('Log(SFR (M$_\odot$/year))', fontsize=12)


#adjust tick label font size
label_size = 12
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')


# Save file
plt.savefig('Lab8_SFR_MainSequence.png')


# # Part C  Starbursts
# 
# Use your `StarFormationRate` code to determine the typical star formation rates for the following systems with the listed Total Infrared Luminosities (TIR): 
# 
# Normal Galaxies: $10^{10}$ L$_\odot$
# 
# LIRG: $10^{11}$ L$_\odot$
# 
# ULIRG: $10^{12} $ L$_\odot$
# 
# HLIRG: $10^{13} $ L$_\odot$



# normal galaxies 
SFR_normal = StarFormationRate(1e10 * LsunErgS, 'TIR')
print(f"Normal Galaxies SFR: {10**SFR_normal} Msun/year")

# LIRGs  
SFR_LIRG = StarFormationRate(1e11 * LsunErgS, 'TIR')
print(f"LIRG SFR: {10**SFR_LIRG} Msun/year")

# ULIRGs
SFR_ULIRG = StarFormationRate(1e12 * LsunErgS, 'TIR')
print(f"ULIRG SFR: {10**SFR_ULIRG} Msun/year")


# HLIRGs
SFR_HLIRG = StarFormationRate(1e13 * LsunErgS, 'TIR')
print(f"HLIRG SFR: {10**SFR_HLIRG} Msun/year")
