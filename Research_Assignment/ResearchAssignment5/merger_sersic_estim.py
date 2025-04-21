# TOPIC: Investigating how well remnants of mergers can be described 
# as a classical elliptical galaxy based on its surface density profile
# by best fitting a sersic profile to the remnants of the merger

# Goal is to find the best fit sersic profile for the remnants of the merger
# and compare it to the best fit sersic profile for a classical elliptical galaxy
# and see how well they match and see if it will follow the de vaucouleurs law

# Surface density profile - sersic profile estimator 
# (heavily based on Lab 6 and Homework 5)

# STEPS:
# 1. Load the data for M31 and MW
# 2. Create a center of mass object for M31 and MW for disk, bulge, and halo
# 3. Concatenate the data for the bulge, disk, and halo
# 4. Change the particle positions into cylindrical coordinates
# 5. Define the surface mass density profile
# 6. Define the sersic profile estimator
# 7. Fit the sersic profile to the surface mass density profile
# 8. Plot the surface mass density profile and the sersic profile

# Load Modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import optimize

from ReadFile import Read
from CenterOfMass import CenterOfMass
from MassProfile import MassProfile
from GalaxyMass import ComponentMass


# Surface mass density profile

def SurfaceDensity(r,m):
    """ Function that computes the surface mass density profile
    given an array of particle masses and radii 
     
    PARMETERS
    ---------
        r : array of `floats` - cyclindrical radius [kpc]
        m : array of `floats` - particle masses [1e10 Msun] 
    
    RETURNS
    -------
        r_annuli : array of `floats` -  radial bins for the 
            annuli that correspond to the surface mass density profile
    
        sigma: array of `floats` - surface mass density profile 
         [1e10 Msun/kpc^2] 
        
       
    """
    
    # Create an array of radii that captures the extent of the bulge
    # 95% of max range of bulge
    radii = np.arange(0.1, 0.95 * r.max(), 0.1)

    # create a mask to select particles within each radius
    # np.newaxis creates a virtual axis to make cyl_r_mag 2 dimensional
    # so that all radii can be compared simultaneously
    # a way of avoiding a loop - returns a boolean 
    enc_mask = r[:, np.newaxis] < radii

    # calculate mass of bulge particles within each annulus.  
    # relevant particles will be selected by enc_mask (i.e., *1)
    # outer particles will be ignored (i.e., *0)
    # axis =0 flattens to 1D
    m_enc = np.sum(m[:, np.newaxis] * enc_mask, axis=0)

    # use the difference between m_enc at adjacent radii 
    # to get mass in each annulus
    m_annuli = np.diff(m_enc) # one element less then m_enc
    
    
    # Surface mass density of stars in the annulus
    # mass in annulus / surface area of the annulus. 
    # This is in units of 1e10
    sigma = m_annuli / (np.pi * (radii[1:]**2 - radii[:-1]**2))
    # array starts at 0, but here starting at 1 and
    # subtracting radius that ends one index earlier.
    
    # Define the range of annuli
    # here we choose the geometric mean between adjacent radii
    r_annuli = np.sqrt(radii[1:] * radii[:-1]) 

    return r_annuli, sigma


# Sersic profile estimator

# $I(r) = I_e exp^{-7.67 ( (r/R_e)^{1/n} - 1)}$
# $ L = 7.2 I_e \pi R_e^2$

def sersicE(r, re, n, mtot):
    """ Function that computes the Sersic Profile for an Elliptical 
    System, assuming M/L ~ 1. As such, this function is also the 
    mass surface density profile. 
    
    PARMETERS
    ---------
        r: `float`
            Distance from the center of the galaxy (kpc)
        re: `float`
            The Effective radius (2D radius that contains 
            half the light) (kpc)
        n:  `float`
            sersic index
        mtot: `float`
            the total stellar mass (Msun)

    RETURNS
    -------
        I: `array of floats`
            the surface brightness/mass density
            profile for an elliptical in Lsun/kpc^2

    """

    # M/L = 1
    lum = mtot

    # the effective surface brightness
    Ie = lum / (7.2 * np.pi * re**2)

    # breaking down a sersic profile
    a = (r/re)**(1/n)
    b = -7.67 * (a - 1)
    
    # surface brightness profile
    I = Ie * np.exp(b)
    
    # $ L = 7.2 I_e \pi R_e^2$
    return I

# Get center of mass object for M31 and MW for disk, bluge, and halo
# Concatinate the data for the bulge, disk, and halo
# Change the particle positions into cylindrical coordinates
M31_COM_bulge = CenterOfMass('M31_000.txt', 3)
M31_COM_disk = CenterOfMass('M31_000.txt', 2)
M31_COM_halo = CenterOfMass('M31_000.txt', 1)

# Use the center of mass object to 
# store the x, y, z, positions and mass of the bulge particles
# be sure to correct for the COM position of M31
M31_COM_p_bulge = M31_COM_bulge.COM_P(0.1)
M31_COM_p_disk = M31_COM_disk.COM_P(0.1)
M31_COM_p_halo = M31_COM_halo.COM_P(0.1)

x = M31_COM_bulge.x - M31_COM_p_bulge[0].value
y = M31_COM_bulge.y - M31_COM_p_bulge[1].value
z = M31_COM_bulge.z - M31_COM_p_bulge[2].value
m = M31_COM_bulge.m # units of 1e10 Msun

cyl_r = np.sqrt(x**2 + y**2) # radial
cyl_theta = np.arctan2(y,x) # theta


# Define surface mass density profile
r_annuli, sigma = SurfaceDensity(cyl_r, m)


# Get mass profile for M31
M31_mass = MassProfile('M31', 0)

# Get the effective radius of the bulge particles
# Compute the total mass using Component Mass, 
# from the GalaxyMass code, and find the radius 
# that contains half this mass. 
bulge_mass = M31_mass.massEnclosed(3, r_annuli).value

# Determine the total mass of the bulge
bulge_total = ComponentMass('M31_000.txt', 3) * 1e12

# Find the effective radius of the bulge, 
# Re encloses half of the total bulge mass

# Half the total bulge mass
b_half = bulge_total / 2


# Find the indices where the bulge mass is larger than b_half
index = np.where(bulge_mass > b_half)

# take first index where Bulge Mass > b_half
# Define the Effective radius of the bulge
re_bulge = r_annuli[index[0][0]] * 3/4

# d) Define the Sersic Profile for the M31 Bulge
# Sersic Index = 4
SersicM31Bulge = sersicE(r_annuli, re_bulge, 4, bulge_total)


# fit
def fit_sersic_profile(r_data, sigma_data, p0=None):
    """
    Fit a Sersic profile to surface density data
    
    Parameters:
    -----------
    r_data : array
        Radial distances (kpc)
    sigma_data : array
        Surface density values (10^10 Msun/kpc^2)
    p0 : tuple, optional
        Initial guesses for (re, n, mtot)
        
    Returns:
    --------
    popt : array
        Best-fit parameters [re, n, mtot]
    pcov : 2D array
        Covariance matrix for the parameters
    """
    # If no initial guesses provided, use reasonable defaults
    if p0 is None:
        # Starting with the values you already have as initial guesses
        p0 = [re_bulge, 4.0, bulge_total]
    
    # Convert sigma to absolute units (matching what sersicE returns)
    sigma_abs = sigma_data * 1e10  # Convert to Msun/kpc^2
    
    # Define bounds for parameters (all positive)
    bounds = ([0.1, 0.5, bulge_total*0.5], [30, 10, bulge_total*1.5])
    
    # Perform the fit
    # Note: we're fitting to the log of the data to handle the large dynamic range
    popt, pcov = optimize.curve_fit(
        sersicE, 
        r_data, 
        sigma_abs, 
        p0=p0,
        bounds=bounds,
    )
    
    return popt, pcov

init_guess = [re_bulge, 4.0, bulge_total]
popt, pcov = fit_sersic_profile(r_annuli, sigma, p0=init_guess)
# Extract the best-fit parameters
re_fit, n_fit, mtot_fit = popt
perr = np.sqrt(np.diag(pcov))
print(f"\nBest-fit Sersic Parameters:")
print(f"Effective Radius (re): {re_fit:.2f} kpc")
print(f"Sersic Index (n): {n_fit:.2f}")
print(f"Total Mass: {mtot_fit:.2e} Msun")
print(f"\nParameter Uncertainties:")
print(f"ΔRe: {perr[0]:.2f} kpc")
print(f"Δn: {perr[1]:.2f}")
print(f"ΔMtot: {perr[2]:.2e} Msun")

# calculate the Sersic profile with the best-fit parameters
SersicM31Bulge_fit = sersicE(r_annuli, re_fit, n_fit, mtot_fit)


fig, ax = plt.subplots(figsize=(9, 8))

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size


# Surface Density Profile
# YOU ADD HERE
ax.loglog(r_annuli, sigma, linewidth=2, label='Simulated Bulge')


# Sersic fit to the surface brightness Sersic fit
# YOU ADD HERE
ax.loglog(r_annuli, SersicM31Bulge/1e10, linewidth=2, label='Sersic Fit')

ax.loglog(r_annuli, SersicM31Bulge_fit/1e10, linewidth=2, label='Sersic Fit (Best Fit)')


plt.xlabel('log r [kpc]', fontsize=22)

# note the y axis units
plt.ylabel(r'log $\Sigma_{bulge}$ [$10^{10} M_\odot$ / kpc$^2$]', 
          fontsize=22)

plt.title('M31 Bulge', fontsize=22)

#set axis limits
plt.xlim(1,50)
plt.ylim(1e-5,0.1)

ax.legend(loc='best', fontsize=22)
fig.tight_layout()

plt.show()