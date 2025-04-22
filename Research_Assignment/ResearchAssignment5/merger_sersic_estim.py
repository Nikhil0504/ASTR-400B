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

from memory_profiler import profile


# Surface mass density profile

@profile
def SurfaceDensity(r,m, binsize=0.1, rmax=None):
    """ Function that computes the surface mass density profile
    given an array of particle masses and radii 
     
    PARMETERS
    ---------
        r : array of `floats` - cyclindrical radius [kpc]
        m : array of `floats` - particle masses [1e10 Msun] 
        binsize : `float` - size of the radial bins [kpc]
        rmax : `float` - maximum radius to consider [kpc]
    
    RETURNS
    -------
        r_annuli : array of `floats` -  radial bins for the 
            annuli that correspond to the surface mass density profile
    
        sigma: array of `floats` - surface mass density profile 
         [1e10 Msun/kpc^2] 
        
       
    """
    
    if rmax is None:
        rmax = 0.95 * np.max(r)

    # Set radial bins
    radii = np.arange(0.1, rmax, binsize)

    # Compute histogram of mass per bin
    m_annuli, edges = np.histogram(r, bins=radii, weights=m)

    # Compute area of each annulus
    r_inner = edges[:-1]
    r_outer = edges[1:]
    area = np.pi * (r_outer**2 - r_inner**2)

    sigma = m_annuli / area

    # Geometric mean radius for plotting
    r_annuli = np.sqrt(r_inner * r_outer)

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

def MassEnclosed(radii, r, masses):
    size = len(radii)
    total_mass = np.zeros(size)
    for i in range(size):
        # Find the indices of the particles within the radius
        indices = np.where(r <= radii[i])
        # Sum the masses of the particles within the radius
        total_mass[i] = np.sum(masses[indices])
    return total_mass

# Get center of mass object for M31 and MW for disk, bluge, and halo
# Concatinate the data for the bulge, disk, and halo
# Change the particle positions into cylindrical coordinates
M31_COM_bulge = CenterOfMass('../../Data/M31_469.txt', 3)
M31_COM_disk = CenterOfMass('../../Data/M31_469.txt', 2)

MW_COM_bulge = CenterOfMass('../../Data/MW_469.txt', 3)
MW_COM_disk = CenterOfMass('../../Data/MW_469.txt', 2)

# Use the center of mass object to 
# store the x, y, z, positions and mass of the bulge particles
# be sure to correct for the COM position of M31
M31_COM_p_bulge = M31_COM_bulge.COM_P(0.1)
M31_COM_p_disk = M31_COM_disk.COM_P(0.1)

MW_COM_p_bulge = MW_COM_bulge.COM_P(0.1)
MW_COM_p_disk = MW_COM_disk.COM_P(0.1)


# Subtract the center of mass position from the particle positions
x = np.concatenate((M31_COM_bulge.x - M31_COM_p_bulge[0].value,
                   M31_COM_disk.x - M31_COM_p_disk[0].value,
                   ))
y = np.concatenate((M31_COM_bulge.y - M31_COM_p_bulge[1].value,
                   M31_COM_disk.y - M31_COM_p_disk[1].value,
                   ))
z = np.concatenate((M31_COM_bulge.z - M31_COM_p_bulge[2].value,
                   M31_COM_disk.z - M31_COM_p_disk[2].value,
                   ))
m = np.concatenate((M31_COM_bulge.m,
                   M31_COM_disk.m,
                   )) # units of 1e10 Msun

# ADD MW DATA
x = np.concatenate((x, MW_COM_bulge.x - MW_COM_p_bulge[0].value,
                   MW_COM_disk.x - MW_COM_p_disk[0].value,
                   ))
y = np.concatenate((y, MW_COM_bulge.y - MW_COM_p_bulge[1].value,
                   MW_COM_disk.y - MW_COM_p_disk[1].value,

                     ))
z = np.concatenate((z, MW_COM_bulge.z - MW_COM_p_bulge[2].value,
                   MW_COM_disk.z - MW_COM_p_disk[2].value,

                ))
m = np.concatenate((m, MW_COM_bulge.m,
                     MW_COM_disk.m,

                     )) # units of 1e10 Msun


r = np.sqrt(x**2 + y**2 + z**2) # spherical
cyl_r = np.sqrt(x**2 + y**2) # radial
cyl_theta = np.arctan2(y,x) # theta


# Define surface mass density profile
r_annuli, sigma = SurfaceDensity(cyl_r, m)


# Get mass profile for M31
M31_mass = MassProfile('M31', 469)
MW_mass = MassProfile('MW', 469)

# Get the effective radius of the bulge particles
# Compute the total mass using Component Mass, 
# from the GalaxyMass code, and find the radius 
# that contains half this mass. 
# bulge_mass = M31_mass.massEnclosed(3, r_annuli).value + M31_mass.massEnclosed(2, r_annuli).value + MW_mass.massEnclosed(3, r_annuli).value + MW_mass.massEnclosed(2, r_annuli).value
bulge_mass = MassEnclosed(r_annuli, r, m)

# Determine the total mass of the merger
bulge_total = np.sum(m) * 1e10

# Find the effective radius of the bulge, 
# Re encloses half of the total bulge mass

# Half the total bulge mass
b_half = bulge_total / 2


# Find the indices where the bulge mass is larger than b_half
index = np.where(bulge_mass > b_half/1e10)
print(r_annuli.shape, index)

# take first index where Bulge Mass > b_half
# Define the Effective radius of the bulge
re_bulge = r_annuli[index][0] * 3/4

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
    bounds = ([0.1, 0.5, bulge_total*0.5], [180, 10, bulge_total*1.5])

    
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

ax.loglog(r_annuli, SersicM31Bulge/1e10, linewidth=2, label='Sersic Fit')

ax.loglog(r_annuli, SersicM31Bulge_fit/1e10, linewidth=2, label='Sersic Fit (Best Fit)')


plt.xlabel('log r [kpc]', fontsize=22)

# note the y axis units
plt.ylabel(r'log $\Sigma_{bulge}$ [$10^{10} M_\odot$ / kpc$^2$]', 
          fontsize=22)

#set axis limits
plt.xlim(1,50)
plt.ylim(1e-5,1)

ax.legend(loc='best', fontsize=22)
fig.tight_layout()

plt.show()



# Grid for re and n (fix mtot to mtot_fit to reduce computation)
re_vals = np.linspace(re_fit*0.5, re_fit*1.5, 50)
n_vals = np.linspace(n_fit*0.5, n_fit*1.5, 50)

# Initialize 2D chi-square array
chi2_grid = np.zeros((len(re_vals), len(n_vals)))

# Observed data
sigma_obs = sigma * 1e10  # convert to same units as sersicE

# Loop over all (re, n) combinations
for i, re_val in enumerate(re_vals):
    for j, n_val in enumerate(n_vals):
        # Model prediction
        sigma_model = sersicE(r_annuli, re_val, n_val, mtot_fit)
        
        # Compute chi-square
        chi2 = np.sum(((sigma_obs - sigma_model)**2) / sigma_model)
        
        chi2_grid[i, j] = chi2

# Plotting the chi-square map
fig2, ax2 = plt.subplots(figsize=(9, 7))
cs = ax2.contourf(n_vals, re_vals, np.log10(chi2_grid), levels=50, cmap='viridis')
cbar = plt.colorbar(cs)
cbar.set_label('log$_{10}$(Chi-Square)', fontsize=18)

ax2.set_xlabel('Sersic Index (n)', fontsize=18)
ax2.set_ylabel('Effective Radius (Re) [kpc]', fontsize=18)
plt.title('Chi-Square Surface for Sersic Fit', fontsize=20)

plt.tight_layout()
plt.show()
