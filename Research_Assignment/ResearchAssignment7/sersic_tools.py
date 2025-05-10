import numpy as np

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


def chi_square(observed, expected):
    """ Function that computes the chi-square statistic
    for a given set of observed and expected values. 
    
    PARMETERS
    ---------
        observed: `array of floats`
            The observed values
        expected: `array of floats`
            The expected values

    RETURNS
    -------
        chi2: `float`
            The chi-square statistic
    """
    
    # Compute the chi-square statistic
    chi2 = np.abs(np.sum((observed - expected)**2 / expected))
    
    return chi2