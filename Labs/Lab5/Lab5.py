
# # Lab 5 ASTR 400B 
# 



# Import Modules 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy import constants as const # import astropy constants
import astropy.units as u


# # Part A :  Mass to Light Ratios 
# 
# Wolf et al. 2010 
# 
# $M(<R_{half}) = \frac {4}{G}\sigma^2 R_e$
# 
# Where $R_{half}$ = 3D half mass radius 
# and $R_e$ is the 2D half mass radius of stars (observed)
# 
# Determine which of the following two systems are galaxies:
# 
# The system 47 Tuc is observed with:  $\sigma = 17.3$ km/s, $R_e = 0.5$ pc, $L_v \sim 10^5 L_\odot$ 
# 
# The system Willman I is observed with: $\sigma = 4.3$ km/s, $R_e = 25$ pc, $L_v = 10^3 L_\odot$



# Gravitational Constant in the desired units
# kpc^3/Gyr^2/Msun
Grav = const.G.to(u.kpc**3/u.Gyr**2/u.Msun)




def WolfMass(sigma, re):
    """ Function that defines the Wolf mass estimator from Wolf+ 2010
    PARAMETERS
    ----------
        sigma: astropy quantity
            1D line of sight velocity dispersion in km/s
        re: astropy quantity
            Effective radius, 2D radius enclosing half the
            stellar mass in kpc
    OUTPUTS
    -------
        mWolf: Returns the dynamical mass within the 
            half light radius in Msun
    """
    
    sigmaKpcGyr = sigma.to(u.kpc/u.Gyr) # velocity dispersion units
    
    mWolf = 4/Grav*sigmaKpcGyr**2*re # Wolf mass estimator
    
    return mWolf

# 47 Tuc Parameters
lumTuc = 1e5*u.Lsun # Luminosity
sigmaTuc = 17.3*u.km/u.s # Velocity dispersion (1D)
reTuc = 0.5/1000*u.kpc # Effective radius (2D half light)

# Dynamical Mass of 47 Tuc
massTuc = WolfMass(sigmaTuc, reTuc)
print(f"The dynamical mass of 47 Tuc is {massTuc:.2e}")

# M/L ~ 1
print(f"The mass to light ratio of 47 Tuc is {massTuc/lumTuc:.2f}")
print('\n')
# Willman I Parameters
lumWill = 1e3*u.Lsun # Luminosity
sigmaWill = 4.3*u.km/u.s # Velocity dispersion (1D)
reWill = 25/1000*u.kpc # Effective radius (2D half light)

# Dynamical Mass of Willman I
massWill = WolfMass(sigmaWill, reWill)
print(f"The dynamical mass of Willman I is {massWill:.2e}")

# M/L
print(f"The mass to light ratio of Willman I is {massWill/lumWill:.2f}")


# # Part B :  Stellar to Halo Mass Relation
# 
# Following the work of [Moster et al. 2013 (MNRAS, 428, 3121)](https://ui.adsabs.harvard.edu/abs/2013MNRAS.428.3121M/abstract)
# 
# 
# `Equation 2:`                  $ \frac{m}{M} = 2N \left [ \left ( \frac{M}{M_1} \right)^{-\beta} + \left (\frac{M}{M_1} \right)^{\gamma} \right]$ 
# 
# $m$ = stellar mass, $M$ = halo mass
# 
# `Equation 11:`        log $M_1(z) = M_{10} + M_{11} \frac{z}{z+1} $ 
# 
# `Equation 12:`        $N(z) = N_{10} + N_{11} \frac{z}{z+1} $
# 
# `Equation 13:`         $\beta(z) = \beta_{10} + \beta_{11} \frac{z}{z+1} $
# 
# `Equation 14:`         $\gamma(z) = \gamma_{10} + \gamma_{11} \frac{z}{z+1} $

# # Q1 
# 
# Modify the class below by adding a function called `StellarMass` that uses the `SHMratio` function and returns the stellar mass.



class AbundanceMatching:
    """ Class to define the abundance matching relations from 
    Moster et al. 2013, which relate the stellar mass of a galaxy
    to the expected dark matter halo mass, according to 
    Lambda Cold Dark Matter (LCDM) theory """
    
    
    def __init__(self, mhalo, z):
        """ Initialize the class
        
        PARAMETERS
        ----------
            mhalo: float
                Halo mass in Msun
            z: float
                redshift
        """
        
        #initializing the parameters:
        self.mhalo = mhalo # Halo Mass in Msun
        self.z = z  # Redshift
        
        
    def logM1(self):
        """eq. 11 of Moster 2013
        OUTPUT: 
            M1: float 
                characteristic mass in log(Msun)
        """
        M10      = 11.59
        M11      = 1.195 
        return M10 + M11*(self.z/(1+self.z))  
    
    
    def N(self):
        """eq. 12 of Moster 2013
        OUTPUT: 
            Normalization for eq. 2
        """
        N10      = 0.0351
        N11      = -0.0247
    
        return N10 + N11*(self.z/(1+self.z))
    
    
    def Beta(self):
        """eq. 13 of Moster 2013
        OUTPUT:  power of the low mass slope"""
        beta10      = 1.376
        beta11      = -0.826
    
        return beta10 + beta11*(self.z/(1+self.z))
    
    def Gamma(self):
        """eq. 14 of Moster 2013
        OUTPUT: power of the high mass slope """
        gamma10      = 0.608
        gamma11      = 0.329
    
        return gamma10 + gamma11*(self.z/(1+self.z))
    
    
    def SHMratio(self):
        """ 
        eq. 2 of Moster + 2013
        The ratio of the stellar mass to the halo mass
        
        OUTPUT: 
            SHMratio float
                Stellar mass to halo mass ratio
        """
        M1 = 10**self.logM1() # Converting characteristic mass 
        # to Msun from Log(Msun)
        
        A = (self.mhalo/M1)**(-self.Beta())  # Low mass end
        
        B = (self.mhalo/M1)**(self.Gamma())   # High mass end
        
        Norm = 2*self.N() # Normalization
    
        SHMratio = Norm*(A+B)**(-1)
    
        return SHMratio 
    
 # Q1: add a function to the class that takes the SHM ratio and returns 
# The stellar mass 

    def StellarMass(self):
        """ Method to compute stellar mass using eq.2 of Moster+13 
         (stellar/halo mass ratio)

        OUTPUT:
            starMass: float
                Stellar mass in Msun
        """
        starMass = self.SHMratio()*self.mhalo
        return starMass


# # Part C : Plot the Moster Relation
# 
# Reproduce the below figure from Moster + 2013 
# Plot this for z=0, 0.5, 1, 2
# 
# ![mos](./MosterFig.png)



mh = np.logspace(10,15,1000) # Logarithmically spaced array


# Define Instances of the Class for each redshift
MosterZ0 = AbundanceMatching(mh,0)
MosterZ05 = AbundanceMatching(mh,0.5)
MosterZ1 = AbundanceMatching(mh,1)
MosterZ2 = AbundanceMatching(mh,2)

fig,ax = plt.subplots(figsize=(10,8))


#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# Plot z = 0
plt.plot(np.log10(mh), np.log10(MosterZ0.StellarMass()),
         linewidth = 5, label='z=0')

# Continue plotting for the other redshifts here
# Plot z = 0.5
plt.plot(np.log10(mh), np.log10(MosterZ05.StellarMass()),
         linewidth = 5, label='z=0.5', ls='-.')

# Plot z = 1
plt.plot(np.log10(mh), np.log10(MosterZ1.StellarMass()),
         linewidth = 5, label='z=1', ls='--')


# Plot z = 2
plt.plot(np.log10(mh), np.log10(MosterZ2.StellarMass()),
         linewidth = 5, label='z=2', ls=':')


# Axes labels 
plt.xlabel('log (M$_h$/M$_\odot$)',fontsize=22) 
plt.ylabel('log (m$_\star$/M$_\odot$)', fontsize=22)

# Legend
plt.legend(loc='lower right',fontsize='x-large')

# save the file 
plt.savefig('AbundanceMatching_Lab5.png')


# # Part D
# 
# # Q1
# 
# In studies that have modeled the Magellanic Clouds prior to 2010, the LMC is 
# traditioanlly modeled with a halo (dark matter) mass of order $3 \times 10^{10}$M$_\odot$.  
# 
# ## A) 
# According to $\Lambda$CDM theory, what should be the stellar mass of the LMC halo be at z=0?  
# 
# ## B) 
# How does this stellar mass compare to the actual observed stellar mass of the LMC at the present day of ~$3 \times 10^9$ M$_\odot$ ? 
# 
# ## C) 
# What is the $\Lambda$CDM expected halo mass for the LMC (using Abundance Matching)?  

# LMC Halo Mass
haloLMC1 = 3e10 # traditional halo mass (model)

# Abundance Matching 
LMC1 = AbundanceMatching(haloLMC1, 0)

#Stellar Mass
LMC1star = LMC1.StellarMass()

print(f"The stellar mass of the LMC halo at z=0 is {LMC1star:.2e}")
print(f"{LMC1star/3e9*100:.2f}% of the observed stellar mass of the LMC")

# say we know that LMC stellar mass = 3e9 Msun
# what the halo mass should be?
haloLMC2 = 17e10
LMC2 = AbundanceMatching(haloLMC2, 0)
LMC2star = LMC2.StellarMass()
print(f"The stellar mass of the LMC halo at z=0 is {LMC2star:.2e}")
print(f"{LMC2star/3e9*100:.2f}% of the observed stellar mass of the LMC")


# # Q2
# 
# ## A) 
# What is the expected stellar mass of an L* galaxy at z=0? 
# 
# ## B)
# What is the expected stellar mass of an L* galaxy at z = 2? 

# Find characteristic mass for z=0
M1halo_z0 = MosterZ0.logM1()
print(f"The characteristic log mass (M1) for z=0 is {M1halo_z0:.2f}")

# Find the stellar mass of an L* galaxy at z=0
Lstar_z0 = AbundanceMatching(10**M1halo_z0,0)
LstarMass_z0 = Lstar_z0.StellarMass()
print(f"The stellar log mass of an L* galaxy at z=0 is {np.log10(LstarMass_z0):.2f}")


# Find characteristic mass for z=2
M1halo_z2 = MosterZ2.logM1()
print(f"The characteristic log mass (M1) for z=2 is {M1halo_z2:.2f}")

# Find the stellar mass of an L* galaxy at z=2
Lstar_z2 = AbundanceMatching(10**M1halo_z2,2)
LstarMass_z2 = Lstar_z2.StellarMass()
print(f"The stellar log mass of an L* galaxy at z=2 is {np.log10(LstarMass_z2):.2f}")