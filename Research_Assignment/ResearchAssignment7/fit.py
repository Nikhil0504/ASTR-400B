# Load Modules
import numpy as np
from scipy.optimize import curve_fit

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib

# my modules
from CenterOfMass import CenterOfMass
from sersic_tools import SurfaceDensity, sersicE, MassEnclosed, chi_square


# Create a center of mass object for M31 and MW
# I.e. an instance of the CenterOfMass class 
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

# get all the particle positions and masses
x = np.concatenate((
    M31_COM_bulge.x - M31_COM_p_bulge[0].value,
    M31_COM_disk.x - M31_COM_p_disk[0].value,
    MW_COM_bulge.x - MW_COM_p_bulge[0].value,
    MW_COM_disk.x - MW_COM_p_disk[0].value
))
y = np.concatenate((
    M31_COM_bulge.y - M31_COM_p_bulge[1].value,
    M31_COM_disk.y - M31_COM_p_disk[1].value,
    MW_COM_bulge.y - MW_COM_p_bulge[1].value,
    MW_COM_disk.y - MW_COM_p_disk[1].value
))
z = np.concatenate((
    M31_COM_bulge.z - M31_COM_p_bulge[2].value,
    M31_COM_disk.z - M31_COM_p_disk[2].value,
    MW_COM_bulge.z - MW_COM_p_bulge[2].value,
    MW_COM_disk.z - MW_COM_p_disk[2].value
))
m = np.concatenate((
    M31_COM_bulge.m,
    M31_COM_disk.m,
    MW_COM_bulge.m,
    MW_COM_disk.m
)) # units of 1e10 Msun


# Determine the positions of the bulge particles in cylindrical coordinates.
cyl_r = np.sqrt(x**2 + y**2) # radial
cyl_theta = np.arctan2(y,x) # theta
cyl_z = z # vertical


# Define the surface mass density profile for the simulated bulge 
# and the corresponding annuli
r_annuli, sigmaM31_MW = SurfaceDensity(cyl_r, m) # sigma in units of 1e10 Msun/kpc^2

# limit r_annuli to 0.1 kpc to 100 kpc
mask = (r_annuli >= 0.1) & (r_annuli < 80)
r_annuli = r_annuli[mask]
sigmaM31_MW = sigmaM31_MW[mask]

# Compute the total mass using Component Mass
M31_MW_mass = MassEnclosed(r_annuli, cyl_r, m)
M31_MW_mass_total = np.sum(m) * 1e10


b_half = M31_MW_mass_total / 2
print(f"M31 Mass = {b_half:.2e} Msun")

# Find the indices where the bulge mass is larger than b_half
index = np.where(M31_MW_mass > b_half/1e10)

# take first index where Bulge Mass > b_half
# check : should match b_half
print(f"M31 Mass = {M31_MW_mass[index][0]*1e10:.2e} Msun")
# Define the Effective radius of the bulge
re_bulge = r_annuli[index[0][0]] * 3/4
print(f"Effective Radius = {re_bulge:.2f} kpc")


# Sersic Index = 4
SersicM31Bulge = sersicE(r_annuli, re_bulge, 4, M31_MW_mass_total)

# fit for sersic index
def fit_sersic(r, n):
    return np.log10(sersicE(r, re_bulge, n, M31_MW_mass_total)/1e10)

# get the initial guess for the sersic index and fit
popt, pcov = curve_fit(fit_sersic, r_annuli, np.log10(sigmaM31_MW), p0=[6], bounds=(0, 10))
fitted_n = popt[0]
print(f"Fitted Sersic Index = {fitted_n:.2f}")

# get sersic profile for the fitted sersic index
SersicM31_MW_fit = sersicE(r_annuli, re_bulge, fitted_n, M31_MW_mass_total)


fig, ax = plt.subplots(figsize=(9, 8))

label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size


# Surface Density Profile
ax.loglog(r_annuli, sigmaM31_MW, linewidth=2, label='Simulated')


# Sersic fit to the surface brightness Sersic fit
# ax.loglog(r_annuli, SersicM31Bulge/1e10, linewidth=2, label='Sersic Fit')
ax.loglog(r_annuli, SersicM31_MW_fit/1e10,
         linewidth=2, linestyle='--', label='Fitted Sersic Profile n=%.2f' % fitted_n)


# labels
plt.xlabel('log r [ kpc]', fontsize=22)

plt.ylabel(r'log $\Sigma_{bulge}$ [$10^{10} M_\odot$ / kpc$^2$]', 
          fontsize=22)

plt.title('M31+MW', fontsize=22)

# #set axis limits
plt.xlim(1,100)
plt.ylim(1e-5,0.1)

ax.legend(loc='best', fontsize=22)
fig.tight_layout()

plt.savefig('M31_MW_Sersic.png', dpi=300)
plt.show()



# get the chi square
n_values = np.linspace(1, 10, 100)
chi_square_fits = np.zeros_like(n_values)
for i, n in enumerate(n_values):
    SersicM31_MW_fit = sersicE(r_annuli, re_bulge, n, M31_MW_mass_total)/1e10
    chi_square_fits[i] = chi_square(np.log10(sigmaM31_MW), np.log10(SersicM31_MW_fit))


# plot the chi square
fig, ax = plt.subplots(figsize=(9, 8))

ax.plot(n_values, chi_square_fits, linewidth=2)
ax.axvline(fitted_n, color='red', linestyle='--', label='Fitted n=%.2f' % fitted_n)
ax.axhline(0, color='black', linestyle='--')

plt.xlabel('Sersic Index n', fontsize=22)
plt.ylabel(r'$\chi^2$', fontsize=22)

plt.ylim(0, 1e2)

ax.legend(loc='best', fontsize=22)
fig.tight_layout()

plt.savefig('M31_MW_ChiSquare.png', dpi=300)
plt.show()