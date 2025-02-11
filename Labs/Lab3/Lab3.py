
#In Class Lab 3 Template
# G Besla ASTR 400B

# Load Modules
from fileinput import filename
import numpy as np
import astropy.units as u

# import plotting modules
import matplotlib.pyplot as plt
import matplotlib


# The Figure illustrates the color magnitude diagram (CMD) for the Carina Dwarf along with the interpreted 
# star formation history from isochrone fitting to the CMD.
# The image is from Tolstoy+2009 ARA&A 47 review paper about dwarf galaxies
# 
# ![Iso](./Lab3_Isochrones.png)
# 

# # This Lab:
# 
# Modify the template file of your choice to plot isochrones that correspond to the inferred star formation episodes (right panel of Figure 1) to recreate the dominant features of the CMD of Carina (left panel of Figure 1). 



# Some Notes about the Isochrone Data
# DATA From   http://stellar.dartmouth.edu/models/isolf_new.html
# files have been modified from download.  ( M/Mo --> M;   Log L/Lo --> L)
# removed #'s from all lines except column heading
# NOTE SETTINGS USED:  Y = 0.245 default   [Fe/H] = -2.0  alpha/Fe = -0.2
# These could all be changed and it would generate a different isochrone




# Filename for data with Isochrone fit for 1 Gyr
# These files are located in the folder IsochroneData
filename1="./IsochroneData/Isochrone1.txt"

# major peak
filename10 = './IsochroneData/Isochrone10.txt'

# next major peak
filename7 = './IsochroneData/Isochrone7.txt'

# smallest peak
filename4 = './IsochroneData/Isochrone13.txt'



# READ IN DATA
# "dtype=None" means line is split using white spaces
# "skip_header=8"  skipping the first 8 lines 
# the flag "names=True" creates arrays to store the date
#       with the column headers given in line 8 

# Read in data for an isochrone corresponding to 1 Gyr
data1 = np.genfromtxt(filename1,dtype=None,names=True,skip_header=8)

# Read in data for an isochrone corresponding to 10 Gyr
data10 = np.genfromtxt(filename10,dtype=None,names=True,skip_header=8)

# Read in data for an isochrone corresponding to 7 Gyr
data7 = np.genfromtxt(filename7,dtype=None,names=True,skip_header=8)

# Read in data for an isochrone corresponding to 13 Gyr
data4 = np.genfromtxt(filename4,dtype=None,names=True,skip_header=8)



# Plot Isochrones 
# For Carina

fig = plt.figure(figsize=(10,10))
ax = plt.subplot(111)

# Plot Isochrones

# Isochrone for 1 Gyr
# Plotting Color vs. Difference in Color 
plt.plot(data1['B']-data1['R'], data1['R'], color='blue', linewidth=5, label='1 Gyr')
###EDIT Here, following the same format as the line above 

# Isochrone for 10 Gyr
plt.plot(data10['B']-data10['R'], data10['R'], color='red', linewidth=5, label='10 Gyr')

# Isochrone for 7 Gyr
plt.plot(data7['B']-data7['R'], data7['R'], color='green', linewidth=5, label='7 Gyr')

# Isochrone for 13 Gyr
plt.plot(data4['B']-data4['R'], data4['R'], color='purple', linewidth=5, label='13 Gyr')


# Add axis labels
plt.xlabel('B-R', fontsize=22)
plt.ylabel('M$_R$', fontsize=22)

#set axis limits
plt.xlim(-0.5,1.6)
plt.ylim(5,-2.5)

#adjust tick label font size
label_size = 22
matplotlib.rcParams['xtick.labelsize'] = label_size 
matplotlib.rcParams['ytick.labelsize'] = label_size

# add a legend with some customizations.
legend = ax.legend(loc='upper left',fontsize='x-large')

#add figure text
plt.figtext(0.6, 0.15, 'CMD for Carina dSph', fontsize=22)
plt.tight_layout()

plt.savefig('IsochroneCarina.png')



# # Q2
# 
# Could there be younger ages than suggested in the Tolstoy plot?
# Try adding younger isochrones to the above plot.

# # Q3
# 
# What do you think might cause the bursts of star formation?

# Bursts of star formation occur due to galaxy mergers that give fresh
# gas to the galaxy, causing it to form stars. Bar instabilities can also
# cause star formation. Feedback from supernovae can also cause star formation.



