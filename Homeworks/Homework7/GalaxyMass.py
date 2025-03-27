import numpy as np 
import pandas as pd 
from ReadFile import Read
import astropy.units as u
from tabulate import tabulate

def ComponentMass(filename, ptype):
    """
    Calculate the total mass of a specific component of a galaxy.

    Parameters
    ----------
    filename : str
        The name of the file containing the data.
    ptype : str
        The type of particle to calculate the mass for (e.g., 'disk', 'bulge', 'halo').
    
    Returns
    -------
    final_mass : float
        The total mass of the specified component in 10^12 solar masses.
        
        """
    # Read in the file
    _, _, data = Read(filename)

    # finding all the indices in the data corresponding to particle type
    index = np.where(data["type"] == ptype)  

    mass = data["m"][index] / 100

    # total masses of the particles to get the total mass of the galaxy component.
    total_mass = np.round(np.sum(mass), 3)  

    final_mass = float(total_mass)
    return final_mass
