from ReadFile import Read

import astropy.units as u
import numpy as np


def ComponentMass(filename: str, ptype: int):
    """
    Function that will compute the total mass of any given galaxy component.

    Parameters:
    -----------
    filename: str
        The name of the file that contains the data.
    ptype: int
        The particle type that we are interested in.
        1. Halo; 2. Disk; 3. Bulge

    Returns:
    --------
    mass: float
        The total mass of the galaxy component.
    """
    # Read the file
    time, total_particles, data = Read(filename)

    # Create a mask to select the particles of the desired type
    mask = data["type"] == ptype

    # Select the mass of the particles of the desired type
    mass = np.sum(data["m"][mask] * 1e10 * u.M_sun)

    # round the mass to 2 decimal places
    mass = np.round(mass, 2)

    return mass


# Test the function
if __name__ == "__main__":
    # Halo
    mass = ComponentMass("../../Data/MW_000.txt", 1)
    print(f"The total mass of the Halo is: {mass:.2e}")

    # Disk
    mass = ComponentMass("../../Data/MW_000.txt", 2)
    print(f"The total mass of the Disk is: {mass:.2e}")

    # Bulge
    mass = ComponentMass("../../Data/MW_000.txt", 3)
    print(f"The total mass of the Bulge is: {mass:.2e}")
