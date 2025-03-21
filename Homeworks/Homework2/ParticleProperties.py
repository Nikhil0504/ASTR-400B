from ReadFile import Read

import numpy as np
import astropy.units as u


def ParticleInfo(filename: str, part_type: int, part_num: int):
    """Returns the distance, velocity, and mass of a particle.

    Parameters
    ----------
    filename : str
        The name of the file to read.
    part_type : int
        The type of the particle.
    part_num : int
        The number of the particle.
    
    Returns
    -------
    dist : astropy.units.quantity.Quantity
        The distance of the particle. Rounded to 3 decimal places.
    vel : astropy.units.quantity.Quantity
        The velocity of the particle. Rounded to 3 decimal places.
    mass : astropy.units.quantity.Quantity
        The mass of the particle.
    """
    _, _, data = Read(filename)

    # Extracting the data for the given particle type
    index = np.where(data["type"] == part_type)
    data = data[index]

    # Extracting the data for the given particle number
    part_info = data[part_num]

    # Calculating the distance, velocity, and mass of the particle
    dist = (
        np.sqrt(part_info["x"] ** 2 + part_info["y"] ** 2 + part_info["z"] ** 2) * u.kpc
    )
    vel = np.sqrt(
        part_info["vx"] ** 2 + part_info["vy"] ** 2 + part_info["vz"] ** 2
    ) * (u.km / u.s)
    mass = part_info["m"] * 1e10 * u.M_sun

    # round the values to 3 decimal places
    dist = np.round(dist, 3)
    vel = np.round(vel, 3)

    return dist, vel, mass


if __name__ == "__main__":
    # File path
    fp = "../../MW_000.txt"
    # Testing the function
    dist, vel, mass = ParticleInfo(fp, 2, 100 - 1) # Particle 100 of type 2
    print("Distance:", dist) # distance in kpc
    print("Distance in lyr:", dist.to(u.lyr).round(3)) # distance in lyr, rounded to 3 decimal places
    print("Velocity:", vel) # velocity in km/s
    print("Mass:", mass) # mass in solar mass
