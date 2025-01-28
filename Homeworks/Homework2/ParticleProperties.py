from ReadFile import Read

import numpy as np
import astropy.units as u


def ParticleInfo(filename, part_type, part_num):
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
    # Testing the function
    dist, vel, mass = ParticleInfo("../../MW_000.txt", 2, 100 - 1)
    print("Distance:", dist)
    print("Distance in lyr:", dist.to(u.lyr).round(3))
    print("Velocity:", vel)
    print("Mass:", mass)
