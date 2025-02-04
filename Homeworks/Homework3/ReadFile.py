import numpy as np
import astropy.units as u


def Read(filename: str):
    """Reads the data from a file and returns the time, total particles and the data.

    Parameters
    ----------
    filename : str
        The name of the file to read.

    Returns
    -------
    time : astropy.units.quantity.Quantity
        The time of the snapshot.
    total_particles : int
        The total number of particles in the snapshot.
    data : numpy.ndarray
        The data of the particles in the snapshot.
    """
    # open the file
    file = open(filename, "r")

    # read the first line
    line1 = file.readline()
    label, value = line1.split()
    time = float(value) * u.Myr

    # read the second line
    line2 = file.readline()
    label, value = line2.split()
    total_particles = int(value)

    # close the file
    file.close()

    # read the data
    data = np.genfromtxt(filename, dtype=None, names=True, skip_header=3)

    return time, total_particles, data


if __name__ == "__main__":
    # File path
    fp = "../../MW_000.txt"
    # Testing the function
    time, total_particles, data = Read(fp)
    print("Time:", time)
    print("Total Particles:", total_particles)

    # Test with second particle
    print(
        "Particle 2:",
        data["type"][1],  # type of particle
        data["m"][1],  # mass of particle
        data["x"][1],  # x position
        data["y"][1],  # y position
        data["z"][1],  # z position
        data["vx"][1],  # x velocity
        data["vy"][1],  # y velocity
        data["vz"][1],  # z velocity
    )
