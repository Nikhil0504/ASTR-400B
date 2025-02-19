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
