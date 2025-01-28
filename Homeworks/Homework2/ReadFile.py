import numpy as np
import astropy.units as u


def Read(filename):
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
    # Testing the function
    time, total_particles, data = Read("../../MW_000.txt")
    print("Time:", time)
    print("Total Particles:", total_particles)

    # Test with second particle
    print(
        "Particle 2:",
        data["type"][1],
        data["m"][1],
        data["x"][1],
        data["y"][1],
        data["z"][1],
        data["vx"][1],
        data["vy"][1],
        data["vz"][1],
    )
