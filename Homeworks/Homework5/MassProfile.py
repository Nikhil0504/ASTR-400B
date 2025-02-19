from ReadFile import Read
from CenterOfMass import CenterOfMass

import astropy.units as u
import numpy as np

from astropy.constants import G
G = G.to(u.kpc*u.km**2/u.s**2/u.Msun)


class MassProfile:
    def __init__(self, galaxy, snap):
        """ Initialize the class

        Parameters
        ----------
            galaxy: str
                Name of the galaxy
            snap: int
                Snapshot number
        """
        self.gname = galaxy
        self.snap = snap

        # add a string of the filenumber to the value “000”
        ilbl = '000' + str(self.snap)
        # remove all but the last 3 digits
        ilbl = ilbl[-3:]
        self.filename = "%s_"%(self.gname) + ilbl + '.txt'

        # Read the data
        self.data = self.read_data()

        # extract x, y, z, m
        self.x = self.data['x'] * u.kpc
        self.y = self.data['y'] * u.kpc
        self.z = self.data['z'] * u.kpc

        self.m = self.data['m']
    
    def read_data(self):
        """ Read the data from the file

        Returns
        -------
            data: array
                Array of the data
        """
        time, tot_part, data = Read(self.filename)
        return data

    def MassEnclosed(self, ptype, r):
        """Calculate the mass enclosed within a given radius
        of the COM position for a specified galaxy and it's component.

        Parameters
        ----------
        ptype : float
            Particle type for which to calculate the mass enclosed
        r : np.ndarray
            Array of radii to calculate the mass enclosed within
        
        Returns
        -------
        mass_enclosed : astropy.units.Quantity
            Array of mass enclosed within the given radii, in solMass
        """

        # only getting the disk particles
        CoM_disk_part = CenterOfMass(self.filename, 2) 
        CoM_P_disk = CoM_disk_part.COM_P() # center of mass (disk)

        # create an array to store the mass enclosed
        mass_encolosed = np.zeros(r.size)

        # loop over the radii to indentify the mass enclosed within each radius
        index = np.where(self.data['type'] == ptype)
        radii = np.sqrt(
            (self.x[index] - CoM_P_disk[0])**2 +
            (self.y[index] - CoM_P_disk[1])**2 +
            (self.z[index] - CoM_P_disk[2])**2
        )
        mass_type = self.m[index] # only of a particular type

        for i in range(len(r)):
            # sum the mass of all particles within the radius
            # multiply by 1e10 for the mass to be correct
            mass_encolosed[i] = np.sum(mass_type[radii < r[i] * u.kpc]) * 1e10
        
        mass_encolosed = mass_encolosed * u.Msun

        return mass_encolosed

    def MassEnclosedTotal(self, r):
        """Calculate the total mass enclosed within a given radius
        of the COM position for a specified galaxy.

        Parameters
        ----------
        r : np.ndarray
            Array of radii to calculate the mass enclosed within

        Returns
        -------
        mass_enclosed : astropy.units.Quantity
            Array of mass enclosed within the given radii, in solMass
        """

        # create an array to store the mass enclosed
        mass_encolosed = np.zeros(r.size) * u.Msun

        for i_type in range(1, 4):
            # no bulge for M33
            if self.gname == 'M33' and i_type == 3:
                continue
            mass_encolosed += self.MassEnclosed(i_type, r)

        return mass_encolosed
    
    def HernquistMass(self, r, a, Mhalo):
        """Calculate the Hernquist mass profile.

        Parameters
        ----------
        r : np.ndarray
            Array of radii to calculate the mass enclosed within
        a : float
            Scale radius of the Hernquist profile
        Mhalo : float
            Total halo mass

        Returns
        -------
        mass_profile : astropy.units.Quantity
            Array of mass enclosed within the given radii, in solMass
        """
        # convert to the correct units
        r = r * u.kpc
        a = a * u.kpc
        numerator = Mhalo * 1e12 * u.Msun * r**2
        denominator = (a + r)**2

        mass_profile = numerator / denominator

        return mass_profile
    
    def CircularVelocity(self, ptype, r):
        """Calculate the circular velocity for a specified galaxy and it's component.
        within a given radius of the COM position.

        Parameters
        ----------
        ptype : int
            The type of particle.
        r : np.ndarray
            Radii to calculate the circular velocity at.
        
        Returns
        -------
        circular_vel: astropy.units.Quantity
            Array of circular speeds, units km/s
        """
        mass_enc = self.MassEnclosed(ptype, r) # mass enclosed within the radius

        # v_circular = sqrt(G * M / r)
        circular_vel = np.sqrt(G * mass_enc / r)

        # convert to km/s
        circular_vel = circular_vel.to(u.km/u.s)

        # round to 2 decimal places
        circular_vel = np.round(circular_vel, 2)

        return circular_vel

    def CircularVelocityTotal(self, r):
        """
        Calculates the total circular velocity of a galaxy. Within a given
        radius of the COM postion.

        Parameters
        ----------
        r : np.ndarray
            Radii to calculate the circular velocity at.
        
        Returns
        -------
        tot_cir_vel: astropy.units.Quantity
            Array of total circular velocity, units km/s
        """
        tot_cir_vel = np.zeros(r.size) 
        
        # convert units
        r *= u.kpc

        total_mass_enc = self.MassEnclosedTotal(r)

        total_cir_vel = np.sqrt(G * total_mass_enc / r).to(u.km/u.s)

        return np.around(total_cir_vel.value, 2) * u.km/u.s


    def HernquistVCirc(self, r, a, Mhalo):
        """Calculate the circular velocity for a Hernquist profile.

        Parameters
        ----------
        r : np.ndarray
            Array of radii to calculate the circular velocity at
        a : float
            Scale radius of the Hernquist profile
        Mhalo : float
            Total halo mass

        Returns
        -------
        circular_vel : astropy.units.Quantity
            Array of circular speeds, in km/s
        """
        # get hernquist mass within the radius
        hern_mass = self.HernquistMass(r, a, Mhalo)

        # convert to the correct units
        r = r * u.kpc
        a = a * u.kpc

        # v_circular = sqrt(G * M / r)
        circular_vel = np.sqrt(G * hern_mass / r * (a / (a + r)))

        # convert to km/s
        circular_vel = circular_vel.to(u.km/u.s)

        # round to 2 decimal places
        circular_vel = np.round(circular_vel, 2) * u.km/u.s

        return circular_vel