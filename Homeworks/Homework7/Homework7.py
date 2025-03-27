
# # Homework 7 Template
# 
# Rixin Li & G . Besla
# 
# Make edits where instructed - look for "****", which indicates where you need to 
# add code. 




# import necessary modules
# numpy provides powerful multi-dimensional arrays to hold and manipulate data
import numpy as np
# matplotlib provides powerful functions for plotting figures
import matplotlib.pyplot as plt
# astropy provides unit system and constants for astronomical calculations
import astropy.units as u
import astropy.constants as const
# import Latex module so we can display the results with symbols
from IPython.display import Latex

# **** import CenterOfMass to determine the COM pos/vel of M33
import CenterOfMass2 as COM

# **** import the GalaxyMass to determine the mass of M31 for each component
import GalaxyMass as GM

# # M33AnalyticOrbit
class M33AnalyticOrbit:
    """ Calculate the analytical orbit of M33 around M31 """
    
    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : str
            The name of the output file to save the orbit data.
        """
        

        ### get the gravitational constant (the value is 4.498502151575286e-06)
        self.G = const.G.to(u.kpc**3/u.Msun/u.Gyr**2).value
        
        ### **** store the output file name
        self.filename = filename
        M33_fp = '../../Data/M33_000.txt'
        M31_fp = '../../Data/M31_000.txt'
        
        ### get the current pos/vel of M33 
        # **** create an instance of the  CenterOfMass class for M33 
        M33_COM = COM.CenterOfMass(M33_fp, 2)

        # **** store the position VECTOR of the M33 COM (.value to get rid of units)
        self.posM33 = M33_COM.COM_P(0.1,4).value

        # **** store the velocity VECTOR of the M33 COM (.value to get rid of units)
        self.velM33 =  M33_COM.COM_V(self.posM33[0]*u.kpc,self.posM33[1]*u.kpc, self.posM33[2]*u.kpc).value
        
        
        ### get the current pos/vel of M31 
        # **** create an instance of the  CenterOfMass class for M31 
        M31_COM = COM.CenterOfMass(M31_fp, 2)

        # **** store the position VECTOR of the M31 COM (.value to get rid of units)
        self.posM31 = M31_COM.COM_P(0.1,2).value

        # **** store the velocity VECTOR of the M31 COM (.value to get rid of units)
        self.velM31 = M31_COM.COM_V(self.posM31[0]*u.kpc, self.posM31[1]*u.kpc, self.posM31[2]*u.kpc).value
        
        
        ### store the DIFFERENCE between the vectors posM33 - posM31
        # **** create two VECTORs self.r0 and self.v0 and have them be the
        # relative position and velocity VECTORS of M33
        self.r = self.posM33 - self.posM31
        self.v = self.velM33 - self.velM31
        
        
        ### get the mass of each component in M31 
        ### disk
        # **** self.rdisk = scale length (no units)
        self.rdisk = 5.0
        # **** self.Mdisk set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mdisk = GM.ComponentMass(M31_fp, 2) * 1e12 
        ### bulge
        self.rbulge = 1.0  # set scale length (no units)

        # **** self.Mbulge  set with ComponentMass function. Remember to *1e12 to get the right units Use the right ptype
        self.Mbulge = GM.ComponentMass(M31_fp, 3) * 1e12
        
        # Halo
        # **** self.rhalo = set scale length from HW5 (no units)
        self.rhalo = 60.0
        # **** self.Mhalo set with ComponentMass function. Remember to *1e12 to get the right units. Use the right ptype
        self.Mhalo = GM.ComponentMass(M31_fp, 1) * 1e12
    
    
    def HernquistAccel(self, M, r_a, r): # it is easiest if you take as an input the position VECTOR 
        """ Calculate the Hernquist acceleration based on the position vector.
        Parameters
        ----------
        r : array_like
            The position vector.
        r_a : float
            The scale length of the Hernquist profile.
        M : float
            The mass of the component.
        
        Returns
        -------
        Hern : array_like
            The acceleration vector.
        """
        
        ### **** Store the magnitude of the position vector
        rmag = np.sqrt(np.sum(r**2)) 
        
        ### *** Store the Acceleration
        Hern = -(self.G * M) * r / (rmag * (r_a + rmag)**2) 
        # NOTE: we want an acceleration VECTOR so you need to make sure that in the Hernquist equation you 
        # use  -G*M/(rmag *(ra + rmag)**2) * r --> where the last r is a VECTOR 
        
        return Hern
    
    
    
    def MiyamotoNagaiAccel(self, M, r_d, r):# it is easiest if you take as an input a position VECTOR  r 
        """ Calculate the disk acceleration based on the Miyamoto-Nagai profile.
        
        Parameters
        ----------
        M: float
            The mass of the disk component.
        r_d: float
            The scale length of the Miyamoto-Nagai profile.
        r: array_like
            The position vector.
        
        Returns
        -------
        MN: array_like
            The acceleration vector.
        """

        # store the magnitude of the x and y components of the position vector
        R = np.sqrt(r[0]**2 + r[1]**2)
        z_d = r_d / 5.0
        B = r_d + np.sqrt(r[2]**2 + z_d**2)
        # we can deal with this by multiplying the whole thing by an extra array that accounts for the 
        # differences in the z direction:
        #  multiply the whole thing by :   np.array([1,1,ZSTUFF]) 
        # where ZSTUFF are the terms associated with the z direction
        MN = -(self.G*M)*r/(R**2 + B**2)**(3/2) * np.array([1, 1, B/np.sqrt(r[2]**2 + z_d**2)])

        ### Acceleration **** follow the formula in the HW instructions
        # AGAIN note that we want a VECTOR to be returned  (see Hernquist instructions)
        # this can be tricky given that the z component is different than in the x or y directions. 
       
        # the np.array allows for a different value for the z component of the acceleration
        return MN
     
    
    def M31Accel(self, r): # input should include the position vector, r
        """ Calculate the acceleration of all acceleration components of M31.

        Parameters
        ----------
        r: array_like
            The position vector.
        
        Returns
        -------
        a: array_like
            The acceleration vector.
        """

        ### Call the previous functions for the halo, bulge and disk
        # **** these functions will take as inputs variable we defined in the initialization of the class like 
        # self.rdisk etc. 
        Halo_acc = self.HernquistAccel(self.Mhalo, self.rhalo, r)
        Bulge_acc = self.HernquistAccel(self.Mbulge, self.rbulge, r)
        Disk_acc = self.MiyamotoNagaiAccel(self.Mdisk, self.rdisk, r)

        # return the SUM of the output of the acceleration functions - this will return a VECTOR 
        a = np.sum([Halo_acc, Bulge_acc, Disk_acc], axis=0)

        return a
    
    
    
    def LeapFrog(self, r, v, dt): # take as input r and v, which are VECTORS. Assume it is ONE vector at a time
        """ Predicts the position and velocity of the orbiting body using the LeapFrog method. 
        
        Parameters
        ----------
        r: array_like
            The position vector.
        v: array_like
            The velocity vector.
        dt: float
            The time step for the prediction.
        
        Returns
        -------
        rnew: array_like
            The predicted position vector at the next time step.
        vnew: array_like
            The predicted velocity vector at the next time step.
        """
        
        # predict the position at the next half timestep
        rhalf = r + 0.5 * dt * v
        
        # using the acceleration field at the rhalf position 
        a_half = self.M31Accel(rhalf)  # calculate acceleration at rhalf
    
        # predict the final velocity at the next timestep 
        vnew = v + dt * a_half  # update velocity using acceleration
        
        # predict the final position using the average of the current velocity and the final velocity
        # this accounts for the fact that we don't know how the speed changes from the current timestep to the 
        # next, so we approximate it using the average expected speed over the time interval dt. 
        rnew = rhalf + 0.5 * dt * vnew  # update position using average velocity
        
        return rnew, vnew  # return the new position and velocity vectors
    
    
    
    def OrbitIntegration(self, t0, dt, tmax):
        """ Function integrator to solve the equations of motion 
        and compute the future orbit of M33 for 10 Gyr into the future.
        
        Parameters
        ----------
        t0 : float
            The starting time.
        dt : float
            The time step for the integration.
        tmax : float
            The maximum time for the integration.
        
        Returns
        -------
        None
        """

        # initialize the time to the input starting time
        t = t0
        
        # initialize an empty array of size :  rows int(tmax/dt)+2  , columns 7
        orbit = np.zeros((int(tmax/dt)+2, 7))
        
        # initialize the first row of the orbit
        orbit[0] = t0, *tuple(self.r), *tuple(self.v)
        # this above is equivalent to 
        # orbit[0] = t0, self.r0[0], self.r0[1], self.r0[2], self.v0[0], self.v0[1], self.v0[2]
        
        
        # initialize a counter for the orbit.  
        i = 1 # since we already set the 0th values, we start the counter at 1
        
        # start the integration (advancing in time steps and computing LeapFrog at each step)
        while (t < tmax):  # as long as t has not exceeded the maximal time 
            
            # **** advance the time by one timestep, dt
           
            t += dt
            
            # **** store the new time in the first column of the ith row
            orbit[i, 0] = t
            # ***** advance the position and velocity using the LeapFrog scheme
            # remember that LeapFrog returns a position vector and a velocity vector  
            # as an example, if a function returns three vectors you would call the function and store 
            # the variable like:     a,b,c = function(input)
            r_old = orbit[i-1, 1:4] # get the old position vector from the previous row
            v_old = orbit[i-1, 4:7] # get the old velocity vector from the previous row
            r_new, v_new = self.LeapFrog(r_old, v_old, dt) # call the LeapFrog function to get the new position and velocity
         
    
            # ****  store the new position vector into the columns with indexes 1,2,3 of the ith row of orbit
            orbit[i, 1:4] = r_new[0], r_new[1], r_new[2]  # store the new position vector
            # TIP:  if you want columns 5-7 of the Nth row of an array called A, you would write : 
            # A[n, 5:8] 
            # where the syntax is row n, start at column 5 and end BEFORE column 8
            
            
            # ****  store the new velocity vector into the columns with indexes 4,5,6 of the ith row of orbit
            orbit[i, 4:7] = v_new[0], v_new[1], v_new[2]
            
            
            # **** update counter i , where i is keeping track of the number of rows (i.e. the number of time steps)
            i += 1
        
        
        # write the data to a file
        np.savetxt(self.filename, orbit, fmt = "%11.3f"*7, comments='#', 
                   header="{:>10s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}{:>11s}"\
                   .format('t', 'x', 'y', 'z', 'vx', 'vy', 'vz'))
        
        # there is no return function