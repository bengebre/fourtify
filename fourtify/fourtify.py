import numpy as np
import pyorb
from astropy.coordinates import SkyCoord
from .tbp import prop

class Fourtify:
    """
    Fourtify finds additional sources along a specified orbit in a transient source dataset.

    Attributes:
    -----------
    obs_radecs : ndarray
        Nx2 array of RA and DEC observations in degrees.
    obs_times : ndarray
        Nx2 array of observation times.  TDB Julian dates.
    obs_locs : ndarray
        Nx2 array of heliocentric observer locations in time. Units are AU.

    Methods:
    --------
    orbit(elems,epoch,thresh): 
        Returns the indices of the observations along the specified orbit.
    __ec2eq(vectors,obliquity):
        Private method that rotates from ecliptic to equitorial coordinates.
    __el2rv(a,e,i,node,peri,M):
        Private method that takes orbital elements and converts them to position and velocity.
    __orb2obs(elems,epoch_tdb,obs_locs,obs_times_tdb):
        Private method that propagates the orbit state vector to the specified observation times.
    """
    
    def __init__(self,obs_radecs,obs_times,obs_locs):
        """
        Parameters:
        -----------
        obs_radecs : ndarray
            Nx2 array of RA and DEC observations in degrees.
        obs_times : ndarray
            Nx2 array of observation times.  TDB Julian dates.
        obs_locs : ndarray
            Nx2 array of heliocentric observer locations in time. Units are AU.
        """

        self.obs_radecs = obs_radecs #angular observation ra/decs
        self.obs_times = obs_times #observation times TDB
        self.obs_locs = obs_locs #heliocentric observer location vectors AU

    @staticmethod
    def __ec2eq(vectors,obliquity):
        """
        For ecliptic to equitorial rotation.

        Parameters:
        -----------
        vectors : ndarray
            Nx3 array of position vectors to rotate.
        obliquity : float
            Obliquity of the ecliptic in radians.

        Returns:
        --------
        ndarray
            Rotated vectors.
        """
        rot_x = np.array([[1, 0, 0],
                          [0, np.cos(obliquity), -np.sin(obliquity)],
                          [0, np.sin(obliquity), np.cos(obliquity)]])
        
        return np.dot(rot_x,vectors.T).T

    @staticmethod
    def __el2rv(a,e,i,node,peri,M):
        """
        Convert orbital elements in degrees (M=mean anomaly) to pos,vel (AU,AU/d)

        Parameters:
        -----------
        a : float
            Semi major axis of orbit. (AU)
        e : float
            Eccentricity of orbit.
        i : float
            Inclination of orbit. (deg)
        node : float
            Right ascension of the ascending node. (deg)
        peri : float
            Argument of periapse. (deg)
        M : float
            Mean anomoly. (deg)

        Returns:
        --------
        ndarray
            Position and velocity of orbit orbital elements
        """
        
        M_sun = 1.988409870698051e+30 #const.M_sun.value
        au_m = 149597870700.0 #float(u.AU.to('m'))
        
        orb = pyorb.Orbit(M0 = M_sun, degrees=True, type='mean')
        orb.update(a=a*au_m, e=e, i=i, Omega=node, omega=peri, anom=M)
    
        return np.concatenate(((orb.r/au_m).T[0],(86400*orb.v/(au_m)).T[0]))

    #'elems' orbital elements from JPL or Find_Orb
    #'epoch_tdb' is TDB epoch of orbit
    #'obs_locs' are skyfield objects that are heliocentric observer locations at the observation TDB times - calc once only
    #'obs_times_tdb' UT times of observer's observations converted to TDB
    @staticmethod
    def __orb2obs(elems,epoch_tdb,obs_locs,obs_times_tdb):
        """
        Take an orbit and propagate it to all observation times then transform resulting 
        heliocentric vectors to the observer's location on the sky.

        Parameters:
        -----------
        elems : tuple
            Orbital elements: (a,e,i,node,peri,M) (AU,float,deg,deg,deg,deg)
        epoch_tdb : float
            Epoch (time) for orbital elements.  (TDB Julian date)
        obs_locs : ndarray
            Heliocentric observer locations in time. (AU)
        obs_times_tdb : ndarray
            Observation times. (TDB)

        Returns:
        --------
        ndarray
            RA and DEC of propagated position vectors at the observer location and times.
        """
        
        ooe = np.deg2rad(23.4392911) #from Horizons vector report web table
        mu = 0.00029591220819207774 #(const.M_sun * const.G).to('AU**3/day**2').value; units:AU**3/d**2
        c = 173.1446326742403 #const.c.to('AU/s').value*86400; units:AU/d
    
        #convert orbital elements to state vector
        rv = Fourtify.__el2rv(*elems)
        
        #prop state vector at epoch to obs_times
        rv_prop = prop(obs_times_tdb,rv,epoch_tdb,mu)
        
        #this is the propagated heliocentric vector rotated to the equatorial plane
        rv_prop_eq = Fourtify.__ec2eq(rv_prop[:,0:3],ooe)
    
        #equatorial object position from observer location
        obj_pos_eq = rv_prop_eq[:,0:3] - obs_locs
        
        #calculate light time to object from observer and decrement TDB propagation time by that duration
        obs_times_tdb_ltc = obs_times_tdb - (np.linalg.norm(obj_pos_eq,axis=1)/c)
        
        #propagate heliocentric *ecliptic* vector to light time corrected time 
        rv_prop_ltc = prop(obs_times_tdb_ltc,rv,epoch_tdb,mu)
        
        #convert ltc positions to equatorial positions
        rv_prop_ltc_eq = Fourtify.__ec2eq(rv_prop_ltc[:,0:3],ooe)
        
        #calculate observer relative object position with ltc
        obj_pos_eq_ltc = rv_prop_ltc_eq[:,0:3] - obs_locs

        #convert observer relative positions to RA/DECs
        coords = SkyCoord(x=obj_pos_eq_ltc[:,0], y=obj_pos_eq_ltc[:,1], z=obj_pos_eq_ltc[:,2], 
                          unit='AU', representation_type='cartesian', frame='icrs')
        radecs = np.column_stack([coords.spherical.lon.value,coords.spherical.lat.value])
        
        return radecs

    def orbit(self,elems,epoch,thresh):
        """
        Takes a specified orbit propagates it to all observation times then finds the sources 
        with small offsets from the propagated path that might belong to the object.

        Parameters:
        -----------
        elems : tuple
            Orbital elements: (a,e,i,node,peri,M) (AU,float,deg,deg,deg,deg)
        epoch : float
            Epoch (time) for orbital elements.  (TDB Julian date)
        thresh : tuple
            Threshold for RA/DEC offset from predicted orbit. (maximum arcsec, arcsec/day, init arcsec)
        
        Returns:
        --------
        list
            Indices of observations that were within the threshold constraints.
        ndarray
            Deviations of returned observation indices from predicted position. (arcsec)
        """
        
        prop_radecs = self.__orb2obs(elems,epoch,self.obs_locs,self.obs_times)
        dradecs = np.linalg.norm(prop_radecs - self.obs_radecs,axis=1)*3600
        found_abs_idx = np.where(dradecs < thresh[0])[0]
        found_rate_idx = np.where(np.abs((dradecs)/(self.obs_times-epoch)) < thresh[1] + thresh[2])[0]
        fidx = sorted(list(set(found_abs_idx) & set(found_rate_idx)))

        return dradecs[fidx],fidx
