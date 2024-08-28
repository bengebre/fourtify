import numpy as np
from scipy.integrate import odeint

def twobody(X, t, mu):
    """
    Time derivative of the state vector.
    """

    denom = (X[0]**2 + X[1]**2 + X[2]**2)**1.5

    dX_dt = [X[3], X[4], X[5],
             -mu*X[0]/denom,
             -mu*X[1]/denom,
             -mu*X[2]/denom]

    return dX_dt

def prop(times, X, t, mu):
    """
    Propagate a state vector (X) at a time (t) to other times (times) given a gravitational parameter (mu).
    """

    #time sorting for integrator
    dts = times - t #time deltas from input
    sidx = np.argsort(dts) #sorted dts indicies
    sdts = dts[sidx] #sorted dts
    gtez_bools = sdts>=0 #sdts greater than or equal to zero boolean
    psdts = np.concatenate(([0],sdts[gtez_bools])) #positive time deltas from 0
    nsdts = np.concatenate(([0],np.flip(sdts[~gtez_bools]))) #negative time deltas from 0

    #integrate in negative time direction then positive time direction
    intn = odeint(twobody, X, nsdts, (mu,), rtol=1e-6, atol=1e-6)[-1:0:-1] #integrate negative time deltas
    intp = odeint(twobody, X, psdts, (mu,), rtol=1e-6, atol=1e-6)[1:] #integrate positive time deltas

    #state at specified times
    X_times = np.concatenate((intn,intp))[sidx,:] #join integrations and reproduce original times order

    return X_times
