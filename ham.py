import os
import sys
import math
import time
import numpy as np
from liblibra_core import *
from libra_py import units as units


def td_ham(t, params):
    """
    This is a generic periodic function

    """

    func_type = params["func_type"]
    mean = params["mean"]          # in eV
    freqs = params["freqs"]        # in cm^-1
    ampls = params["ampls"]        # unitless
    scaling1 = params["scaling1"]
    scaling2 = params["scaling2"]


    nfreqs = len(freqs)

    #dt = 1.0 * units.fs2au
    #t = (istep + step) * dt

    res = 0.0

    for n in range(nfreqs):
        w = freqs[n] * units.wavn2au

        si = 0.0
        if func_type == 0:
            si = math.sin(w * t) 
        elif func_type == 1:
            si = math.sin(w * t) 
            si = si * si

        res = res + ampls[n] * si

    res = scaling1 * (mean + scaling2 * res) * units.ev2au

    return res
    
    
def compute_Hvib(t, params):
    """
    Model N-state Hamiltonian
    
    """            

    nstates = params["nstates"]


    ham_adi = CMATRIX(nstates, nstates)
    nac_adi = CMATRIX(nstates, nstates)    
       
    # Energy levels
    e = 0.0
    for istate in range(nstates-1):
        de = td_ham(t, params["gap_params"][istate])
        e = e + de
        ham_adi.set(istate+1, istate+1,  e * (1.0+0.0j) )     

    # NACs
    cnt = 0
    for istate1 in range(nstates):
        for istate2 in range(istate1+1, nstates):
            nac = td_ham(t, params["nac_params"][cnt])
            nac_adi.set(istate1, istate2,  nac * (1.0+0.0j) )    
            nac_adi.set(istate2, istate1, -nac * (1.0+0.0j) )       
            cnt = cnt + 1
            
    return ham_adi, nac_adi



def compute_Hvib_timeseries(nsteps, params):

    nstates = params["nstates"]
    period = params["period"]
    dt = params["dt"]
    istep = params["istep"]
    times = np.zeros(nsteps) # time in a.u.
    energy = np.zeros( (nsteps, nstates) )  # energy gap in a.u.
    nac = np.zeros( (nsteps, int(nstates*(nstates-1)/2) ) ) # NACs in a.u. 


    for step in range(nsteps):
        t = dt * (istep + step % period)
        times[step] = t
        #print (t/dt)
        ham_adi, nac_adi = compute_Hvib(t, params)

        for istate in range(nstates):
            energy[step, istate] = ham_adi.get(istate, istate).real

        cnt = 0
        for istate1 in range(nstates):
            for istate2 in range(istate1+1, nstates):
                nac[step, cnt] = nac_adi.get(istate1, istate2).real
                cnt = cnt + 1

    return times, energy, nac


class tmp:
    pass

def compute_model_nbra_direct(q, params, full_id):
    """   
    Read in the vibronic Hamiltonians along the trajectories    

    Args: 
        q ( MATRIX(1,1) ): coordinates of the particle, ndof, but they do not really affect anything
        params ( dictionary ): model parameters

            * **params["timestep"]** ( int ):  [ index of the file to read ]
            * **params["prefix"]**   ( string ):  [ the directory where the hdf5 file is located ]
            * **params["filename"]** ( string ):  [ the name of the HDF5 file ]
            * **params["period"]** ( int ): the length of the pretended short trajectory that is repeated 
        
    Returns:       
        PyObject: obj, with the members:

            * obj.hvib_adi ( CMATRIX(n,n) ): adiabatic vibronic Hamiltonian 
            
    """
                                    
    hvib_adi, basis_transform, time_overlap_adi = None, None, None
  
    Id = Cpp2Py(full_id)
    indx = Id[-1]    
    timestep = params["timestep"]
    nadi = params["nstates"]
    period = params["period"]
    dt = params["dt"]
    istep = params["istep"]
        
    #============ Electronic Hamiltonian, NAC, and Vibronic Hamiltonian =========== 
    t = dt * (istep + timestep % period)
    ham_adi, nac_adi = compute_Hvib( t, params )   
    hvib_adi = ham_adi - 1j * nac_adi
        
    #=========== Basis transform, if available =====
    basis_transform = CMATRIX(nadi, nadi) 
    basis_transform.identity()        
                                                
    #========= Time-overlap matrices ===================
    time_overlap_adi = CMATRIX(nadi, nadi)
    time_overlap_adi.identity()    

        
    obj = tmp()
    obj.ham_adi = ham_adi
    obj.nac_adi = nac_adi
    obj.hvib_adi = hvib_adi
    obj.basis_transform = basis_transform
    obj.time_overlap_adi = time_overlap_adi
            
    return obj


