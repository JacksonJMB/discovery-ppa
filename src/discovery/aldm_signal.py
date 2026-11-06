import sys
import os
import re
import inspect
import types
import typing
import warnings
from collections.abc import Iterable

import numpy as np
import scipy.interpolate as si
import scipy.constants as sc
import jax
import jax.numpy as jnp
from astropy.coordinates import SkyCoord
import math

from . import signals
from . import matrix
from . import const
from . import solar


def Sa_to_coupling(Sa, axion_mass, DM_density=0.4):
    # Sa in radians, axion_mass in eV, DM_density in GeV/cm^3
    # Returns axion-photon coupling in GeV^-1
    return Sa * axion_mass*1e-9 / (DM_density * const.inv_cm_in_GeV**3)**0.5

def get_psr_fmat_aldm(psr, axion_mass):
    # axion mass (rest energy) must be provided in eV
    omega = axion_mass * sc.eV / sc.hbar 
    # lc = get_coherence_length(v0, axion_mass)

    # f  = np.arange(1, components + 1, dtype=np.float64) / T
    # df = np.diff(np.concatenate((np.array([0]), f)))

    fmat = np.zeros((psr.toas.shape[0], 2), dtype=np.float64)
    fmat[:,0] = np.sin(omega * psr.toas)
    fmat[:,1] = np.cos(omega * psr.toas)

    return omega, fmat

# def get_phi_aldm_pq(psr_p, psr_q, ma):
    
#     phi_cc = 1.0 + np.sinc(y_pq)

def makegp_aldm_psr(psr, log10_ma, v=1e-3, common=[], name='aldmGP'):
    diststar_param = f'{psr.name}_diststar'
    log10_Sa = 'log10_Sa'
    params = [diststar_param, log10_Sa]

    ma = 10.0**log10_ma
    omega, fmat = get_psr_fmat_aldm(psr, ma)

    # def getphi(params):
    #     y_ep = psr.pdist / get_coherence_length(v, ma)
    #     phi_cc = 2.0 - 2.0 * np.sinc(y_ep) * np.cos(omega*psr.pdist * const.kpc_in_s)  # = phi_ss. For a single pulsar, we have y_{pq}=0
    #     phi_sc = 0.0
    #     phi_cs = -phi_sc
    #     return 10**(2.0*params['log10_Sa']) * np.array([[phi_cc, phi_cs], [phi_sc, phi_cc]], dtype=np.float64)
    def getphi(params):
        distance = params[diststar_param] * psr.pdist
        y_ep = params[diststar_param] * psr.pdist / get_coherence_length(v, ma)
        angle = omega * distance * const.kpc_in_s
        phi_cc = 2.0 - 2.0 * np.sinc(y_ep) * np.cos(angle)  # = phi_ss. For a single pulsar, we have y_{pq}=0
        return 10**(2.0*params[log10_Sa]) * np.array([phi_cc, phi_cc], dtype=np.float64)
    getphi.params = params

    gp = matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), fmat)
    # gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(0,len(f))} # better for cosine
    gp.name = psr.name
    if hasattr(psr, "pos"):
        gp.pos = psr.pos
    gp.gpname, gp.gpcommon = name, common

    return gp

def get_global_fmat_aldm(psrs, axion_mass):
    # axion mass (rest energy) must be provided in eV
    omega = axion_mass * sc.eV / sc.hbar 
    # lc = get_coherence_length(v0, axion_mass)

    # f  = np.arange(1, components + 1, dtype=np.float64) / T
    # df = np.diff(np.concatenate((np.array([0]), f)))

    fmats = []
    for psr in psrs:
        fmat_psr = np.zeros((psr.toas.shape[0], 2), dtype=np.float64)
        fmat_psr[:,0] = np.sin(omega * psr.toas)
        fmat_psr[:,1] = np.cos(omega * psr.toas)
        fmats.append(fmat_psr)
    fmat_full = np.block_diag(fmats)

    return omega, fmat_full

def makeglobalgp_aldm(psrs, log10_ma, v=1e-3, common=[], name='aldmGP'):
    # dist = f'{psr.name}_dist'
    params = ['log10_Sa']
    n_psrs = len(psrs)

    ma = 10**log10_ma
    omega, fmat = get_global_fmat_aldm(psrs, ma)

    lc = get_coherence_length(v, ma)
    def getphi(params):
        phimats = [] # Needs more work
        for i,p in enumerate(psrs):
            for j,q in enumerate(psrs):
                p_to_q_vector = p.pdist*p.unit_xyz - q.pdist*q.unit_xyz
                x_pq = np.sqrt(p_to_q_vector @ p_to_q_vector) # dist between p and q in kpc
                
                y_pq = x_pq / lc
                y_ep = p.pdist / lc
                y_eq = q.pdist / lc

                p_angle = omega * p.pdist * const.kpc_in_s
                q_angle = omega * q.pdist * const.kpc_in_s
                diff_angle = p_angle - q_angle

                phi_cc = 1.0 + np.sinc(y_pq)*np.cos(diff_angle) - np.sinc(y_ep)*np.cos(p_angle) - np.sinc(y_eq)*np.cos(q_angle)  # = phi_ss
                phi_sc = np.sinc(y_pq)*np.sin(diff_angle) - np.sinc(y_ep)*np.sin(p_angle) - np.sinc(y_eq)*np.sin(q_angle)
                phi_cs = -phi_sc
                
                phimat[2*i,2*j] = phi_cc; phimat[2*i+1,2*j+1] = phi_cc
                phimat[2*i,2*j] = phi_cc; phimat[2*i+1,2*j+1] = phi_cc
                jnp.array([[phi_cc, phi_cs], [phi_sc, phi_cc]], dtype=jnp.float64)
        return 10**(2*params['log10_Sa']) * np.array([[phi_cc, phi_cs], [phi_sc, phi_cc]], dtype=np.float64)
    getphi.params = params

    gp = matrix.VariableGP(matrix.NoiseMatrix2D_var(getphi), fmat)
    # gp.index = {f'{psr.name}_{name}_coefficients({len(f)})': slice(0,len(f))} # better for cosine
    gp.name = psr.name
    if hasattr(psr, "pos"):
        gp.pos = psr.pos
    gp.gpname, gp.gpcommon = name, common

    return gp

def makepsrgp_aldm(psr):
    log10_coupling = f'{psr.name}_log10_coupling'
    log10_axion_mass = f'{psr.name}_log10_axion_mass'
    psr_dist = f'{psr.name}_dist'

    params = [log10_coupling, log10_axion_mass, psr_dist]

    rotation_factor = (fref / psr.freqs)**2
    bins = quantize(psr.toas)
    Umat = np.vstack([(bins == i) * rotation_factor for i in range(bins.max() + 1)]).T
    # observations = psr.obsnames
    # Umat = np.vstack([(observations == obs) * rotation_factor for obs in psr.obsnames_unique]).T
    ones = np.ones(Umat.shape[1], dtype=np.float64)

    if all(par in noisedict for par in params):
        phi = (10.0**(2.0 * noisedict[log10_erm])) * ones

        return matrix.ConstantGP(matrix.NoiseMatrix1D_novar(phi), Umat)
    else:
        ones = matrix.jnparray(ones)
        def getphi(params):
            return (10.0**(2.0 * params[log10_erm])) * ones
        getphi.params = params
        
        return matrix.VariableGP(matrix.NoiseMatrix1D_var(getphi), Umat)

# def makeglobalgp_aldm(psrs, prior):
#     argspec = inspect.getfullargspec(prior)
#     argmaps = [[(arg if arg in common else f'{name}_{arg}' if f'{name}_{arg}' in common else f'{psr.name}_{name}_{arg}') +
#                 (f'({components})' if argspec.annotations.get(arg) == typing.Sequence else '')
#                 for arg in argspec.args if arg not in ['f', 'df']] for psr in psrs]


"""
Modified from https://github.com/XueXiao-Physics/PPA/blob/main/ppa/ppa.py
"""

def get_coherence_length(v, axion_mass):
    # v as a fraction of c, axion_mass in eV
    lc = sc.hbar * sc.c / sc.eV / v / sc.parsec*1e-3 / axion_mass
    return lc # coherence length in kpc

def Get_Star_Locations(psrs, sDTE):

        PSR_POS = self.PSR_LOC * sDTE[:,None] * self.DTE0[:,None]
        DL_MTX = np.zeros((self.NPSR,self.NPSR))

        for p in range(self.NPSR):
            for q in range(p+1,self.NPSR):
                l_p = PSR_POS[p]
                l_q = PSR_POS[q]
                DL_MTX[p,q] = DL_MTX[q,p] = np.sqrt(np.power(l_p-l_q,2).sum())


        return PSR_POS,DL_MTX


def Get_Phi_ADM(psrs,v,ma,sDTE,Sa2):

    DTE = [psr.DTE0 * sDTE for psr in psrs]
    PSR_POS , DL_MTX = self.Get_Star_Locations( sDTE )
    omega = ma * sc.eV / sc.hbar
    lc = get_lc( v , ma )


    sincpq_all =  np.sinc( DL_MTX / lc / np.pi )
    sincp_all = np.sinc( DTE/ lc / np.pi )
    cp_all = np.cos( omega * DTE * KPC_TO_S )
    sp_all = np.sin( omega * DTE * KPC_TO_S )
    cpq_all = cp_all[:,None]*cp_all[None,:] + sp_all[:,None]*sp_all[None,:]
    spq_all = sp_all[:,None]*cp_all[None,:] - cp_all[:,None]*sp_all[None,:]    
    
    
    ADM_Phi_compact = np.zeros( (self.NPSR*2,self.NPSR*2 ))
    pointer_x = 0
    for p in range(self.NPSR):
        pointer_y = 0
        for q in range(self.NPSR):
            sincpq = sincpq_all[p,q];sincp = sincp_all[p];sincq = sincp_all[q]
            cpq = cpq_all[p,q]; spq = spq_all[p,q]
            cp = cp_all[p] ; sp = sp_all[p] ; cq = cp_all[q] ; sq = sp_all[q]
            
            Phi_ccss = 1 + sincpq*cpq - sincp*cp - sincq*cq
            Phi_sc =  sincpq*spq - sincp*sp + sincq*sq
            Phi_cs = -Phi_sc

            Phi_pq = np.array([[Phi_ccss,Phi_cs],[Phi_sc,Phi_ccss]]) * Sa2
            ADM_Phi_compact[ pointer_x:pointer_x+2 , pointer_y:pointer_y+2 ] = Phi_pq
            pointer_y += 2
        pointer_x += 2
    #ADM_Phi_inv_compact , ADM_Phi_logdet = svd_inv(ADM_Phi_compact)
    # ADM_Phi_inv_compact = sl.inv(ADM_Phi_compact) ; ADM_Phi_logdet = nl.slogdet(ADM_Phi_compact)[1]
    # Phi_logdet += ADM_Phi_logdet

    return ADM_Phi_compact


#=====================================================#
#    F for axion                                      #
#=====================================================#

def Get_F_ADM(ma,adm_signal,external_TOA = False):

    omega = ma * sc.eV/sc.hbar
    F_ADM = []
    for p in range(self.NPSR):
        F_ADM_psr = []
        for ss in range( self.NSUBSETS_by_SS[p] ):
            NOBS = self.NOBS[p][ss]
            t = self.TOAs[p][ss] 
            if type(external_TOA) in [list,np.ndarray,np.array]: # June 2025
                NOBS = len(external_TOA)
                t = external_TOA
            if adm_signal in ["auto","full"]:
                F_ADM_ss = np.zeros((2,NOBS))
                F_ADM_ss[ 0 ] = np.cos(omega * t.astype(np.float64))
                F_ADM_ss[ 1 ] = np.sin(omega * t.astype(np.float64))
                ndim = 2
            elif adm_signal in ["none"]:
                F_ADM_ss = np.zeros((0,NOBS))
                ndim = 0

            F_ADM_psr.append(F_ADM_ss)
        F_ADM.append(F_ADM_psr)
        
    #return F_matrix  , F_blocks_bysubsets
    return F_ADM, ndim