import numpy, scipy
import numpy as np

import inspect
import os, sys, typing
from typing import List, Tuple, Callable

import pyscf
from pyscf import gto, scf, fci
from pyscf.fci import direct_spin1

from qcgf.gf import GreensFunctionMixin

def greens_func_multiply_ip(ham, vector, linear_part, **kwargs):
    return np.array(ham(vector.real, **kwargs) + 1j * ham(vector.imag, **kwargs) + linear_part * vector)

def greens_func_multiply_ea(ham, vector, linear_part, **kwargs):
    return np.array(-ham(vector.real, **kwargs) - 1j * ham(vector.imag, **kwargs) + linear_part * vector)

def greens_func_multiply_ip_mor(ham, vector, linear_part):
    return np.array(np.dot(ham, vector) + linear_part * vector)

def greens_func_multiply_ea_mor(ham, vector, linear_part):
    return np.array(-np.dot(ham, vector) + linear_part * vector)


class DirectSpin1FullConfigurationInteraction(GreensFunctionMixin):
    def __init__(self, scf_obj: scf.hf.SCF = None):
        """
        Initialize the DirectSpin1FullConfigurationInteraction solver.

        Parameters:
            scf_obj : SCF class object, optional
                The SCF object used for the mean-field calculations.
            tol : float, optional
                Tolerance for numerical computations.
            verbose : int, optional
                Verbosity level (0, 1, 2, 3, 4).

        Attributes:
            fci_obj : FCI class object
                The FCI object used for FCI calculations.
            scf_obj : SCF class object
                The SCF object used for the mean-field calculations.
            tol : float
                Tolerance for numerical computations.
            verbose : int
                Verbosity level (0, 1, 2, 3, 4).
            stdout : file
                File object to which log messages will be written.
        """
        self._fci_obj = None
        self._scf_obj = scf_obj  # Fix typo: mf -> scf_obj

        self.nele = None
        self.norb = None

        self._mol_obj = scf_obj.mol
        self.verbose  = scf_obj.verbose
        self.stdout   = scf_obj.stdout

    def build(self, coeff: np.ndarray = None) -> None:
        """
        Build and solve the FCI object.

        Parameters:
            coeff : 2D array of floats, optional
                The orbital coefficients. If not provided,
                the coefficients from the SCF object will 
                be used.
        """
        if coeff is None:
            coeff = self._scf_obj.mo_coeff

        fci_obj  = fci.FCI(self._mol_obj, mo=coeff, singlet=False)
        fci_obj.verbose = self.verbose
        fci_obj.stdout  = self.stdout
        
        kwargs = inspect.signature(fci_obj.kernel).parameters

        h0     = kwargs.get('ecore', None).default
        h1e    = kwargs.get('h1e'  , None).default
        h2e    = kwargs.get('eri'  , None).default

        norb   = kwargs.get('norb' , None).default
        norb2  = norb * (norb + 1) // 2
        nele   = kwargs.get('nelec', None).default

        assert isinstance(h0, float)
        assert h1e.shape == (norb, norb)
        assert h2e.shape == (norb2, norb2)

        fci_ene, fci_vec = fci_obj.kernel(
            h1e=h1e, eri=h2e, norb=norb, nelec=nele, 
            ci0=None, ecore=h0, verbose=self.verbose
            )
        
        self._fci_obj = fci_obj
        self._fci_obj.h0  = h0
        self._fci_obj.h1e = h1e
        self._fci_obj.h2e = h2e

        self._fci_obj.norb = norb
        self._fci_obj.nele = nele

        self._fci_obj.fci_ene = fci_ene
        self._fci_obj.fci_vec = fci_vec

    def get_gf_ip(self, p: int, q: int, omega: float, eta: float = None, is_mor: bool = False, omega_mor: List[float] = None) -> np.ndarray:
        '''
        Compute FCI IP Green's function in MO basis, this is the fast routine without 
        constructing the full FCI Hamiltonian.

        Parameters:
            p : int
                Orbital index used for computing the Green's function.
            q : int
                Orbital index used for computing the Green's function.
            omega : float
                Frequency for which the Green's function is computed.
            eta : float, optional
                Broadening factor for numerical stability.
            is_mor : bool, optional
                If True, use Model Order Reduction (MOR) technique.
            omega_mor : list of floats, optional
                List of frequencies used for MOR.

        Returns:
            gfvals : 3D array of complex floats
                The computed Green's function values.
        '''
        scf_obj = self._scf_obj
        
        if self._fci_obj is None:
            self.build()
        fci_obj = self._fci_obj

        norb = fci_obj.norb
        nele = fci_obj.nele
        nele_alph, nele_beta = nele

        h1e = fci_obj.h1e
        h2e = fci_obj.h2e

        ene_fci = fci_obj.fci_ene
        vec_fci = fci_obj.fci_vec
        ham_ip  = fci_obj.absorb_h1e(h1e, h2e, norb, (nele_alph-1, nele_beta), 0.5)

        e_q = fci.addons.des_a(vec_fci, norb, (nele_alph, nele_beta), q)

        gf_pq_omega = 0.0

        if is_mor:
            omega_comp = omega_mor
        else:
            omega_comp = omega

        b_p = fci.addons.des_a(vec_fci, norb, (nele_alph, nele_beta), p)
        if is_mor:
            x_vec = numpy.zeros((len(b_p), 1), dtype=numpy.complex128)

        def hx(x, args=None):
            return ham_ip.dot(x)

FCIGF = DirectSpin1FullConfigurationInteraction

if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(
        atom = 'H 0 0 0; Li 0 0 1.1',
        basis = 'sto-3g',
        verbose = 0,
    )
    mf = scf.RHF(mol)
    mf.kernel()

    gf = FCIGF(mf)
    gf.build()
    gf.get_gf_ip(0, 0, 0.1)