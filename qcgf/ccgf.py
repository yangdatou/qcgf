import numpy, scipy

import pyscf
from pyscf import gto, scf, cc
from pyscf.cc.gccsd import amplitudes_to_vector, vector_to_amplitudes
from pyscf.cc.eom_gccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea
from pyscf.cc.eom_gccsd import vector_to_amplitudes_ip, vector_to_amplitudes_ea

from qcgf.gf import GreenFunctionMixin

class _GeneralizedSpinCoupledClusterSingleDouble(cc.gccsd.GCCSD):
    h1e = None
    eri = None

    norb  = None
    nelec = None

    ene_ccsd = None
    amp_ccsd = None
    lam_ccsd = None

def ccsd_gf_ip(cc_obj, omegas, t1=None, t2=None, l1=None, l2=None, eta=1e-4):
    '''
    Compute IP part of GCCSD-GF

    Input:
        gf_obj: GreenFunctionMixin object

    '''

class GeneralizedSpinCoupledClusterSingleDouble(GreenFunctionMixin):
    _cc_obj  = None

    def build(self, coeff: numpy.ndarray = None) -> None:
        """
        Build and solve the CCSD object.
        Note: h0 will not be included.

        Parameters:
            coeff : 2D array of floats, optional
                The orbital coefficients. If not provided,
                the coefficients from the SCF object will 
                be used.
        """
        if coeff is None:
            coeff = self._scf_obj.mo_coeff

        from qcgf.utils import make_h1_and_h2
        nelec, norb, h1e, eri = make_h1_and_h2(self._mol_obj, coeff)
        
        fci_obj = _DirectSpin1FullConfigurationInteraction(self._mol_obj)
        ene_fci, vec_fci = fci_obj.kernel(
            h1e=h1e,  eri=eri, norb=norb, nelec=nelec, 
            ci0=None, ecore=0.0, verbose=self.verbose
            )

        fci_obj.h1e = h1e
        fci_obj.eri = eri

        fci_obj.norb  = norb
        fci_obj.nelec = nelec

        fci_obj.ene_fci = ene_fci
        fci_obj.vec_fci = vec_fci

        self._fci_obj = fci_obj