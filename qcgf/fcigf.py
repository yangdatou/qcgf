import inspect
import os, sys, typing
from typing import List, Tuple, Callable

import numpy, scipy

import pyscf
from pyscf import gto, scf, fci, lib

from pyscf.fci import direct_spin1
from pyscf.fci.direct_spin1 import _unpack
from pyscf.fci.direct_spin1 import contract_1e
from pyscf.fci.direct_spin1 import contract_2e

from qcgf.gf  import GreensFunctionMixin
from qcgf.lib import gmres

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

        self.nelec = None
        self.norb  = None

        self._mol_obj   = scf_obj.mol
        self.verbose    = scf_obj.verbose
        self.stdout     = scf_obj.stdout
        self.max_memory = scf_obj.max_memory

    def build(self, coeff: numpy.ndarray = None) -> None:
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

        h0     = kwargs.get('ecore').default
        h1e    = kwargs.get('h1e'  ).default
        h2e    = kwargs.get('eri'  ).default

        norb   = kwargs.get('norb').default
        norb2  = norb * (norb + 1) // 2
        nelec  = kwargs.get('nelec').default

        assert isinstance(h0, float)
        assert h1e.shape == (norb, norb)
        assert h2e.shape == (norb2, norb2)

        fci_ene, fci_vec = fci_obj.kernel(
            h1e=h1e, eri=h2e, norb=norb, nelec=nelec, 
            ci0=None, ecore=h0, verbose=self.verbose
            )
        
        self._fci_obj     = fci_obj
        self._fci_obj.h0  = h0
        self._fci_obj.h1e = h1e
        self._fci_obj.h2e = h2e

        self._fci_obj.norb = norb
        self._fci_obj.nelec = nelec

        self._fci_obj.fci_ene = fci_ene
        self._fci_obj.fci_vec = fci_vec

    def get_ip(self, p: int, q: int, omega: float, eta: float = None, is_mor: bool = False, omega_mor: List[float] = None) -> numpy.ndarray:
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
        h1e  = fci_obj.h1e
        h2e  = fci_obj.h2e

        nelec    = fci_obj.nelec
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0] + 1, nelec[1])
        link_index_ip = _unpack(norb, nelec_ip, None, spin=None)
        link_index_ea = _unpack(norb, nelec_ea, None, spin=None)
        diag_ip = fci_obj.make_hdiag(h1e, h2e, norb, nelec_ip)
        diag_ea = fci_obj.make_hdiag(h1e, h2e, norb, nelec_ea)

        ene_fci = fci_obj.fci_ene
        vec_fci = fci_obj.fci_vec

        gf_pq_omega = 0.0

        if is_mor:
            omega_comp = omega_mor
        else:
            omega_comp = omega

        e_q = fci.addons.des_a(vec_fci, norb, nelec, q)
        b_p = fci.addons.des_a(vec_fci, norb, nelec, p)

        if is_mor:
            x_vec = numpy.zeros_like(b_p, dtype=numpy.complex128)

        def h(v):
            hv_re  = contract_1e(h1e, v.real, norb, nelec_ip, link_index=link_index_ip)
            hv_re += contract_2e(h2e, v.real, norb, nelec_ip, link_index=link_index_ip)
            hv_im  = contract_1e(h1e, v.imag, norb, nelec_ip, link_index=link_index_ip)
            hv_im += contract_2e(h2e, v.imag, norb, nelec_ip, link_index=link_index_ip)
            hv = hv_re + 1j * hv_im + (omega - ene_fci - 1j * eta) * v
            return hv

        x0 = b_p / (diag_ip + omega - ene_fci - 1j * eta)
        x  = gmres(h, b=b_p, x0=x0, diag=(diag_ip + omega - ene_fci - 1j * eta), 
                   gmres_m=self.gmres_m, stdout=self.stdout,
                   tol=self.conv_tol, max_cycle=self.max_cycle)

        return None

    def get_ip_slow(self, omegas: List[float], ps: (List[int]|None)=None, qs: (List[int]|None)=None, eta: float=0.0) -> numpy.ndarray:
        '''
        Slow version of computing FCI IP Green's function by constructing 
        the full FCI Hamiltonian.

        Parameters:
            ps : list of ints, optional
                Orbital indices used for computing the Green's function.
                If not provided, all orbitals will be used.
            qs : list of ints, optional
                Orbital indices used for computing the Green's function.
                If not provided, all orbitals will be used.
            omegas : float
                Frequency for which the Green's function is computed.
            eta : float, optional
                Broadening factor for numerical stability.

        Returns:
            gfvals : 3D array of complex floats
                The computed Green's function values.
        '''
        scf_obj = self._scf_obj

        if self._fci_obj is None:
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        nelec_ip = (nelec[0] - 1, nelec[1])

        if ps is None:
            ps = numpy.arange(norb)

        if qs is None:
            qs = numpy.arange(norb)

        omegas = numpy.asarray(omegas)
        nomega = len(omegas)
        ps = numpy.asarray(ps)
        qs = numpy.asarray(qs)
        np = len(ps)
        nq = len(qs)
    
        h1e     = fci_obj.h1e
        h2e     = fci_obj.h2e
        ene_fci = fci_obj.fci_ene
        vec_fci = fci_obj.fci_vec
        size    = vec_fci.size

        link_index_ip = _unpack(norb, nelec_ip, None, spin=None)
        size_ip  = link_index_ip[0].shape[0] * link_index_ip[1].shape[0]
        hdiag_ip = fci_obj.make_hdiag(h1e, h2e, norb, nelec_ip)
        assert hdiag_ip.shape == (size_ip, )

        h_ip  = fci.direct_spin1.pspace(h1e, h2e, norb, nelec_ip, hdiag=hdiag_ip, np=self.max_memory)[1]
        assert h_ip.shape == (size_ip, size_ip)

        bps = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ip)
        eqs = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ip)

        def gen_gfn(omega):
            h_ip_omega = h_ip + (omega - ene_fci - 1j * eta) * numpy.eye(size_ip)
            return numpy.einsum("pI,qJ,IJ->pq", bps, eqs, numpy.linalg.inv(h_ip_omega))

        gfns_ip = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)

        broadening = eta; H_fci = h_ip; e_fci = ene_fci
        e_vector   = [fci.addons.des_a(fci_obj.ci, norb, nelec, q).reshape(-1) for q in qs]

        gfns_ip_ref = numpy.zeros((np, nq, nomega), dtype=numpy.complex128)
        for iomega, omega in enumerate(omegas):
            H_fci_inv = numpy.linalg.inv(H_fci + numpy.eye(H_fci.shape[0]) * (omega - e_fci - 1j * broadening))
            for ip, p in enumerate(ps):
                b_vector = fci.addons.des_a(fci_obj.ci, norb, nelec, p).reshape(-1)
                sol = numpy.dot(H_fci_inv, b_vector)
                for iq, q in enumerate(qs):
                    gfns_ip_ref[iq,ip,iomega] = numpy.dot(e_vector[iq], sol)

        for iomega, omega in enumerate(omegas):
            for ip, p in enumerate(ps):
                for iq, q in enumerate(qs):
                    print(f"omega = {omega:6.4f}, p = {p:2d}, q = {q:2d}, gf_ip = {gfns_ip[iomega, ip, iq]: 6.4f}, gf_ip_ref = {gfns_ip_ref[iq, ip, iomega]: 6.4f}")

        assert 1 == 2
        return gfns_ip

FCIGF = DirectSpin1FullConfigurationInteraction

if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(
        atom = 'H 0 0 0; Li 0 0 1.1',
        basis = 'sto3g',
        verbose = 5,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    nmo = len(mf.mo_energy)

    gf = FCIGF(mf)
    gf.build()

    eta    = 0.01
    omegas = numpy.linspace(-0.5, 0.5, 201)
    ps = numpy.arange(nmo)
    qs = numpy.arange(nmo)
    gf_ip = gf.get_ip_slow(omegas, ps=ps, qs=qs, eta=eta)

    import fcdmft
    from fcdmft.solver import fcigf
    print(fcdmft.__file__)
    gf_ref = fcdmft.solver.fcigf.FCIGF(gf._fci_obj, gf._scf_obj, tol=1e-6)

    gf_ip_ref = gf_ref.ipfci_mo_slow(ps, qs, omegas, eta).conj()
    gf_ea_ref = gf_ref.eafci_mo_slow(ps, qs, omegas, eta)

    for iomega, omega in enumerate(omegas):
        for ip, p in enumerate(ps):
            for iq, q in enumerate(qs):
                print(f"omega = {omega:6.4f}, p = {p:2d}, q = {q:2d}, gf_ip = {gf_ip[iomega, ip, iq]: 6.4f}, gf_ip_ref = {gf_ip_ref[iq, ip, iomega]: 6.4f}")
    