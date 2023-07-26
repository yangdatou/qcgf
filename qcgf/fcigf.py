import os, sys, typing
from typing import List, Tuple, Callable

import numpy, scipy

import pyscf
from pyscf import gto, scf, ao2mo
from pyscf import fci, lib

from pyscf.fci import direct_spin1
from pyscf.fci.direct_spin1 import _unpack
from pyscf.fci.direct_spin1 import contract_1e
from pyscf.fci.direct_spin1 import contract_2e

from qcgf.gf  import GreensFunctionMixin
from qcgf.lib import gmres

from pyscf.tools.dump_mat import dump_rec
print_matrix = lambda c, t=None, stdout=sys.stdout: ((print("\n" + t, file=stdout) if t is not None else 0), dump_rec(stdout, c))

class _DirectSpin1FullConfigurationInteraction(direct_spin1.FCISolver):
    h1e = None
    h2e = None

    norb  = None
    nelec = None

    fci_ene = None
    fci_vec = None

class DirectSpin1FullConfigurationInteraction(GreensFunctionMixin):
    def __init__(self, scf_obj: scf.hf.SCF = None, tol=1e-8, verbose=0) -> None:
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

        self.conv_tol   = tol
        self.verbose    = verbose
        self._mol_obj   = scf_obj.mol
        self.stdout     = scf_obj.stdout
        self.max_memory = scf_obj.max_memory

    def build(self, coeff: numpy.ndarray = None) -> None:
        """
        Build and solve the FCI object.
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
        nelec, norb, h1e, h2e = make_h1_and_h2(self._mol_obj, coeff)
        
        fci_obj = _DirectSpin1FullConfigurationInteraction(self._mol_obj)
        fci_ene, fci_vec = fci_obj.kernel(
            h1e=h1e,  eri=h2e, norb=norb, nelec=nelec, 
            ci0=None, ecore=0.0, verbose=self.verbose
            )

        fci_obj.h1e = h1e
        fci_obj.h2e = h2e

        fci_obj.norb  = norb
        fci_obj.nelec = nelec

        fci_obj.fci_ene = fci_ene
        fci_obj.fci_vec = fci_vec

        self._fci_obj = fci_obj

    def get_ip(self, omegas: List[float], ps: (List[int]|None)=None, qs: (List[int]|None)=None, eta: float=0.0, 
                     is_mor: bool=False, omegas_mor: (List[int]|None)=None) -> numpy.ndarray:
        '''
        Compute FCI IP Green's function in MO basis. This is the fast routine 
        without constructing the full FCI Hamiltonian.

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
            is_mor : bool
                Whether to use the method of residue.
            omega_mor : float, optional
                The frequency used for the method of residue.

        Returns:
            gfns_ip : 3D array of complex floats, shape (nomega, np, nq)
                The computed Green's function values.
        '''
        scf_obj = self._scf_obj

        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
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
    
        ene_fci = fci_obj.fci_ene
        vec_fci = fci_obj.fci_vec
        size    = vec_fci.size

        link_index_ip = _unpack(norb, nelec_ip, None, spin=None)
        na_ip    = link_index_ip[0].shape[0]
        nb_ip    = link_index_ip[1].shape[0]
        size_ip  = na_ip * nb_ip
        hdiag_ip = fci_obj.make_hdiag(fci_obj.h1e, fci_obj.h2e, norb, nelec_ip)
        assert hdiag_ip.shape == (size_ip, )

        h2e_ip = fci_obj.absorb_h1e(fci_obj.h1e, fci_obj.h2e, norb, nelec_ip, 0.5)
        bps    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ip)
        eqs    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ip)

        if is_mor:
            nmor = len(omegas_mor)
            omegas_mor = numpy.asarray(omegas_mor)
            omegas_comps = omegas_mor
        else:
            omegas_comps = omegas

        gfns_ip = numpy.zeros((nomega, np, nq), dtype=numpy.complex128)

        for ip, p in enumerate(ps):
            b_p = bps[ip]

            for iomega, omega in enumerate(omegas_comps):

                def h_ip_omega(v):
                    assert v.shape == (size_ip, )
                    v        = v.reshape(na_ip, nb_ip)
                    hv_real  = contract_2e(h2e_ip, v.real, norb, nelec_ip, link_index=link_index_ip)
                    hv_imag  = contract_2e(h2e_ip, v.imag, norb, nelec_ip, link_index=link_index_ip)

                    hv  = hv_real + 1j * hv_imag 
                    hv += (omega - ene_fci - 1j * eta) * v

                    return hv.reshape(size_ip, )

                hdiag_ip_omega = hdiag_ip + omega - ene_fci - 1j * eta
                x_p = gmres(h_ip_omega, b=b_p, x0=b_p / hdiag_ip_omega, diag=hdiag_ip_omega, 
                            tol=1e-10, max_cycle=self.max_cycle, m=self.gmres_m, 
                            verbose=self.verbose, stdout=self.stdout)
                x_p = x_p.reshape(-1)

                for iq, q in enumerate(qs):
                    e_q = eqs[iq]
                    gfns_ip[iomega, ip, iq] = numpy.dot(e_q, x_p)
        
        return gfns_ip

    def get_ip_slow(self, omegas: List[float], ps: (List[int]|None)=None, qs: (List[int]|None)=None, eta: float=0.0) -> numpy.ndarray:
        '''
        Slow version of computing FCI IP Green's function by constructing 
        the full FCI Hamiltonian. Note that the ene_fci does not include
        the nuclear repulsion energy.

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
            gfns_ip : 3D array of complex floats, shape (nomega, np, nq)
                The computed Green's function values.
        '''
        scf_obj = self._scf_obj

        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
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
            return numpy.einsum("pI,qJ,JI->qp", bps, eqs, numpy.linalg.inv(h_ip_omega))

        gfns_ip = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        return gfns_ip


    def get_ea_slow(self, omegas: List[float], ps: (List[int]|None)=None, qs: (List[int]|None)=None, eta: float=0.0) -> numpy.ndarray:
        '''
        Slow version of computing FCI EA Green's function by constructing 
        the full FCI Hamiltonian. Note that the ene_fci does not include
        the nuclear repulsion energy.

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
            gfns_ea : 3D array of complex floats, shape (nomega, np, nq)
                The computed Green's function values.
        '''
        scf_obj = self._scf_obj

        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        nelec_ea = (nelec[0] + 1, nelec[1])

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

        link_index_ea = _unpack(norb, nelec_ea, None, spin=None)
        size_ea  = link_index_ea[0].shape[0] * link_index_ea[1].shape[0]
        hdiag_ea = fci_obj.make_hdiag(h1e, h2e, norb, nelec_ea)
        assert hdiag_ea.shape == (size_ea, )

        h_ea  = fci.direct_spin1.pspace(h1e, h2e, norb, nelec_ea, hdiag=hdiag_ea, np=self.max_memory)[1]
        assert h_ea.shape == (size_ea, size_ea)

        bps = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ea)
        eqs = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ea)

        def gen_gfn(omega):
            h_ea_omega = - h_ea + (omega + ene_fci + 1j * eta) * numpy.eye(size_ea)
            return numpy.einsum("pI,qJ,JI->qp", bps, eqs, numpy.linalg.inv(h_ea_omega))

        gfns_ea = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        return gfns_ea

FCIGF = DirectSpin1FullConfigurationInteraction

if __name__ == '__main__':
    from pyscf import gto, scf
    mol = gto.M(
        atom = 'H 0 0 0; Li 0 0 1.1',
        basis = 'sto3g',
        verbose = 0,
    )
    mf = scf.RHF(mol)
    mf.kernel()
    nmo = len(mf.mo_energy)

    gf = FCIGF(mf, tol=1e-8, verbose=4)
    gf.build()

    eta    = 0.01
    omegas = numpy.linspace(-0.5, 0.5, 201)
    ps = numpy.arange(nmo)
    qs = numpy.arange(nmo)

    import fcdmft.solver.fcigf
    gf._scf = gf._scf_obj
    gf._fci = gf._fci_obj
    gf.tol  = gf.conv_tol

    gf_ip_1     = gf.get_ip_slow(omegas, ps=ps, qs=qs, eta=eta)
    gf_ip_2     = gf.get_ip(omegas, ps=ps, qs=qs, eta=eta)
    gf_ip_ref_1 = fcdmft.solver.fcigf.FCIGF.ipfci_mo_slow(gf, ps, qs, omegas, eta).transpose(2, 0, 1)
    gf_ip_ref_2 = fcdmft.solver.fcigf.FCIGF.ipfci_mo(gf, ps, qs, omegas, eta).transpose(2, 1, 0)

    err_1 = numpy.linalg.norm(gf_ip_1 - gf_ip_ref_1)
    err_2 = numpy.linalg.norm(gf_ip_1 - gf_ip_ref_2)
    print(f"IP GF difference = {err_1:6.4e}")
    print(f"IP GF difference = {err_2:6.4e}")

    err_1 = numpy.linalg.norm(gf_ip_2 - gf_ip_ref_1)
    err_2 = numpy.linalg.norm(gf_ip_2 - gf_ip_ref_2)
    print(f"IP GF difference = {err_1:6.4e}")
    print(f"IP GF difference = {err_2:6.4e}")

    # gf_ea      = gf.get_ea_slow(omegas, ps=ps, qs=qs, eta=eta)
    # gf_ea_ref  = fcdmft.solver.fcigf.FCIGF.eafci_mo_slow(gf, ps, qs, omegas, eta).transpose(2, 0, 1)
    # gf_ea_diff = numpy.linalg.norm(gf_ea - gf_ea_ref)
    # print(f"ea GF difference = {gf_ea_diff:6.4e}")

    # for iomega, omega in enumerate(omegas):
    #     print("")
    #     print(f"omega = {omega:6.4f}")
    #     print("IP:")
    #     print_matrix(gf_ip_1[iomega].real)
    #     print_matrix(gf_ip_2[iomega].real)


    