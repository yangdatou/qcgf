import os, sys, typing, functools
from functools import reduce
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
    eri = None

    norb  = None
    nelec = None

    ene_fci = None
    vec_fci = None

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
        # Initialize the problem
        log = lib.logger.new_logger(self, self.verbose)

        # Get the scf and fci objects, extract the information
        scf_obj = self._scf_obj
        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        h1e, eri = fci_obj.h1e, fci_obj.eri
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0] + 1, nelec[1])

        ene_fci = fci_obj.ene_fci
        vec_fci = fci_obj.vec_fci
        size    = vec_fci.size

        # Extract the information about the frequency and orbitals
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
    
        # Set up the IP manybody Hamiltonian
        link_index_ip = _unpack(norb, nelec_ip, None, spin=None)
        na_ip    = link_index_ip[0].shape[0]
        nb_ip    = link_index_ip[1].shape[0]
        size_ip  = na_ip * nb_ip
        hdiag_ip = fci_obj.make_hdiag(h1e, eri, norb, nelec_ip)
        assert hdiag_ip.shape == (size_ip, )

        # Build the RHS and LHS of the response equation
        bps    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ip)
        eqs    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ip)

        h2e_ip = fci_obj.absorb_h1e(h1e, eri, norb, nelec_ip, fac=0.5)
        # if size_ip * size_ip * 8 / 1024**3 > self.max_memory:
        #     raise ValueError("Not enough memory for FCI IP Hamiltonian.")
        # h_ip   = fci.direct_spin1.pspace(h1e, eri, norb, nelec_ip, hdiag=hdiag_ip, np=size_ip)[1]
        # assert h_ip.shape == (size_ip, size_ip)

        assert not is_mor

        def gen_gfn(omega):
            omega_e0_eta_ip = omega - ene_fci - 1j * eta
            hdiag_ip_omega  = hdiag_ip + omega_e0_eta_ip

            def h_ip_omega(v):
                assert v.shape == (size_ip, )
                v        = v.reshape(na_ip, nb_ip)
                hv_real  = contract_2e(h2e_ip, v.real, norb, nelec_ip, link_index=link_index_ip)
                hv_imag  = contract_2e(h2e_ip, v.imag, norb, nelec_ip, link_index=link_index_ip)

                hv  = hv_real + 1j * hv_imag 
                hv += omega_e0_eta_ip * v

                return hv.reshape(size_ip, )

            xps = gmres(h_ip_omega, bs=bps, xs0=bps / hdiag_ip_omega, diag=hdiag_ip_omega, 
                        tol=self.conv_tol, max_cycle=self.max_cycle, m=self.gmres_m, 
                        verbose=self.verbose, stdout=self.stdout)
            xps = xps.reshape(np, size_ip)
            return numpy.dot(xps, eqs.T)

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        gfns_ip = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        lib.logger.timer(self, 'Solving GF-IP', *cput0)
        return gfns_ip

    def get_ea(self, omegas: List[float], ps: (List[int]|None)=None, qs: (List[int]|None)=None, eta: float=0.0, 
                     is_mor: bool=False, omegas_mor: (List[int]|None)=None) -> numpy.ndarray:
        '''
        Compute FCI EA Green's function in MO basis. This is the fast routine 
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
            gfns_ea : 3D array of complex floats, shape (nomega, np, nq)
                The computed Green's function values.
        '''
        # Initialize the problem
        log = lib.logger.new_logger(self, self.verbose)

        # Get the scf and fci objects, extract the information
        scf_obj = self._scf_obj
        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        h1e, eri = fci_obj.h1e, fci_obj.eri
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0] + 1, nelec[1])

        ene_fci = fci_obj.ene_fci
        vec_fci = fci_obj.vec_fci
        size    = vec_fci.size

        # Extract the information about the frequency and orbitals
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

        # Set up the EA manybody Hamiltonian
        link_index_ea = _unpack(norb, nelec_ea, None, spin=None)
        na_ea    = link_index_ea[0].shape[0]
        nb_ea    = link_index_ea[1].shape[0]
        size_ea  = na_ea * nb_ea
        hdiag_ea = fci_obj.make_hdiag(h1e, eri, norb, nelec_ea)
        assert hdiag_ea.shape == (size_ea, )

        bps = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ea)
        eqs = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ea)

        h2e_ea = fci_obj.absorb_h1e(h1e, eri, norb, nelec_ea, fac=0.5)
        # if size_ea * size_ea * 8 / 1024**3 > self.max_memory:
        #     raise ValueError("Not enough memory for FCI ea Hamiltonian.")
        # h_ea   = fci.direct_spin1.pspace(h1e, eri, norb, nelec_ea, hdiag=hdiag_ea, np=size_ea)[1]
        # assert h_ea.shape == (size_ea, size_ea)

        assert not is_mor

        def gen_gfn(omega):
            omega_e0_eta_ea = omega + ene_fci + 1j * eta
            hdiag_ea_omega  = - hdiag_ea + omega_e0_eta_ea

            def h_ea_omega(v):
                assert v.shape == (size_ea, )
                v        = v.reshape(na_ea, nb_ea)
                hv_real  = contract_2e(h2e_ea, v.real, norb, nelec_ea, link_index=link_index_ea)
                hv_imag  = contract_2e(h2e_ea, v.imag, norb, nelec_ea, link_index=link_index_ea)

                hv  = - (hv_real + 1j * hv_imag)
                hv += omega_e0_eta_ea * v

                return hv.reshape(size_ea, )

            xps = gmres(h_ea_omega, bs=bps, xs0=bps / hdiag_ea_omega, diag=hdiag_ea_omega, 
                        tol=self.conv_tol, max_cycle=self.max_cycle, m=self.gmres_m, 
                        verbose=self.verbose, stdout=self.stdout)
            xps = xps.reshape(np, size_ea)
            return numpy.dot(xps, eqs.T)

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        gfns_ea = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        lib.logger.timer(self, 'Solving GF-EA', *cput0)
        return gfns_ea

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
        # Initialize the problem
        log = lib.logger.new_logger(self, self.verbose)

        # Get the scf and fci objects, extract the information
        scf_obj = self._scf_obj
        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        h1e, eri = fci_obj.h1e, fci_obj.eri
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0] + 1, nelec[1])

        ene_fci = fci_obj.ene_fci
        vec_fci = fci_obj.vec_fci
        size    = vec_fci.size

        # Extract the information about the frequency and orbitals
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
    
        # Set up the IP manybody Hamiltonian
        link_index_ip = _unpack(norb, nelec_ip, None, spin=None)
        na_ip    = link_index_ip[0].shape[0]
        nb_ip    = link_index_ip[1].shape[0]
        size_ip  = na_ip * nb_ip
        hdiag_ip = fci_obj.make_hdiag(h1e, eri, norb, nelec_ip)
        assert hdiag_ip.shape == (size_ip, )

        # Build the RHS and LHS of the response equation
        bps    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ip)
        eqs    = numpy.asarray([fci.addons.des_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ip)

        # h2e_ip = fci_obj.absorb_h1e(h1e, eri, norb, nelec_ip, fac=0.5)
        if size_ip * size_ip * 8 / 1024**3 > self.max_memory:
            raise ValueError("Not enough memory for FCI IP Hamiltonian.")
        h_ip   = fci.direct_spin1.pspace(h1e, eri, norb, nelec_ip, hdiag=hdiag_ip, np=size_ip)[1]
        assert h_ip.shape == (size_ip, size_ip)

        def gen_gfn(omega):
            omega_e0_eta_ip = omega - ene_fci - 1j * eta
            h_ip_omega      = h_ip + omega_e0_eta_ip * numpy.eye(size_ip)
            # h_ip_omega_inv = numpy.linalg.inv(h_ip_omega)
            # xps = numpy.dot(h_ip_omega_inv, bps.T)
            xps = numpy.linalg.solve(h_ip_omega, bps.T).T
            return numpy.dot(xps, eqs.T)

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        gfns_ip = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        lib.logger.timer(self, 'Solving GF-IP', *cput0)
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
        # Initialize the problem
        log = lib.logger.new_logger(self, self.verbose)

        # Get the scf and fci objects, extract the information
        scf_obj = self._scf_obj
        if not isinstance(self._fci_obj, _DirectSpin1FullConfigurationInteraction):
            self.build()
        fci_obj = self._fci_obj

        norb     = fci_obj.norb
        nelec    = fci_obj.nelec
        h1e, eri = fci_obj.h1e, fci_obj.eri
        nelec_ip = (nelec[0] - 1, nelec[1])
        nelec_ea = (nelec[0] + 1, nelec[1])

        ene_fci = fci_obj.ene_fci
        vec_fci = fci_obj.vec_fci
        size    = vec_fci.size

        # Extract the information about the frequency and orbitals
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

        # Set up the EA manybody Hamiltonian
        link_index_ea = _unpack(norb, nelec_ea, None, spin=None)
        na_ea    = link_index_ea[0].shape[0]
        nb_ea    = link_index_ea[1].shape[0]
        size_ea  = na_ea * nb_ea
        hdiag_ea = fci_obj.make_hdiag(h1e, eri, norb, nelec_ea)
        assert hdiag_ea.shape == (size_ea, )

        bps = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, p).reshape(-1) for p in ps]).reshape(np, size_ea)
        eqs = numpy.asarray([fci.addons.cre_a(vec_fci, norb, nelec, q).reshape(-1) for q in qs]).reshape(nq, size_ea)

        # h2e_ea = fci_obj.absorb_h1e(h1e, eri, norb, nelec_ea, fac=0.5)
        if size_ea * size_ea * 8 / 1024**3 > self.max_memory:
            raise ValueError("Not enough memory for FCI ea Hamiltonian.")
        h_ea   = fci.direct_spin1.pspace(h1e, eri, norb, nelec_ea, hdiag=hdiag_ea, np=size_ea)[1]
        assert h_ea.shape == (size_ea, size_ea)

        def gen_gfn(omega):
            omega_e0_eta_ea = omega + ene_fci + 1j * eta
            h_ea_omega      = - h_ea + omega_e0_eta_ea * numpy.eye(size_ea)
            # h_ea_omega_inv = numpy.linalg.inv(h_ea_omega)
            # xps = numpy.dot(h_ea_omega_inv, bps.T)
            xps = numpy.linalg.solve(h_ea_omega, bps.T).T
            return numpy.dot(xps, eqs.T)

        cput0 = (lib.logger.process_clock(), lib.logger.perf_counter())
        gfns_ea = numpy.asarray([gen_gfn(omega) for omega in omegas]).reshape(nomega, np, nq)
        lib.logger.timer(self, 'Solving GF-EA', *cput0)
        return gfns_ea

FCIGF = DirectSpin1FullConfigurationInteraction

if __name__ == "__main__":
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
    ps = [0, 1, 2]
    qs = [0, 1, 2]

    import fcdmft.solver.fcigf
    gf._scf = gf._scf_obj
    gf._fci = gf._fci_obj
    gf.tol  = gf.conv_tol

    so = lib.StreamObject()
    so.verbose = 5
    so.stdout  = sys.stdout

    gf_ip_1     = gf.get_ip_slow(omegas, ps=ps, qs=qs, eta=eta)
    gf_ip_2     = gf.get_ip(omegas, ps=ps, qs=qs, eta=eta)

    err_1 = numpy.linalg.norm(gf_ip_1 - gf_ip_ref_1)
    err_2 = numpy.linalg.norm(gf_ip_1 - gf_ip_ref_2)
    assert err_1 < 1e-6
    assert err_2 < 1e-6

    err_1 = numpy.linalg.norm(gf_ip_2 - gf_ip_ref_1)
    err_2 = numpy.linalg.norm(gf_ip_2 - gf_ip_ref_2)
    assert err_1 < 1e-6
    assert err_2 < 1e-6

    gf_ea_1     = gf.get_ea_slow(omegas, ps=ps, qs=qs, eta=eta)
    cpu1 = lib.logger.timer(so, 'gf.get_ea_slow', *cpu0)
    gf_ea_2     = gf.get_ea(omegas, ps=ps, qs=qs, eta=eta)
    cpu1 = lib.logger.timer(so, 'gf.get_ea', *cpu1)
    gf_ea_ref_1 = fcdmft.solver.fcigf.FCIGF.eafci_mo_slow(gf, ps, qs, omegas, eta).transpose(2, 0, 1)
    cpu1 = lib.logger.timer(so, 'fcdmft.solver.fcigf.FCIGF.eafci_mo_slow', *cpu1)
    gf_ea_ref_2 = fcdmft.solver.fcigf.FCIGF.eafci_mo(gf, ps, qs, omegas, eta).transpose(2, 1, 0)
    cpu1 = lib.logger.timer(so, 'fcdmft.solver.fcigf.FCIGF.eafci_mo', *cpu1)

    err_1 = numpy.linalg.norm(gf_ea_1 - gf_ea_ref_1)
    err_2 = numpy.linalg.norm(gf_ea_1 - gf_ea_ref_2)
    print(f"EA GF difference = {err_1:6.4e}")
    print(f"EA GF difference = {err_2:6.4e}")

    err_1 = numpy.linalg.norm(gf_ea_2 - gf_ea_ref_1)
    err_2 = numpy.linalg.norm(gf_ea_2 - gf_ea_ref_2)
    print(f"EA GF difference = {err_1:6.4e}")
    print(f"EA GF difference = {err_2:6.4e}")
