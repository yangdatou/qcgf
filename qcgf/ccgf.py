import numpy, scipy

import pyscf
from pyscf import gto, scf, cc
from pyscf.cc.gccsd import amplitudes_to_vector, vector_to_amplitudes
from pyscf.cc.eom_gccsd import amplitudes_to_vector_ip, amplitudes_to_vector_ea
from pyscf.cc.eom_gccsd import vector_to_amplitudes_ip, vector_to_amplitudes_ea

from qcgf.gf import GreenFunctionMixin

def _rhs_1_ip(t1, t2, orb_idx=0):
    p    = orb_idx
    nocc = t1.shape[0]
    nvir = t1.shape[1]

    r1 = None
    if p < nocc:
        i = p
        r1 = numpy.zeros((nocc, ), dtype=t1.dtype)
        r1[i] = 1.0
    else:
        a = p - nocc
        r1 = t1[:, a]
    
    assert r1.shape == (nocc, )
    return r1

def _rhs_2_ip(t1, t2, orb_idx=0):
    p    = orb_idx
    nocc = t1.shape[0]
    nvir = t1.shape[1]

    r2 = None
    if p < nocc:
        i = p
        r2 = numpy.zeros((nocc, nocc, nvir), dtype=t1.dtype)
        r2[i] = 1.0
    else:
        a = p - nocc
        r2 = t2[:, :, a, :]
    
    assert r2.shape == (nocc, nocc, nvir)
    return r2

def _rhs_ip(t1, t2, l1, l2, orb_idx=0):
    '''
    Make the right hand side for orb_idx of the CCGF-IP equations

    Input:
        t1 : numpy.ndarray
            T1 amplitudes
        t2 : numpy.ndarray
            T2 amplitudes
        l1 : numpy.ndarray
            L1 (not used)
        l2 : numpy.ndarray
            L2 (not used)
        orb_idx: int

    Returns:
        rhs_1 : numpy.ndarray
            right hand side
        rhs_2 : numpy.ndarray
            right hand side
    '''

    p    = orb_idx
    nocc = t1.shape[0]
    nvir = t1.shape[1]

    rhs_1 = None
    rhs_2 = None

    if p < nocc:
        i = p

        rhs_1 = numpy.zeros((nocc, ), dtype=t1.dtype)
        rhs_1[i] = 1.0

        rhs_2 = numpy.zeros((nocc, nocc, nvir), dtype=t1.dtype)
        rhs_2[i] = 1.0

    else:
        a = p - nocc

        rhs_1 = t1[:, a]
        rhs_2 = t2[:, :, a, :]

    assert rhs_1 is not None
    assert rhs_2 is not None

    assert rhs_1.shape == (nocc, )
    assert rhs_2.shape == (nocc, nocc, nvir)

    return rhs_1, rhs_2

def _lhs_ip(t1, t2, l1, l2, orb_idx=0):
    '''
    Make the left hand side for orb_idx of the CCGF-IP equations

    Input:
        t1 : numpy.ndarray
            T1 amplitudes
        t2 : numpy.ndarray
            T2 amplitudes
        l1 : numpy.ndarray
            L1
        l2 : numpy.ndarray
            L2
        orb_idx: int

    Returns:
        lhs_1 : numpy.ndarray
            left hand side
        lhs_2 : numpy.ndarray
            left hand side
    '''

    p    = orb_idx
    nocc = t1.shape[0]
    nvir = t1.shape[1]

    lhs_1 = None
    lhs_2 = None

    if p < nocc:
        i = p

        lhs_1 = numpy.zeros((nocc, ), dtype=t1.dtype)
        lhs_1[i] = -1.0
        lhs_1   += numpy.einsum('ia,a->i',     l1, t1[i, :])
        lhs_1   += numpy.einsum('ilcd,lcd->i', l2, t2[i, :, :, :]) * 2.0
        lhs_1   -= numpy.einsum('ilcd,ldc->i', l2, t2[i, :, :, :])

        r2 = numpy.zeros((nocc, nocc, nvir), dtype=t1.dtype)
        r2[i] = 1.0

    else:
        a = p - nocc

        r1 = t1[:, a]
        r2 = t2[:, :, a, :]

    assert r1 is not None
    assert r2 is not None

    assert r1.shape == (nocc, )
    assert r2.shape == (nocc, nocc, nvir)

    return r1, r2



def ccsd_gf_ip(cc_obj, omegas, t1=None, t2=None, l1=None, l2=None, eta=1e-4):
    '''
    Compute IP part of GCCSD-GF

    Input:
        gf_obj: GreenFunctionMixin object

    '''
    num_orb   = norb = 10
    omegas    = numpy.array(omegas)
    num_omega = len(omegas)

    t1 = cc_obj.t1
    t2 = cc_obj.t2
    
    nocc, nvir  = t1.shape
    nov = nocc * nvir
    size_vec    = nov  + nov * (nov + 1) // 2 # single + double 
    size_vec_ip = nocc + nocc * nov # IP single + double
    size_vec_ea = nvir + nvir * nov # EA single + double

    ip_obj  = pyscf.cc.eom_rccsd.EOMIP(cc_obj)
    imds    = ip_obj.make_imds()
    diag    = ip_obj.get_diag()

    zs   = numpy.zeros((size_vec_ip, num_orb))
    lhs  = numpy.zeros((size_vec_ip, num_orb))

    rhs  = _rhs_ip(t1, t2, orb_idx=p)
    assert rhs.shape == (size_vec_ip, num_orb)
    gfs  = numpy.zeros((num_omega, num_orb, num_orb))     # gfs = einsum("wxp,wxq->wpq", lhs, zs)

    for p in range(norb):
        

        for iomega, omega in enumerate(omegas)
            def ax(x):
                return greens_func_multiply(eomip.matvec, vector, curr_omega - 1j * broadening, imds=eomip_imds)

            diag_w = diag + curr_omega-1j*broadening
            x0 = b_vector/diag_w
            solver = gmres.GMRES(matr_multiply, b_vector, x0, diag_w, tol=self.tol)
            cput1 = (time.process_time(), time.perf_counter())
            sol = solver.solve().reshape(-1)
            cput1 = logger.timer(self, 'IPGF GMRES orbital p = %d/%d, freq w = %d/%d (%d iterations)'%(
                ip+1,len(ps),iomega+1,len(omega_list),solver.niter), *cput1)
            x0 = sol
            for iq, q in enumerate(qs):
                gfvals[ip,iq,iomega] = -np.dot(e_vector[iq], sol)
    return gfvals

class CCGF(GreenFunctionMixin):
    def __init__(self):
        pass