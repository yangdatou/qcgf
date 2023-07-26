import inspect

from pyscf import fci
from pyscf.fci.direct_spin1 import absorb_h1e

def make_h1_and_h2(mol_obj=None, coeff_mo=None):
    fci_obj  = fci.FCI(mol_obj, mo=coeff_mo, singlet=False)
    
    kwargs = inspect.signature(fci_obj.kernel).parameters
    norb   = kwargs.get('norb').default
    norb2  = norb * (norb + 1) // 2
    nelec  = kwargs.get('nelec').default

    h1e = kwargs.get('h1e'  ).default
    eri = kwargs.get('eri'  ).default
    assert h1e.shape == (norb, norb)
    assert eri.shape == (norb2, norb2)

    return nelec, norb, h1e, eri