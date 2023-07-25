
from pyscf import fci

def make_h1_and_h2(mol_obj=None, coeff_mo=None):
    fci_obj  = fci.FCI(mol_obj, mo=coeff_mo, singlet=False)
    
    kwargs = inspect.signature(fci_obj.kernel).parameters
    h1e    = kwargs.get('h1e'  ).default
    h2e    = kwargs.get('eri'  ).default

    norb   = kwargs.get('norb').default
    norb2  = norb * (norb + 1) // 2
    nelec  = kwargs.get('nelec').default

    assert h1e.shape == (norb, norb)
    assert h2e.shape == (norb2, norb2)

    return nelec, norb, h1e, h2e