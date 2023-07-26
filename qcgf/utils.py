import inspect
from pyscf import fci


def make_h1_and_h2(molecule=None, coefficients=None):
    fci_object = fci.FCI(molecule, mo=coefficients, singlet=False)
    
    # Get the parameters of the fci.kernel function
    kwargs = inspect.signature(fci_object.kernel).parameters
    
    # Get the default values of h1e and h2e
    h1e_default = kwargs.get('h1e').default
    h2e_default = kwargs.get('eri').default
    
    # Get the default values of norb and nelec from the FCI object
    norb = kwargs.get('norb').default
    norb2 = norb * (norb + 1) // 2
    nelec = kwargs.get('nelec').default
    
    # Check the shapes of h1e and h2e
    assert h1e_default.shape == (norb, norb)
    assert h2e_default.shape == (norb2, norb2)
    
    return nelec, norb, h1e_default, h2e_default