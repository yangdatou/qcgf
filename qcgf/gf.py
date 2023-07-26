import pyscf
from pyscf import lib
from pyscf import __config__

class GreensFunctionMixin(lib.StreamObject):
    '''Green's function base class

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        max_memory : float or int
            Allowed memory in MB.  Default equals to :class:`Mole.max_memory`
        conv_tol : float
            converge threshold.  Default is 1e-9
        max_cycle : int
            max number of iterations.  If max_cycle <= 0, SCF iteration will
            be skiped and the kernel function will compute only the total
            energy based on the intial guess. Default value is 50.
        gmres_m : int
            m used in GMRES method.
    '''
    verbose    = getattr(__config__, 'gf_verbose',  4)
    max_memory = getattr(__config__, 'gf_max_memory', 40000)  # 2 GB
    conv_tol   = getattr(__config__, 'gf_conv_tol', 1e-6)
    max_cycle  = getattr(__config__, 'gf_max_cycle', 50)
    gmres_m    = getattr(__config__, 'gf_gmres_m', 30)

GF = GreensFunctionMixin