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

    nelec = None
    norb  = None

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
            _cc_obj : FCI class object
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
        
        self._scf_obj = scf_obj  # Fix typo: mf -> scf_obj
        self.conv_tol   = tol
        self.verbose    = verbose
        self._mol_obj   = scf_obj.mol
        self.stdout     = scf_obj.stdout
        self.max_memory = scf_obj.max_memory

    def build(self):
        raise NotImplementedError

    def get_ip(self):
        raise NotImplementedError

    def get_ea(self):
        raise NotImplementedError

    def get_rhs_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_rhs_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lag_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def get_lag_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def gen_hop_ip(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def gen_hop_ea(self, orb_list=None, verbose=None):
        raise NotImplementedError

    def kernel(self):
        raise NotImplementedError

GF = GreensFunctionMixin