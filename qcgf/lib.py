import sys, os, typing
from typing import Union, Optional, Callable, Tuple, List

import numpy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gcrotmk

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

def gmres(h: Union[Callable, numpy.ndarray], b: numpy.ndarray, 
          x0: Optional[numpy.ndarray] = None,
          diag: Optional[numpy.ndarray] = None, tol: float = 1e-3, 
          max_cycle: int = 200, m: int = 30,
          stdout: typing.TextIO = sys.stdout) -> numpy.ndarray:
    """Implements the Generalized Minimum Residual (GMRES) method.

    Solves the linear equation h x = b using GMRES. GMRES is an iterative method
    for the numerical solution of a nonsymmetric system of linear equations. The
    method approximates the solution by the vector in a Krylov subspace with minimal 
    residual.

    Args:
        h: can be either a callable or a matrix. Used to initialize the LinearOperator.
        b: Vector or list of vectors.

    Kwargs:
        x0 : 1D array
            Initial guess
        diag: 1D array
            Diagonal preconditioner.
        tol : float
            Tolerance to terminate the operation aop(x).
        max_cycle : int
            max number of iterations.

    Returns:
        x: Solution vector, ndarray like b.
    """
    b = numpy.asarray(b)
    n = max(b.shape)
    if callable(h):
        hop  = LinearOperator((n, n), matvec=h)
    else:
        assert h.shape == (n, n)
        hop = LinearOperator((n, n), matvec=(lambda x: numpy.dot(h, x)))
    b    = b.reshape(-1)
    x0   = x0.reshape(-1)

    mop = None
    if diag is not None:
        diag = diag.reshape(-1)
        mop  = LinearOperator((n, n), matvec=(lambda x: x / diag))

    num_iter = 0
    def callback(rk):
        nonlocal num_iter
        num_iter += 1
        stdout.write(f"GMRES: iter = {num_iter:4d}, residual = {numpy.linalg.norm(rk):6.4e}\n")

    x, info = gcrotmk(hop, b, x0=x0, M=mop, maxiter=max_cycle, callback=callback, m=m, tol=tol, atol=tol)
    x = x.reshape(-1)

    if info > 0:
        raise ValueError(f"Convergence to tolerance not achieved in {info} iterations")

    return x