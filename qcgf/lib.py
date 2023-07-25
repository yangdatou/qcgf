import sys, os, typing
from typing import Union, Optional, Callable, Tuple, List

import numpy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gcrotmk

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

OptionalArray = Optional[numpy.ndarray]

def _unpack(v=None, vs=None):   
    res = None

    if v is not None and vs is None:
        v   = numpy.asarray(v)
        res = v.reshape(1, -1) if v.ndim == 1 else None
    
    if vs is not None and v is None:
        vs  = numpy.asarray(vs)
        res = vs if vs.ndim == 2 else vs.reshape(1, -1) if vs.ndim == 1 else None

    return res

def gmres(h: (Callable|numpy.ndarray), 
          bs:  OptionalArray = None, b:  OptionalArray = None,
          xs0: OptionalArray = None, x0: OptionalArray = None,
          diag: OptionalArray = None,  
          m: int = 30, tol: float = 1e-6, max_cycle: int = 200,
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
        xs0 : 1D array
            Initial guess.
        diag: 1D array
            Diagonal preconditioner.
        tol : float
            Tolerance to terminate the operation aop(x).
        max_cycle : int
            max number of iterations.

    Returns:
        xs: Solution vector, ndarray like b.
    """
    bs  = _unpack(b, bs)
    x0s = _unpack(x0, xs0)
    
    assert bs is not None
    nb, n = bs.shape
    nnb   = nb * n

    assert diag.shape == (n, )

    if callable(h):
        def matvec(xs):
            xs  = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        hop = LinearOperator((nnb, nnb), matvec=matvec)
    else:
        def matvec(xs):
            xs  = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h.dot(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        assert h.shape == (n, n)
        hop = LinearOperator((nnb, nnb), matvec=matvec)

    mop = None
    if diag is not None:
        diag = diag.reshape(-1)
        def matvec(xs):
            xs  = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([x / diag for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )
        mop  = LinearOperator((nnb, nnb), matvec=matvec)

    num_iter = 0
    def callback(rk):
        nonlocal num_iter
        num_iter += 1
        stdout.write(f"GMRES: iter = {num_iter:4d}, residual = {numpy.linalg.norm(rk)/nb:6.4e}\n")

    stdout.write(f"\nGMRES Start\n")
    stdout.write(f"GMRES: tol = {tol:6.4e}, max_cycle = {max_cycle:4d}, m = {m:4d}\n")
    if xs0:
        stdout.write(f"GMRES: initial residual = {numpy.linalg.norm(hop(xs0)-bs)/nb:6.4e}\n")
        
    xs, info = gcrotmk(
        hop, bs.reshape(-1), x0=xs0.reshape(-1), M=mop, 
        maxiter=max_cycle, callback=callback, m=m, 
        tol=tol/nb, atol=tol/nb
        )

    if info > 0:
        raise ValueError(f"Convergence to tolerance not achieved in {info} iterations")

    if nb == 1:
        xs = xs.reshape(n, )
    else:
        xs = xs.reshape(nb, n)

    return xs