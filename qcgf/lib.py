import sys
import os
import typing
import numpy
from typing import Union, Optional, Callable, Tuple, List
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gcrotmk
from pyscf.lib import logger

'''
GMRES/GCROT(m,k) for solving Green's function linear equations
'''

OptionalArray = Optional[numpy.ndarray]

def _check_input_vectors(v: OptionalArray = None, vs: OptionalArray = None) -> OptionalArray:
    """
    Check the input vectors to ensure they have the correct shape.

    Args:
        v (OptionalArray): The vector v.
        vs (OptionalArray): The vector vs.

    Returns:
        OptionalArray: The reshaped vectors v or vs if they are 1D, otherwise None.
    """
    res = None

    if v is not None and vs is None:
        v = numpy.asarray(v)
        res = v.reshape(1, -1) if v.ndim == 1 else None

    if vs is not None and v is None:
        vs = numpy.asarray(vs)
        res = vs if vs.ndim == 2 else vs.reshape(1, -1) if vs.ndim == 1 else None

    return res

def gmres(h: Union[Callable, numpy.ndarray], 
          bs: OptionalArray = None, b: OptionalArray = None,
          xs0: OptionalArray = None, x0: OptionalArray = None,
          diag: OptionalArray = None,  
          m: int = 30, tol: float = 1e-6, max_cycle: int = 200,
          verbose: int = 0, stdout: typing.TextIO = sys.stdout) -> numpy.ndarray:
    """
    Solve a matrix equation using the flexible GCROT(m,k) algorithm.

    Solves the linear equation h @ x = b using GMRES. GMRES is an iterative method
    for the numerical solution of a nonsymmetric system of linear equations. The
    method approximates the solution by the vector in a Krylov subspace with minimal 
    residual.

    Args:
        h (Union[Callable, numpy.ndarray]): Can be either a callable or a matrix. Used to initialize the LinearOperator.
        bs (OptionalArray): Vector or list of vectors.
        b (OptionalArray): Vector or list of vectors.

    Kwargs:
        xs0 (OptionalArray): 1D array
            Initial guess.
        diag (OptionalArray): 1D array
            Diagonal preconditioner.
        tol (float): Tolerance to terminate the operation aop(x).
        max_cycle (int): Max number of iterations.

    Returns:
        numpy.ndarray: Solution vector, ndarray like b.
    """
    log = logger.Logger(stdout, verbose)
    bs = _check_input_vectors(b, bs)
    x0s = _check_input_vectors(x0, xs0)
    
    assert bs is not None
    nb, n = bs.shape
    nnb = nb * n

    assert diag.shape == (n, )

    if callable(h):
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h(x) for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        hop = LinearOperator((nnb, nnb), matvec=matvec)
    else:
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([h @ x for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )

        assert h.shape == (n, n)
        hop = LinearOperator((nnb, nnb), matvec=matvec)

    mop = None
    if diag is not None:
        diag = diag.reshape(-1)
        def matvec(xs):
            xs = numpy.asarray(xs).reshape(nb, n)
            hxs = numpy.asarray([x / diag for x in xs]).reshape(nb, n)
            return hxs.reshape(nnb, )
        mop = LinearOperator((nnb, nnb), matvec=matvec)

    num_iter = 0
    def callback(rk):
        nonlocal num_iter
        num_iter += 1
        log.info(f"GMRES: iter = {num_iter:4d}, residual = {numpy.linalg.norm(rk)/nb:6.4e}")

    log.info(f"\nGMRES Start")
    log.info(f"GMRES: nb  = {nb:4d}, n = {n:4d},  m = {m:4d}")
    log.info(f"GMRES: tol = {tol:4.2e}, max_cycle = {max_cycle:4d}")
    
    if xs0 is not None:
        xs0 = xs0.reshape(-1)

    xs, info = gcrotmk(
        hop, bs.reshape(-1), x0=xs0, M=mop, 
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