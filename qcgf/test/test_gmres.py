import unittest, numpy
from qcgf.lib import gmres
from qcgf.lib import _unpack

TOL = 1e-8

def set_up_matrix(size, p=1.0):
    # Preallocate a complex matrix and diagonal vector
    a = numpy.zeros((size, size), dtype=numpy.complex128)
    d = numpy.zeros((size, ),     dtype=numpy.complex128)

    # Create a random number array for the diagonal
    random_numbers = 30.0 * numpy.random.random(size)

    # Iterate over the diagonal elements of the matrix
    for i in range(size):
        # Compute the complex value for the diagonal
        complex_value = 1.0 + 6.0j + p + random_numbers[i]
        
        # Set the diagonal elements
        d[i] = a[i, i] = complex_value

        # Set the off-diagonal elements
        if i + 2 < size:
            a[i, i + 2] = 1.0
        if i + 3 < size:
            a[i, i + 3] = 0.7
        if i + 1 < size:
            a[i + 1, i] = 3.0j

    return a, d

class TestGMRES(unittest.TestCase):
    def test_unpack(self):
        nb = 10
        n  = 100
        
        b1 = numpy.random.rand(n)
        b2 = numpy.random.rand(nb, n)
        b3 = numpy.random.rand(nb, n, n)

        assert _unpack(None, None) is None
        assert _unpack(b1, b2)     is None

        assert _unpack(b2, None)   is None
        assert _unpack(None, b3)   is None

        assert _unpack(b1, None).shape == (1, n)
        assert _unpack(None, b1).shape == (1, n)
        assert _unpack(None, b2).shape == (nb, n)
        
    def test_gmres_1(self):
        nn = 1000
        f  = 1.0
        p  = 0.0+1j*0.0

        a, d  = set_up_matrix(nn)
        b     = numpy.random.rand(nn) + 1j*numpy.random.rand(nn)
        b    /= numpy.linalg.norm(b)
        x0    = numpy.dot(numpy.linalg.inv(a), b)
        x0   += numpy.random.rand(nn) + 1j*numpy.random.rand(nn)

        x_ref = numpy.dot(numpy.linalg.inv(a), b)

        x_sol = gmres(a, b=b, x0=x0, diag=d, tol=TOL, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b
        assert numpy.linalg.norm(x_sol - x_ref) < TOL
        assert numpy.linalg.norm(r_sol)         < TOL

        x_sol = gmres(lambda x: numpy.dot(a, x), b=b, x0=x0, diag=d, tol=TOL, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b
        assert numpy.linalg.norm(x_sol - x_ref) < TOL
        assert numpy.linalg.norm(r_sol)         < TOL

        x_sol = gmres(a, b=b, x0=None, diag=d, tol=TOL, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b
        assert numpy.linalg.norm(x_sol - x_ref) < TOL
        assert numpy.linalg.norm(r_sol)         < TOL

        x_sol = gmres(lambda x: numpy.dot(a, x), b=b, x0=None, diag=d, tol=TOL, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b
        assert numpy.linalg.norm(x_sol - x_ref) < TOL
        assert numpy.linalg.norm(r_sol)         < TOL


    def test_gmres_2(self):
        nb = 10
        nn = 1000
        f  = 1.0
        p  = 0.0+1j*0.0

        a, d   = set_up_matrix(nn)
        bs     = [(lambda b: b / numpy.linalg.norm(b))(numpy.random.rand(nn) + 1j*numpy.random.rand(nn)) for i in range(nb)]
        xs_ref = numpy.asarray([numpy.dot(numpy.linalg.inv(a), b) for b in bs])
        xs0    = xs_ref + numpy.random.rand(nb, nn) + 1j*numpy.random.rand(nb, nn)

        xs_sol = gmres(a, bs=bs, xs0=xs0, diag=d, tol=TOL, max_cycle=200, m=30)
        rs_sol = numpy.asarray([numpy.dot(a, x_sol) - b for x_sol, b in zip(xs_sol, bs)])
        assert numpy.linalg.norm(xs_sol - xs_ref) < TOL
        assert numpy.linalg.norm(rs_sol)          < TOL

        xs_sol = gmres(lambda x: numpy.dot(a, x), bs=bs, xs0=xs0, diag=d, tol=TOL, max_cycle=200, m=30)
        rs_sol = numpy.asarray([numpy.dot(a, x_sol) - b for x_sol, b in zip(xs_sol, bs)])
        assert numpy.linalg.norm(xs_sol - xs_ref) < TOL
        assert numpy.linalg.norm(rs_sol)          < TOL

        xs_sol = gmres(a, bs=bs, xs0=None, diag=d, tol=TOL, max_cycle=200, m=30)
        rs_sol = numpy.asarray([numpy.dot(a, x_sol) - b for x_sol, b in zip(xs_sol, bs)])
        assert numpy.linalg.norm(xs_sol - xs_ref) < TOL
        assert numpy.linalg.norm(rs_sol)          < TOL

        xs_sol = gmres(lambda x: numpy.dot(a, x), bs=bs, xs0=None, diag=d, tol=TOL, max_cycle=200, m=30)
        rs_sol = numpy.asarray([numpy.dot(a, x_sol) - b for x_sol, b in zip(xs_sol, bs)])
        assert numpy.linalg.norm(xs_sol - xs_ref) < TOL
        assert numpy.linalg.norm(rs_sol)          < TOL

if __name__ == '__main__':
    unittest.main()