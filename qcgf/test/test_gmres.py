import unittest, numpy
from qcgf.lib import gmres
from qcgf.lib import _unpack

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

        a  = numpy.zeros(shape=(nn, nn),dtype=numpy.complex128)
        b  = numpy.random.rand(nn) + 0j*numpy.random.rand(nn)
        b /= numpy.linalg.norm(b)

        diag = numpy.zeros(shape=(nn), dtype=numpy.complex128)

        for i in range(nn):
            a[i,i]  = 1.0 * f + 6.0 * 1j * f + p + 30.*numpy.random.random()
            diag[i] = a[i,i] 

            if i+2 < nn:
                a[i,i+2] = 1.0 * f

            if i+3 < nn:
                a[i,i+3] = 0.7 * f

            if i+1 < nn:
                a[i+1,i] = 3.0*1j * f

        x0    = numpy.dot(numpy.linalg.inv(a), b)
        x0   += numpy.random.rand(nn) + 1j*numpy.random.rand(nn)

        x_ref = numpy.dot(numpy.linalg.inv(a), b)
        x_sol = gmres(lambda x: numpy.dot(a, x), b, xs0=x0, diag=diag, tol=1e-6, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b

        assert numpy.linalg.norm(x_sol - x_ref) < 1e-6
        assert numpy.linalg.norm(r_sol)         < 1e-6


    def test_gmres_2(self):
        nn = 1000
        nb = 10
        f  = 1.0
        p  = 0.0+1j*0.0

        # Generate the RHS of the linear equation
        gen_b = lambda b: (b / numpy.linalg.norm(b))
        bs    = [gen_b(numpy.random.rand(nn) + 0.1j*numpy.random.rand(nn)) for i in range(nb)]

        a     = numpy.zeros(shape=(nn, nn),dtype=numpy.complex128)
        diag  = numpy.zeros(shape=(nn), dtype=numpy.complex128)

        for i in range(nn):
            a[i, i] = 1.0 * f + 6.0 * 1j * f + p + 30.*numpy.random.random()
            diag[i] = a[i, i] 

            if i+2 < nn:
                a[i, i+2] = 1.0 * f

            if i+3 < nn:
                a[i, i+3] = 0.7 * f

            if i+1 < nn:
                a[i+1, i] = 3.0*1j * f

        xs_ref = []
        xs0    = []
        for ib, b in enumerate(bs):
            xs_ref.append(numpy.dot(numpy.linalg.inv(a), b))

            x  = numpy.dot(numpy.linalg.inv(a), b)
            x += numpy.random.rand(nn) + 1j*numpy.random.rand(nn)
            xs0.append(x)
            
        xs0    = numpy.asarray(xs0)
        xs_ref = numpy.asarray(xs_ref)

        xs_sol = gmres(lambda x: numpy.dot(a, x), bs, xs0=None, diag=diag, tol=1e-6, max_cycle=200, m=30)
        rs_sol = [numpy.dot(a, x) - b for x, b in zip(xs_sol, bs)]

        assert numpy.linalg.norm(xs_sol - xs_ref) < 1e-6
        assert numpy.linalg.norm(rs_sol)         < 1e-6, numpy.linalg.norm(rs_sol)

if __name__ == '__main__':
    unittest.main()