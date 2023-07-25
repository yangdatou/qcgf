import unittest, numpy
from qcgf.lib import gmres

class TestGMRES(unittest.TestCase):
    def test_gmres(self):
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
        x_sol = gmres(lambda x: numpy.dot(a, x), b, x0, diag, tol=1e-6, max_cycle=200, m=30)
        r_sol = numpy.dot(a, x_sol) - b

        assert numpy.linalg.norm(x_sol - x_ref) < 1e-6
        assert numpy.linalg.norm(r_sol)         < 1e-6

if __name__ == '__main__':
    unittest.main()