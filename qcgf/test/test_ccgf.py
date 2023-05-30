# BEGIN: 8d7f6a5f8d7f
import numpy
import unittest

from qcgf.ccgf import _rhs_1_ip, _rhs_2_ip, _rhs_ip, _lhs_ip

class TestCCGF(unittest.TestCase):
    def setUp(self):
        self.nocc = 5
        self.nvir = 10
        self.t1 = numpy.random.rand(self.nocc, self.nvir)
        self.t2 = numpy.random.rand(self.nocc, self.nocc, self.nvir, self.nvir)
        self.l1 = numpy.random.rand(self.nocc, self.nvir)
        self.l2 = numpy.random.rand(self.nocc, self.nocc, self.nvir, self.nvir)

    def test_rhs_1_ip(self):
        for i in range(self.nocc):
            r1 = _rhs_1_ip(self.t1, self.t2, i)
            self.assertEqual(r1.shape, (self.nocc,))
            self.assertAlmostEqual(numpy.linalg.norm(r1), 1.0)

    def test_rhs_2_ip(self):
        for i in range(self.nocc + self.nvir):
            r2 = _rhs_2_ip(self.t1, self.t2, i)
            if i < self.nocc:
                self.assertEqual(r2.shape, (self.nocc, self.nocc, self.nvir))
                self.assertAlmostEqual(numpy.linalg.norm(r2[i]), 1.0)
            else:
                a = i - self.nocc
                self.assertEqual(r2.shape, (self.nocc, self.nocc, 1))
                self.assertAlmostEqual(numpy.linalg.norm(r2[:, :, a]), 1.0)

    def test_rhs_ip(self):
        for i in range(self.nocc + self.nvir):
            rhs_1, rhs_2 = _rhs_ip(self.t1, self.t2, self.l1, self.l2, i)
            self.assertEqual(rhs_1.shape, (self.nocc,))
            self.assertEqual(rhs_2.shape, (self.nocc, self.nocc, self.nvir))

            if i < self.nocc:
                self.assertAlmostEqual(numpy.linalg.norm(rhs_1[i]), 1.0)
                self.assertAlmostEqual(numpy.linalg.norm(rhs_2[i]), 1.0)
            else:
                a = i - self.nocc
                self.assertAlmostEqual(numpy.linalg.norm(rhs_1[:, a]), 1.0)
                self.assertAlmostEqual(numpy.linalg.norm(rhs_2[:, :, a]), 1.0)

    def test_lhs_ip(self):
        for i in range(self.nocc + self.nvir):
            lhs_1, lhs_2 = _lhs_ip(self.t1, self.t2, self.l1, self.l2, i)
            self.assertEqual(lhs_1.shape, (self.nocc,))
            self.assertEqual(lhs_2.shape, (self.nocc, self.nocc, self.nvir))

            if i < self.nocc:
                self.assertAlmostEqual(numpy.linalg.norm(lhs_1[i]), 1.0)
                self.assertAlmostEqual(numpy.linalg.norm(lhs_2[i]), 1.0)
            else:
                a = i - self.nocc
                self.assertAlmostEqual(numpy.linalg.norm(lhs_1[:, a]), 1.0)
                self.assertAlmostEqual(numpy.linalg.norm(lhs_2[:, :, a]), 1.0)

if __name__ == '__main__':
    unittest.main()
# END: 8d7f6a5f8d7f