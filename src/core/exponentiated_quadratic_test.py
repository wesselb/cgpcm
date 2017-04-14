import unittest

from exponentiated_quadratic import *
from util import inf


class TestEQ(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.t1 = var('t1')
        self.t2 = var('t2')
        self.t3 = var('t3')
        self.exp1 = EQ(- const(1) * self.t1 ** 2
                       - const(2) * self.t2 ** 2
                       - const(.5) * self.t1 * self.t2
                       - const(2) * self.t1 * self.t3
                       + const(3) * self.t2
                       + const(4))
        self.exp2 = EQ(const(-1) * self.t1 ** 2
                       + const(-.5) * self.t1
                       + const(4))
        self.var_map = {'t3': constant(np.eye(2))}

    def test1(self):
        ref = np.array([[55.81808295, 11.76773162],
                        [11.76773162, 55.81808295]])
        res = self.sess.run(self.exp1.integrate_box(('t1', -inf, 0),
                                                    ('t2', -inf, 0),
                                                    **self.var_map))
        np.testing.assert_almost_equal(res, ref, decimal=6)

    def test2(self):
        ref = np.array([[217.3921457, 318.3540954],
                        [318.3540954, 217.3921457]])
        box = [('t1', const(-1), const(2)), ('t2', self.t3, const(3))]
        res = self.sess.run(self.exp1.integrate_box(*box, **self.var_map))
        np.testing.assert_almost_equal(res, ref, decimal=5)

    def test3(self):
        ref = 65.73974603
        res = self.sess.run(self.exp2.integrate_half('t1'))
        np.testing.assert_almost_equal(res, ref)
