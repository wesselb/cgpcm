import unittest

from tf_util import *


class TestBVNCDF2(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.delta = 1e-8

        # Construct some complicated function
        self.v = ones(3)
        self.random = self.sess.run(randn([3]))
        self.div = to_float(5)
        self.rho = sum(self.v * self.random) / self.div
        self.x = constant(np.array([[0.1, 0.2], [0.2, 0.3]]))
        self.y = bvn_cdf2(self.x, self.rho)

    def test_gradient_dy0_dx00(self):
        e = constant(np.array([[1., 0], [0, 0]])) * self.delta / 2.
        y1 = bvn_cdf2(self.x - e, self.rho)
        y2 = bvn_cdf2(self.x + e, self.rho)
        out = self.sess.run(y2[0] - y1[0]) / self.delta
        ref = self.sess.run(tf.gradients([self.y[0]], self.x)[0][0, 0])
        np.testing.assert_almost_equal(out, ref)

    def test_gradient_dy1_dx10(self):
        e = constant(np.array([[0, 0], [1., 0]])) * self.delta / 2.
        y1 = bvn_cdf2(self.x - e, self.rho)
        y2 = bvn_cdf2(self.x + e, self.rho)
        out = self.sess.run(y2[1] - y1[1]) / self.delta
        ref = self.sess.run(tf.gradients([self.y[1]], self.x)[0][1, 0])
        np.testing.assert_almost_equal(out, ref)

    def test_gradient_dy0_drho(self):
        e = to_float(1) * self.delta / 2
        y1 = bvn_cdf2(self.x, self.rho - e)
        y2 = bvn_cdf2(self.x, self.rho + e)
        out = self.sess.run(y2[0] - y1[0]) / self.delta
        ref = self.sess.run(tf.gradients([self.y[0]], self.rho)[0])
        np.testing.assert_almost_equal(out, ref)

    def test_gradient_dy1_dv0(self):
        e = constant(np.array([1., 0, 0])) * self.delta / 2
        rho1 = sum((self.v - e) * self.random) / self.div
        rho2 = sum((self.v + e) * self.random) / self.div
        y1 = bvn_cdf2(self.x, rho1)
        y2 = bvn_cdf2(self.x, rho2)
        out = self.sess.run(y2[1] - y1[1]) / self.delta
        ref = self.sess.run(tf.gradients([self.y[1]], self.v)[0][0])
        np.testing.assert_almost_equal(out, ref)

    def test_gradient_dy1_dv2(self):
        e = constant(np.array([0, 0, 1.])) * self.delta / 2
        rho1 = sum((self.v - e) * self.random) / self.div
        rho2 = sum((self.v + e) * self.random) / self.div
        y1 = bvn_cdf2(self.x, rho1)
        y2 = bvn_cdf2(self.x, rho2)
        out = self.sess.run(y2[1] - y1[1]) / self.delta
        ref = self.sess.run(tf.gradients([self.y[1]], self.v)[0][2])
        np.testing.assert_almost_equal(out, ref)
