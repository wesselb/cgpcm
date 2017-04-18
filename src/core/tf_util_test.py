import unittest
import numpy as np
import tensorflow as tf

from tf_util import mul, randn, tile, trmul


class TestBroadcast(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()
        self.a_l = self.sess.run(randn([15, 10]))
        self.a_l_tiled = self.sess.run(tile(self.a_l, 5))
        self.b_r = self.sess.run(randn([5, 10, 20]))

        self.a_r = self.sess.run(randn([10, 15]))
        self.a_r_tiled = self.sess.run(tile(self.a_r, 5))
        self.b_l = self.sess.run(randn([5, 20, 10]))

    def test_mul_left_broadcast(self):
        res = self.sess.run(mul(self.a_l, self.b_r))
        ref = self.sess.run(mul(self.a_l_tiled, self.b_r))
        np.testing.assert_almost_equal(res, ref)

    def test_mul_right_broadcast(self):
        res = self.sess.run(mul(self.b_l, self.a_r))
        ref = self.sess.run(mul(self.b_l, self.a_r_tiled))
        np.testing.assert_almost_equal(res, ref)

    def test_trmul_left_broadcast(self):
        res = self.sess.run(trmul(self.a_l, self.a_l))
        ref = self.sess.run(trmul(self.a_l_tiled, self.a_l))
        np.testing.assert_almost_equal(res, ref)

    def test_trmul_right_broadcast(self):
        res = self.sess.run(trmul(self.a_r, self.a_r))
        ref = self.sess.run(trmul(self.a_r, self.a_r_tiled))
        np.testing.assert_almost_equal(res, ref)
