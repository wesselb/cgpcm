import unittest

from tf_util import *
from distribution import Normal
import util


class TestSmartSign(unittest.TestCase):
    def setUp(self):
        util.seed()
        self._sess = Session()

        dims = 100
        dist = Normal(np.diag(np.random.rand(dims)),
                      np.random.rand(dims)[:, None])
        self._samples = self._sess.run(dist.sample(200))

    def test1(self):
        signs = np.sign(np.random.randn(shape(self._samples)[1]))
        corrupted_samples = signs[None, :] * self._samples

        signs_pred = util.sign_smart(corrupted_samples.T)
        assert (np.all(signs * signs_pred))
