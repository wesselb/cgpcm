import abc
from utils import *


class Kernel:
    __metaclass__ = abc.ABCMeta

    def __init__(self, hyps):
        self._hyps = hyps

    def __getitem__(self, hyp):
        return self._hyps[hyp]

    @abc.abstractmethod
    def __call__(self, points):
        """
        Evaluate the kernel for a matrix of points.

        :param points: points to evaluate kernel at
        :returns: kernel matrix
        """
        pass


class DEQ(Kernel):
    """
    Decaying exponentiated-quadratic kernel.

    :param hyps: dictionary containing the following hyperparameters:

                    - `s2`: :math:`\\sigma^2`,
                    - `alpha`: :math:`\\alpha`, and
                    - `gamma`: :math:`\\gamma`
    """
    def __call__(self, x, y=None):
        if y is None:
            y = x
        dists2, norms2_x, norms2_y = pw_dists2(x, y)
        return self['s2'] * tf.exp(-self['alpha'] * (norms2_x + norms2_y)
                                   - self['gamma'] * dists2)


class UhlenbeckOrnstein(Kernel):
    """
    Uhlenbeck-Ornstein kernel.

    :param hyps: dictionary containing the following hyperparameters:

                    - `s2`: :math:`\\sigma^2`,
                    - `gamma`: :math:`\\gamma`
    """
    def __call__(self, x, y=None):
        if y is None:
            y = x
        dists2, norms2_x, norms2_y = pw_dists2(x, y)
        return self['s2'] * tf.exp(-self['gamma'] * dists2 ** .5)


class RLCSeriesCircuit(Kernel):
    """
    Kernel corresponding to an RLC series circuit excited by white noise due to
    thermal noise of the resistance.

    :param hyps: dictionary containing the following hyperparameters:

                    - `s2`: :math:`\\sigma^2`,
                    - `gamma`: :math:`\\gamma`
    """
    def __call__(self, x, y=None):
        if y is None:
            y = x
        dists2, norms2_x, norms2_y = pw_dists2(x, y)
        dists = self['gamma'] * dists2 ** .5
        return self['s2'] * (1 + dists) * tf.exp(-dists)

