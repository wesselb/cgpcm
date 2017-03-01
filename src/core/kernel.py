import abc
from tfutil import *
from par import Parametrisable


class Kernel(Parametrisable):
    __metaclass__ = abc.ABCMeta

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

    :param s2: kernel variance :math:`\\sigma^2`
    :param alpha: kernel decay parameter :math:`\\alpha`
    :param gamma: kernel length scale parameter :math:`\\gamma`
    """

    _required_pars = ['s2', 'alpha', 'gamma']

    def __call__(self, x, y=None):
        if y is None:
            y = x
        dists2, norms2_x, norms2_y = pw_dists2(x, y, output_norms=True)
        return self.s2 * tf.exp(-self.alpha * (norms2_x + norms2_y)
                                - self.gamma * dists2)


class Exponential(Kernel):
    """
    Exponential kernel.

    :param s2: kernel variance :math:`\\sigma^2`
    :param gamma: kernel length scale parameter :math:`\\gamma`
    """

    _required_pars = ['s2', 'gamma']

    def __call__(self, x, y=None):
        if y is None:
            y = x
        dists = pw_dists2(x, y, output_norms=False) ** .5
        return self.s2 * tf.exp(-self.gamma * dists)
