import abc

from tf_util import *
from parametrisable import Parametrisable


class Kernel(Parametrisable):
    __metaclass__ = abc.ABCMeta

    def __call__(self, x, y=None):
        """
        Evaluate the kernel for a matrix of points.

        :param x: points to evaluate kernel at along rows
        :param y: points to evaluate kernel at along columns
        :returns: kernel matrix
        """
        if y is None:
            y = x
        if len(shape(x)) != len(shape(y)):
            raise ValueError('arguments must have same number of dimensions')
        if len(shape(x)) == 1:
            return self.__call__(x[:, None], y[:, None])
        else:
            return self._call(x, y)

    @abc.abstractmethod
    def _call(self, x, y):
        pass


class DEQ(Kernel):
    """
    Decaying exponentiated-quadratic kernel.

    :param s2: kernel variance :math:`\\sigma^2`
    :param alpha: kernel decay parameter :math:`\\alpha`
    :param gamma: kernel length scale parameter :math:`\\gamma`
    """

    _required_pars = ['s2', 'alpha', 'gamma']

    def _call(self, x, y):
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

    def _call(self, x, y):
        dists = pw_dists2(x, y, output_norms=False) ** .5
        return self.s2 * tf.exp(-self.gamma * dists)
