from core.tf_util import *


class Normal(object):
    """
    Normal distribution.

    :param var: variance
    :param mean: mean
    """

    def __init__(self, var, mean=None):
        self.dims = shape(var)[-1]
        if mean is None:
            self.mean = zeros([self.dims, 1])
        else:
            self.mean = mean
        self.var = var

    @classmethod
    def from_natural(cls, precision, nat_mean=None):
        """
        Construct from natural parameters.
        
        :param precision: precision
        :param nat_mean: natural mean
        :return: :class:`core.distribution.Normal` instance
        """
        L = tf.cholesky(reg(precision))
        if nat_mean is None:
            return cls(cholinv(L))
        else:
            return cls(cholinv(L), tf.cholesky_solve(L, nat_mean))

    @property
    def m2(self):
        """
        Get the second moment.

        :return: second moment
        """
        return self.var + outer(self.mean)

    def log_pdf(self, x):
        """
        Get the log of the pdf evaluated for a feature matrix `x`.

        :param x: points to evaluate at
        :return: log pdf
        """
        if shape(x)[1] != self.dims:
            raise RuntimeError('Dimensionality of data points does not match '
                               'distribution')
        n = shape(x)[0]  # Number of data points
        chol = tf.cholesky(self.var)
        y = trisolve(chol, tf.transpose(x) - self.mean)
        return -(log_det(chol) + n * to_float(self.dims) * log_2_pi()
                 + tf.reduce_sum(y * y)) / 2.

    def kl(self, other):
        """
        Compute the KL divergence with respect to another normal.

        :param other: other normal
        :return: KL divergence
        """
        if self.dims != other.dims:
            raise RuntimeError('KL divergence can only be computed between '
                               'distributions of the same dimensionality')
        chol_self = tf.cholesky(self.var)
        chol_other = tf.cholesky(other.var)
        mu_diff_part = tf.reduce_sum(trisolve(chol_other,
                                              other.mean - self.mean) ** 2)
        trace_part = tf.reduce_sum(trisolve(chol_other, chol_self) ** 2)
        return .5 * (trace_part + mu_diff_part - to_float(self.dims)
                     + log_det(chol_other) - log_det(chol_self))

    def sample(self, num=1):
        """
        Generate a sample.

        :param num: number of samples
        :return: sample
        """
        sample = randn([self.dims, num])
        return mul(tf.cholesky(self.var), sample) + self.mean


class Uniform(object):
    """
    Uniform distribution.

    :param lb: lower bound
    :param ub: upper bound
    """

    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def sample(self):
        """
        Generate a sample.

        :return: sample
        """
        return self.lb + (self.ub - self.lb) * tf.random_uniform([])
