import tensorflow as tf
import numpy as np
import tensorflow.contrib.distributions as tf_dists
from tensorflow.python.client import timeline
from tensorflow.python.framework import ops

import config

from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
importr("pbivnorm")


def pw_dists2(x, y, output_norms=False):
    """
    Compute pairwise distances between the rows of `x` and `y`.

    :param x: first set of features
    :param y: second set of features
    :param output_norms: boolean that indicates whether to also output the
                         norms of `x` and `y`
    :return: squared distances, further outputs norms of `x` and norms of `y`
             if `output_norms` is set to `True`
    """
    norms2_x = tf.reduce_sum(x ** 2, axis=[1])[:, None]
    norms2_y = tf.reduce_sum(y ** 2, axis=[1])[None, :]
    if output_norms:
        return norms2_x - 2 * mul(x, y, adj_b=True) + norms2_y, norms2_x, \
               norms2_y
    else:
        return norms2_x - 2 * mul(x, y, adj_b=True) + norms2_y


def ctransp(x):
    """
    Batch conjugate transpose a tensor.

    :param x: tensor
    :return: tensor where the last two dimensions are conjugate transposed
    """
    return tf.conj(transp(x))


def transp(x):
    """
    Batch transpose a tensor.

    :param x: tensor
    :return: tensor where the last two dimensions are transposed
    """
    perm = range(len(shape(x)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return tf.transpose(x, perm=perm)


def trace(x):
    """
    Batch trace a tensor.

    :param x: tensor
    :return: traces of the last two dimensions of the tensor
    """
    return tf.reduce_sum(tf.matrix_diag_part(x), [-1])


def eye(n):
    """
    :math:`N`-dimensional identity matrix.

    :param n: number of dimensions
    :return: identity matrix
    """
    return tf.diag(tf.ones([n], dtype=config.dtype))


def invert_perm(perm):
    """
    Invert a permutation.

    :param perm: permutation
    :return: inverse of permutation
    """
    n = len(perm)
    inverse_perm = range(n)
    for i in range(n):
        inverse_perm[perm[i]] = i
    return inverse_perm


def randn(size):
    """
    Alias for `tf.random_normal`.
    """
    return tf.random_normal(size, dtype=config.dtype)


def zeros(size):
    """
    Alias for `tf.zeros`.
    """
    return tf.zeros(size, dtype=config.dtype)


def ones(size):
    """
    Alias for `tf.ones`.
    """
    return tf.ones(size, dtype=config.dtype)


def log_det(chol):
    """
    Compute log determinant given the Cholesky decomposition.

    :param chol: Cholesky decomposition
    :return: log determinant
    """
    return 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(chol)), [-1])


def to_float(obj):
    """
    Cast to float.

    :param obj: object
    :return: object casted to float
    """
    return tf.cast(tf.to_float(obj), config.dtype)


def log_2_pi():
    """
    Compute :math:`\\log (2 \\pi)`.

    :return: :math:`\\log (2 \\pi)`.
    """
    return tf.log(to_float(2 * np.pi))


def mul(a, b, adj_a=False, adj_b=False):
    """
    Alias for `tf.batch_matmul`.
    """
    return tf.matmul(a, b, adj_x=adj_a, adj_y=adj_b)


def trisolve(a, b, lower_a=True, adj_a=False):
    """
    Alias for `tf.matrix_triangular_solve`.
    """
    return tf.matrix_triangular_solve(a, b, lower=lower_a, adjoint=adj_a)


def trisolve3(a, b, c, lower_a=True, lower_c=True, adj_a=False, adj_c=False):
    """
    Compute :math:`A^{-1}BC^{-1}` where :math:`A` and :math:`C` are lower
    triangular.

    :param a: :math:`A`
    :param b: :math:`B`
    :param c: :math:`C`
    :param lower_a: boolean whether `a` is lower triangular
    :param lower_c: boolean whether `c` is lower triangular
    :param adj_a: boolean whether to adjoint `a`
    :param adj_c: boolean whether to adjoint `c`
    :return: :math:`A^{-1}BC^{-1}`
    """
    return trisolve(a,
                    ctransp(trisolve(c,
                                     ctransp(b),
                                     lower_a=lower_c,
                                     adj_a=not adj_c)),
                    lower_a=lower_a,
                    adj_a=adj_a)


def mul3(a, b, c, adj_a=False, adj_b=False, adj_c=False):
    """
    Compute :math:`ABC`.

    :param a: :math:`A`
    :param b: :math:`B`
    :param c: :math:`C`
    :param adj_a: boolean whether to adjoint `a`
    :param adj_b: boolean whether to adjoint `b`
    :param adj_c: boolean whether to adjoint `c`
    :return: :math:`ABC`
    """
    return mul(a, mul(b, c, adj_a=adj_b, adj_b=adj_c), adj_a=adj_a)


def frob2(x):
    """
    Compute square of Frobenius norm.

    :param x: :math:`X`
    :return: :math:`\\|X\\|^2_F`.
    """
    return trmul(x, x)


def trmul(a, b, adj_a=False, adj_b=False):
    """
    Compute :math:`\operatorname{tr}(A^TB)`.

    :param a: :math:`A`
    :param b: :math:`B`
    :param adj_a: boolean whether to adjoint `a`
    :param adj_b: boolean whether to adjoint `b`
    :return: :math:`\operatorname{tr}(A^TB)`.
    """
    a = ctransp(a) if adj_a else a
    b = ctransp(b) if adj_b else b
    return tf.reduce_sum(a * b, [-2, -1])


def shape(x):
    """
    Get the shape of `x`.

    :param x: object to get shape of
    :return: shape
    """
    if type(x) == np.ndarray:
        return x.shape
    else:
        return tuple(int(y) for y in x.get_shape())


def rms(x, dims=None):
    """
    Compute RMS.

    :param x: tensor to compute RMS of
    :param dims: reduction dimensions
    :return: RMS of `x`
    """
    return tf.sqrt(tf.reduce_mean(x ** 2, dims))


def sum(*args, **kw_args):
    """
    Alias for `tf.sum`.
    """
    return tf.reduce_sum(*args, **kw_args)


def tile(x, pre=None, post=None):
    """
    Tile a tensor.

    :param x: tensor to tile
    :param pre: sizes of dimensions to be inserted before the first index
    :param post: sizes of dimensions to be inserted after the last index
    :return: tiled tensor
    """
    # Convert from tuples
    pre = list(pre) if type(pre) == tuple else pre
    post = list(post) if type(post) == tuple else post
    # Convert to lists
    pre = [] if pre is None else [pre] if type(pre) != list else pre
    post = [] if post is None else [pre] if type(post) != list else post
    dims = len(shape(x))
    x = expand_dims(x, len(pre), len(post))
    return tf.tile(x, pre + [1] * dims + post)


def expand_dims(x, pre=0, post=0):
    """
    Expand dimensions of tensor.

    :param x: tensor to expand
    :param pre: number of dimensions to be inserted before the first index
    :param post: number of dimensions to be inserted after the last index
    :return: expanded tensor
    """
    for i in range(pre):
        x = tf.expand_dims(x, 0)
    for i in range(post):
        x = tf.expand_dims(x, -1)
    return x


def cholinv(x):
    """
    Compute the inverse given the Cholesky decomposition.

    :param x: Cholesky decomposition
    :return: inverse
    """
    return tf.cholesky_solve(x, eye(shape(x)[-1]))


def outer(a, b=None):
    """
    Compute the outer product.

    :param a: :math:`A`
    :param b: :math:`B`
    :return: :math:`AB^T`
    """
    b = a if b is None else b
    if shape(a)[-1] == 1 and shape(b)[-1] == 1:
        return a * transp(b)
    else:
        return mul(a, b, adj_b=True)


def get_var(name, offset=0):
    """
    Get a TensorFlow variable by name.

    :param name: name of variable
    :param offset: in the case of multiple variables having the same name, this
                   parameter get be used to differentiate between those
    :return: variable
    """
    vars = [x for x in tf.global_variables()
            if x.name == '{}:{}'.format(name, offset)]
    if len(vars) == 0:
        raise RuntimeError('no TensorFlow variable with the name '
                           '"{}:{}"'.format(name, offset))
    elif len(vars) > 1:
        raise RuntimeError('multiple TensorFlow variables with the name '
                           '"{}:{}"'.format(name, offset))
    else:
        return vars[0]


def placeholder(*args, **kw_args):
    """
    Alias for `tf.placeholder`.
    """
    return tf.placeholder(config.dtype, *args, **kw_args)


def reg(x, diag=None):
    """
    Regularise a matrix.

    :param x: matrix
    :param diag: size of diagonal
    :return: matrix with a small diagonal added
    """
    if diag is None:
        diag = config.reg
    return x + diag * eye(shape(x)[-1])


def var_pos(init_value, name=None):
    """
    Construct a TensorFlow variable that is constrained to be positive.

    :param init_value: initial value
    :param name: name of variable
    :return: variable
    """
    return tf.exp(tf.Variable(tf.log(init_value), name=name))


class SessionDecoratorDebug(object):
    """
    Decorator to add debugging functionality.

    :param sess: session
    """

    def __init__(self, sess):
        self.sess = sess
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()

    def run(self, *args, **kw_args):
        """
        Alias for `tf.Session.run`.
        """
        if 'debug' in kw_args:
            if kw_args['debug']:
                kw_args['options'] = self.run_options
                kw_args['run_metadata'] = self.run_metadata
            del kw_args['debug']
        return self.sess._run(*args, **kw_args)

    def report(self, fn='debug.ctl'):
        """
        Write debugging report. Open in 'chrome://tracing'.

        :param fn: file name to write to
        """
        tl = timeline.Timeline(self.run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(fn, 'w') as f:
            f.write(ctf)

    def as_default(self):
        """
        Alias for `tf.Session.as_default`.
        """
        return self.sess.as_default()


def _bvn_cdf(x, mu, var):
    """
    Bivariate normal CDF.

    :param x: matrix where rows correspond to points
    :param mu: mean as a row vector
    :param var: variance
    :return: CDF evaluated at the rows of `x`
    """
    x -= mu
    x /= np.sqrt(np.diag(var))
    rho = var[0, 1] / np.sqrt(var[0, 0] * var[1, 1])
    return _bvn_cdf2(x, rho)


def _bvn_cdf2(x, rho):
    """
    Standard bivariate normal CDF.

    :param x: matrix where rows correspond to points
    :param rho: correlation
    :return: CDF evaluated at the rows of `x`
    """
    return np.array(r.pbivnorm(x, rho=rho))


def _bvn_cdf2_grad(op, grad):
    """
    TensorFlow gradient for `bvn_cdf2`.

    :param op: operation
    :param grad: initial gradient
    :return: gradient
    """
    xs = op.inputs[0][:, 0]
    ys = op.inputs[0][:, 1]
    rho = op.inputs[1]
    q = tf.sqrt(1 - rho ** 2)

    pdfs = 1 / (2 * np.pi * q) \
           * tf.exp(-(xs ** 2 - 2 * rho * xs * ys + ys ** 2)
                    / (2 * (1 - rho ** 2)))
    grad_rho = sum(grad * pdfs)

    dist_z = tf_dists.Normal(to_float(0), to_float(1))
    dist_x = tf_dists.Normal(rho * xs, q)
    dist_y = tf_dists.Normal(rho * ys, q)
    grad_x = tf.stack([grad, grad], axis=-1) \
             * tf.stack([dist_z.prob(xs) * dist_x.cdf(ys),
                         dist_z.prob(ys) * dist_y.cdf(xs)], axis=-1)

    return [grad_x, grad_rho]


def bvn_cdf2(x, rho, name=None):
    """
    TensorFlow operation for standard bivariate normal CDF.

    :param x: matrix where rows correspond to points
    :param rho: correlation
    :param name: name of operation
    :return: CDF evaluated at the rows of `x`
    """
    with ops.name_scope(name, 'bvn_cdf2', [x]) as name:
        return py_func(_bvn_cdf2,
                       [x, rho],
                       [config.dtype],
                       grad=_bvn_cdf2_grad,
                       name=name)[0]


def py_func(*args, **kw_args):
    """
    Extension of `tf.py_func` that also allows for gradient specification.
    Source: https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
    """
    if 'grad' not in kw_args:
        raise RuntimeError('must specify gradient')

    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, int(1e8)))
    tf.RegisterGradient(rnd_name)(kw_args['grad'])
    del kw_args['grad']

    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(*args, **kw_args)


def initialise_uninitialised_variables(sess):
    """
    Initialise uninitialised variables.

    :param sess: TensorFlow session
    """
    for var in tf.global_variables():
        if not sess.run(tf.is_variable_initialized(var)):
            sess.run(tf.variables_initializer([var]))
