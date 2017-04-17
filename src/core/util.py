from scipy.optimize import minimize_scalar
from sklearn import mixture
import colorsys
import scipy.stats
import numpy as np
import tensorflow as tf
import os

from tf_util import shape

# Some handy constants
lower_perc = 100 * scipy.stats.norm.cdf(-2)
upper_perc = 100 * scipy.stats.norm.cdf(2)
inf = np.inf


def color_range(num, saturation=1.0, value=1.0):
    """
    Generate a range of colors that span hue.

    :param num: num of colors in the range
    :param saturation: saturation
    :param value: value
    :return: list of RGB colors
    """
    hsvs = [(x * 1.0 / num, saturation, value) for x in range(num)]
    return map(lambda x: colorsys.hsv_to_rgb(*x), hsvs)


def length_scale(ls):
    """
    Construct constant from length scale.

    :param ls: length scale
    :return: constant
    """
    return (.5 * np.pi) * (.5 / ls ** 2)


def is_inf(x):
    """
    Check if a variable is infinite.

    :param x: variable to check
    :return: boolean whether `x` is infinite
    """
    return is_numeric(x) and np.isinf(x)


def is_numeric(x):
    """
    Check if a variable is numeric.

    :param x: variable
    :return: boolean indicating whether `x` is numeric
    """
    try:
        float(x)
        return True
    except TypeError:
        return False
    except ValueError:
        return False
    except AttributeError:
        return False


def fft(*args, **kw_args):
    """
    Alias for `np.fft.fft` that afterwards applies `np.fft.fftshift` and
    optionally normalises appropriately through the keyword `normalise`.
    """
    if 'axis' not in kw_args:
        kw_args['axis'] = -1
    if kw_args['axis'] == -1:
        kw_args['axis'] = len(shape(args[0])) - 1
    if 'normalise' not in kw_args or kw_args['normalise'] is False:
        scale = 1
    else:
        scale = shape(args[0])[kw_args['axis']]
    return np.fft.fftshift(np.fft.fft(*args, **kw_args)) / scale


def fft_freq(*args, **kw_args):
    """
    Alias for `np.fft.fft_freq` that afterwards applies `np.fft.fftshift`.
    """
    return np.fft.fftshift(np.fft.fftfreq(*args, **kw_args))


def fft_db(*args, **kw_args):
    """
    Similar to :func:`core.util.fft`, but returns the modulus of the result in
    decibel.
    """
    return 10 * np.log10(np.abs(np.real(fft(*args, **kw_args))))


def zero_pad(x, num, axis=0):
    """
    Zero pad a vector.

    :param x: array
    :param axis: axis to zero pad along
    :param num: number of zeros
    :return: zero-padded array
    """
    zeros = np.zeros(num)
    dims = len(shape(x))
    add_before = max(axis, 0)
    add_after = max(dims - axis - 1, 0)
    for i in range(add_before):
        zeros = np.expand_dims(zeros, 0)
    for i in range(add_after):
        zeros = np.expand_dims(zeros, -1)
    return np.concatenate((zeros, x, zeros), axis=axis)


def smse(x, y):
    """
    Compute standardised mean squared error of `x` with respect to `y`.

    :param x: predictions
    :param y: references
    :return: standardised mean squared error
    """
    naive_mse = np.mean((np.mean(y) - y) ** 2)
    mse = np.mean((x - y) ** 2)
    return mse / naive_mse


def mll(mu, std, y):
    """
    Compute mean log loss.

    :param mu: mean of predictions
    :param std: marginal standard deviations of prediction
    :param y: references
    :return: mean log loss
    """
    # Filter out absolutely certain data points
    mu, std, y = mu[std > 0], std[std > 0], y[std > 0]
    return .5 * np.mean(np.log(2 * np.pi * std ** 2)
                        + (mu - y) ** 2 / std ** 2)


def vlinspace(*args, **kw_args):
    """
    Alias for `np.linspace`, but returns a vector instead.
    """
    return np.array([np.linspace(*args, **kw_args)]).T


def nearest_index(xs, y):
    """
    Get the index of the element in `xs` that is nearest to `y`.

    :param xs: list
    :param y: element
    :return: index of element in `xs` that is nearest to `y`
    """
    return np.argmin(np.abs(xs - y))


def seed(number=2147483647):
    """
    Seed TensorFlow and numpy.

    :param number: number to seed with
    """
    tf.set_random_seed(number)
    np.random.seed(number)


def dict2(d=None, **kw_args):
    """
    Dictionary constructor that can simultaneously copy a dictionary and modify
    certain fields according to `**kw_args`.

    :param d: dictionary
    :param \*\*kw_args: keyword arguments
    :return: dictionary
    """
    if d is None:
        d = {}
    else:
        d = dict(d)
    d.update(kw_args)
    return d


def mkdirs(path):
    """
    Ensure that all directories in a file path exist.

    :param path: file path
    """
    dir_path = os.path.dirname(path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


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
