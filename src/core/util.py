import colorsys
import scipy.stats
import numpy as np

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
    return .5 / ls ** 2


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
    Alias for `np.fft.fft` which afterwards applies `np.fft.fftshift`.
    """
    return np.fft.fftshift(np.fft.fft(*args, **kw_args))


def fft_freq(*args, **kw_args):
    """
    Alias for `np.fft.fft_freq` which afterwards applies `np.fft.fftshift`.
    """
    return np.fft.fftshift(np.fft.fftfreq(*args, **kw_args))


def psd(*args, **kw_args):
    """
    Similar to `np.fft.fft`, but computes the PSD in dB instead.
    """
    return 10 * np.log10(np.abs(fft(*args, **kw_args)))


def zero_pad(x, num, axis=0):
    """
    Zero pad an array.

    :param x: array
    :param axis: axis to zero pad along
    :param num: number of zeros
    :return: zero-padded array
    """
    zeros = np.zeros(num)
    if axis > 0:
        for i in range(axis - 1):
            zeros = np.expand_dims(zeros, 0)
    return np.concatenate((zeros, x, zeros), axis=axis)


def smse(x, y):
    """
    Compute standardised mean squared error of `x` with respect to `y`.

    :param x: predictions
    :param y: references
    :return: standardised mean squared error
    """
    y_mean = np.mean(y)
    y_var = np.mean((y - y_mean) ** 2)
    mse = np.mean((x - y) ** 2)
    return mse / y_var


def mll(mu, std, y):
    """
    Compute mean log loss.

    :param mu: mean of predictions
    :param std: marginal standard deviations of prediction
    :param y: references
    :return: mean log loss
    """
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
