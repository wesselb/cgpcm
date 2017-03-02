import scipy.io as sio
import scipy.io.wavfile as sio_wav

from tfutil import *
from util import *
import cgpcm
import kernel


class Data(object):
    """
    Data.

    :param t: times
    :param y: values
    """

    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.n = shape(t)[0]

    def subsample(self, n):
        """
        Subsample the data.

        :param n: number of points in subsample
        :return: subsampled data and remainder of data
        """
        return self._split(np.random.choice(self.n, n, replace=False))

    def fragment(self, length, start=0):
        """
        Fragment the data.

        :param length: length of fragment
        :param start: start of fragment
        :return: fragmented data
        """
        return self._split(range(start, start + length))

    def _split(self, i_in):
        i_in = sorted(i_in)
        i_out = sorted(list(set(range(self.n)) - set(i_in)))
        return Data(self.t[i_in], self.y[i_in]), \
               Data(self.t[i_out], self.y[i_out])

    def smse(self, y):
        """
        Compute the standardised mean squared error for some predictions.

        :param y: predictions
        :return: standardised means squared error
        """
        return smse(y, self.y)

    def mll(self, mu, std):
        """
        Compute the mean log loss for some predictions.

        :param mu: mean of predictions
        :param std: standard deviation of predictions
        :return: mean log loss
        """
        return mll(mu, std, self.y)

    def make_noisy(self, var):
        """
        Make the data noisy.

        :param var: variance of noise
        :return: noisy data
        """
        noise = var ** .5 * np.random.randn(*shape(self.y))
        return Data(self.t, self.y + noise)

    @property
    def mean(self):
        """
        Mean of data.
        """
        return np.mean(self.y)

    @property
    def std(self):
        """
        Standard deviation of data.
        """
        return np.std(self.y, ddof=1)

    def __neg__(self):
        return Data(self.t, -self.y)

    def __add__(self, other):
        return Data(self.t, self.y + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Data(self.t, self.y - other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __div__(self, other):
        return Data(self.t, self.y / other)

    def __rdiv__(self, other):
        return 1 / self.__div__(other)

    def __mul__(self, other):
        return Data(self.t, self.y * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, item):
        return self.y[item]

    @classmethod
    def from_gp(cls, sess, kernel, t):
        """
        Draw from a GP.

        :param sess: TensorFlow session
        :param kernel: kernel of GP
        :param t: times
        """
        e = randn([shape(t)[0], 1])
        K = reg(kernel(t[:, None]))
        y = sess.run(mul(tf.cholesky(K), e))
        return cls(t, y)

    @classmethod
    def from_wav(cls, fn):
        """
        Data from a WAV file.

        :param fn: file name
        """
        fs, y = sio_wav.read(fn)
        y = y.astype(float)
        t = np.arange(shape(y)[0]) / float(fs)
        return cls(t, y)


def load_hrir(n=1000, h_wav_fn='data/KEMAR_R10e355a.wav'):
    """
    Load HRIR data.

    :param n: number of response data points
    :param h_wav_fn: HRIR WAV file
    :return: data for function, filter, and noise
    """
    h = Data.from_wav(h_wav_fn)
    dt = h.t[1] - h.t[0]
    t = np.arange(n) * dt
    x = Data(t, np.random.randn(n))
    x /= x.std  # Make sure noise is unity variance
    f = Data(t, np.convolve(h.y, x.y)[:n])

    # Normalise
    h /= f.std
    f /= f.std
    return f, h


def load_timit_tobar2015():
    """
    Load TIMIT data set from Tobar et al. (2015).

    :return: data for function
    """
    mat = sio.loadmat('data/TIMIT_unknown.mat')
    y = np.squeeze(mat['audio'])
    fs = np.squeeze(mat['Fs']).astype(float)
    t = np.arange(shape(y)[0]) / fs
    f = Data(t, y).fragment(1750, start=11499)[0]
    f = (f - f.mean) / f.std
    return f


def load_gp_exp(sess, n=250, k_len=.1):
    """
    Sample from a GP with an exponential kernel.

    :param sess: TensorFlow session
    :param n: number of time points
    :param k_len: length of kernel
    :return: data for function and kernel
    """
    t = np.linspace(0, 1, n)
    tk = np.linspace(0, 1, 301)
    k_fun = kernel.Exponential(s2=1., gamma=length_scale(k_len))
    f = Data.from_gp(sess, k_fun, t)
    k = Data(tk, k_fun(tk[0, :], np.array([[0]])))

    # Normalise
    f -= f.mean
    k /= max(k.y)
    return f, k


def load_akm(sess, causal, n=250, nh=31, k_len=.1, k_wiggles=2, resample=0):
    """
    Sample from the AKM.

    :param sess: TensorFlow session
    :param causal: boolean that indicates whether the AKM is causal
    :param n: number of time points
    :param nh: number of inducing points for the filter
    :param k_len: length of kernel
    :param k_wiggles: number of wiggles in kernel
    :param resample: number of times to resample function
    :return: data for function, kernel, and filter, and parameters of the AKM
    """
    t = np.linspace(0, 1, n)
    tk = np.linspace(-2 * k_len, 2 * k_len, 301)
    pars = cgpcm.cgpcm_pars(sess=sess,
                            t=t,
                            y=[],
                            nx=0,
                            nh=nh,
                            k_len=k_len,
                            k_wiggles=k_wiggles,
                            causal=causal)

    # Construct AKM and sample
    akm = cgpcm.AKM(**pars)
    akm.sample(t)
    for i in range(resample):
        akm.sample_f(t)

    # Construct data
    f = Data(t, akm.f())
    k = Data(tk, akm.k(tk))
    i = nearest_index(tk, k_len / 5)
    h = Data(tk, akm.h(tk, assert_positive_at_index=i))

    # Normalise
    f -= f.mean
    f /= f.std
    h /= max(k.y) ** .5
    k /= max(k.y)
    return f, k, h, pars