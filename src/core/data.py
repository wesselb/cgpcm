import scipy.io as sio
import scipy.io.wavfile as sio_wav

from tfutil import *
import util
import cgpcm
import kernel


class Data(object):
    """
    Data.

    :param t: times
    :param y: values, set to None in case of missing data
    """

    def __init__(self, x, y=None):
        self.x = x
        self.n = shape(x)[0]
        if y is None:
            # Data is missing
            self.y = np.zeros(shape(x))
            self.y.fill(np.nan)
        else:
            self.y = y

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

    def filter(self, f):
        """
        Select parts of the data according to a filter.

        :param f: filter
        :return: selected data
        """
        filtered = filter(lambda (i, x): f(x), enumerate(self.x))
        return self._split(zip(*filtered)[0])  # Select indices

    def _split(self, i_in):
        i_in = sorted(i_in)
        i_out = sorted(list(set(range(self.n)) - set(i_in)))
        return Data(self.x[i_in], self.y[i_in]), \
               Data(self.x[i_out], self.y[i_out])

    def smse(self, y):
        """
        Compute the standardised mean squared error for some predictions.

        :param y: predictions
        :return: standardised means squared error
        """
        return util.smse(y, self.y)

    def mll(self, mu, std):
        """
        Compute the mean log loss for some predictions.

        :param mu: mean of predictions
        :param std: standard deviation of predictions
        :return: mean log loss
        """
        return util.mll(mu, std, self.y)

    def make_noisy(self, var):
        """
        Make the data noisy.

        :param var: variance of noise
        :return: noisy data
        """
        noise = var ** .5 * np.random.randn(*shape(self.y))
        return Data(self.x, self.y + noise)

    def positive_part(self):
        """
        Get part of data associated to positive times.

        :return: positive part of data
        """
        return self._split(self.x >= 0)[0]

    @property
    def energy(self):
        """
        Energy of data.
        """
        return np.trapz(self.y ** 2, self.x)

    @property
    def energy_causal(self):
        """
        Energy of data, assuming that the data is causal.
        """
        return self.positive_part().energy

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

    def _assert_compat(self, other):
        if not np.allclose(self.x, other.x):
            raise ValueError('Data objects must be compatible')

    def _to_y(self, other):
        if type(other) is Data:
            self._assert_compat(other)
            return other.y
        else:
            return other

    def __neg__(self):
        return Data(self.x, -self.y)

    def __add__(self, other):
        return Data(self.x, self.y + self._to_y(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return Data(self.x, self.y - self._to_y(other))

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __div__(self, other):
        return Data(self.x, self.y / self._to_y(other))

    def __rdiv__(self, other):
        return 1 / self.__div__(other)

    def __mul__(self, other):
        return Data(self.x, self.y * self._to_y(other))

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
    dt = h.x[1] - h.x[0]
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
    k_fun = kernel.Exponential(s2=1., gamma=util.length_scale(k_len))
    f = Data.from_gp(sess, k_fun, t)
    k = Data(tk, k_fun(tk[0, :], np.array([[0]])))

    # Normalise
    f -= f.mean
    k /= max(k.y)
    return f, k


def load_akm(sess, causal, n=250, nh=31, tau_w=.1, tau_f=.05, resample=0):
    """
    Sample from the AKM.

    :param sess: TensorFlow session
    :param causal: causal AKM
    :param n: number of time points
    :param nh: number of inducing points for the filter
    :param tau_w: length kernel window
    :param tau_f: kernel length of function prior
    :param resample: number of times to resample function
    :return: data for function, kernel, and filter, and parameters of the AKM
    """
    # Config
    frac_tau_f_pos = .1
    k_stretch = 8  # Causal case will range from 0 to 6
    e = Data(np.linspace(0, 1, n), None)

    # Construct AKM
    akm = cgpcm.AKM.from_recipe(sess=sess,
                                e=e,
                                nx=0,
                                nh=nh,
                                tau_w=tau_w,
                                tau_f=tau_f,
                                causal=causal)

    # Sample
    akm.sample(e.x)
    for i in range(resample):
        akm.sample_f(e.x)

    # Construct data
    f = akm.f()
    tk = np.linspace(-k_stretch * tau_w,
                     k_stretch * tau_w, 301)
    k = akm.k(tk)
    i = util.nearest_index(tk, frac_tau_f_pos * tau_f)
    h = akm.h(tk, assert_positive_at_index=i)

    # Normalise
    f -= f.mean
    f /= f.std
    h /= h.energy_causal ** .5
    k /= max(k.y)

    # Compute PSD
    psd = util.psd(util.zero_pad(k.y, 1000))
    psd = Data(util.fft_freq(len(psd)), psd)

    return f, k, h, psd
