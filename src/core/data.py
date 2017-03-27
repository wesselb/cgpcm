import scipy.io as sio
import scipy.io.wavfile as sio_wav
import scipy.signal as signal

from tf_util import *
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
        self.x = np.squeeze(x)
        self.n = len(self.x)
        if y is None:
            # Data is missing
            self.y = np.zeros(shape(x))
            self.y.fill(np.nan)
        else:
            self.y = np.squeeze(y)

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
        return Data(self.x[i_in], self.y[i_in]), \
               Data(self.x[i_out], self.y[i_out])

    def smse(self, y):
        """
        Compute the standardised mean squared error for some predictions.

        :param y: predictions
        :return: standardised means squared error
        """
        return util.smse(self._to_y(y), self.y)

    def mll(self, mu, std, x_max=np.inf):
        """
        Compute the mean log loss for some predictions.

        :param mu: mean of predictions
        :param std: standard deviation of predictions
        :param x_max: maximum absolute value of x to evaluate for
        :return: mean log loss
        """
        def process(x):
            d = Data(self.x, self._to_y(x))
            d = d[np.abs(d.x) <= x_max]
            return d.y

        return util.mll(process(mu), process(std), process(self))

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

    def shift(self, delta_x):
        """
        Shift the data by some amount.

        :param delta_x: amount to shift by
        :return: shifted data
        """
        return Data(self.x + delta_x, self.y)

    def autocorrelation(self):
        """
        Compute autocorrelation of the data.

        :return: autocorrelation
        """
        dist = self.x[-1] - self.x[0]
        return Data(np.linspace(-dist, dist, 2 * len(self.x) - 1),
                    np.convolve(self.y[::-1], self.y))

    def fft_db(self):
        """
        Compute the modulus of the DFT in decibel. Assuming that the
        data is evenly spaced.

        :return: amplitude of spectrum in decibel and sampling frequency
        """
        spec = util.fft_db(util.zero_pad(self.y, 1000))
        return Data(util.fft_freq(len(spec)), spec), 1 / self.dx

    def equals_approx(self, other):
        """
        Check whether data set is approximately equal to another data set.

        :param other: other data set
        :return: equal
        """
        return np.allclose(self.x, other.x) and np.allclose(self.y, other.y)

    def at(self, other_x):
        """
        Find the data at some new x values through interpolation.

        :param other_x: new x values
        :return: new data
        """
        return Data(other_x, np.interp(other_x, self.x, self.y))

    def zero_phase(self):
        """
        Transform signal to zero-phase form.

        :return: zero-phase form
        """
        y = np.fft.ifft(np.abs(np.fft.fft(self.y)))
        y = np.real(np.fft.fftshift(y))
        x = self.dx * np.arange(self.n)
        x -= x[self.n / 2]
        return Data(x, y)

    def zero_pad(self, n):
        """
        Zero pad the signal.

        :param n: number of points to add
        :return: zero-padded signal
        """
        x_app = self.dx * np.arange(1, n + 1)
        y_app = np.zeros(n)
        return Data(np.concatenate((self.x[0] - x_app[::-1],
                                    self.x,
                                    self.x[-1] + x_app[::-1])),
                    np.concatenate((y_app, self.y, y_app)))

    def minimum_phase(self):
        """
        Transform signal to minimum-phase form.

        :return: minimum-phase form
        """
        mag = np.abs(np.fft.fft(self.y))
        spec = np.exp(signal.hilbert(np.log(mag)).conj())
        y = np.real(np.fft.ifft(spec))
        return Data(np.linspace(self.x[0], self.x[-1], len(y)), y)

    def interpolate_fft(self, factor=2):
        """
        Use the FFT to interpolate.
        
        :param factor: factor by which to interpolate
        :return: interpolated data
        """
        spec = np.fft.fftshift(np.fft.fft(self.y))
        spec = util.zero_pad(spec, (factor - 1) * (self.n / 2))
        y = np.real(np.fft.ifft(np.fft.fftshift(spec)))
        scale = float(len(y)) / self.n
        return Data(np.linspace(self.x[0], self.x[-1], len(y)), scale * y)

    @property
    def dx(self):
        """
        Assuming that `x` is evenly spaced, that spacing.
        """
        return self.x[1] - self.x[0]

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

    @property
    def len(self):
        """
        Number of data points.
        """
        return len(self.x)

    @property
    def max(self):
        """
        Maximum of data.
        """
        return max(self.y)

    @property
    def min(self):
        """
        Minimum of data.
        """
        return min(self.y)

    @property
    def domain(self):
        """
        Domain of data.
        """
        return min(self.x), max(self.x)

    @property
    def range(self):
        """
        Range of data.
        """
        return min(self.y), max(self.y)


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
        return Data(self.x[item], self.y[item])

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


def load_hrir(n=1000, h_wav_fn='data/KEMAR_R10e355a.wav', resample=0):
    """
    Load HRIR data.

    :param n: number of response data points
    :param h_wav_fn: HRIR WAV file
    :param resample: number of times to resample
    :return: data for function, filter, and noise
    """
    # Load data
    h = Data.from_wav(h_wav_fn)

    # Take `h.len` extra points to avoid transition effects
    t = np.arange(n + h.len) * h.dx
    for i in range(resample + 1):
        x = Data(t, np.random.randn(n + h.len))
    x /= x.std  # Make sure noise is unity variance

    # Convolve and take the right fragment
    f = Data(t, np.convolve(h.y, x.y)).fragment(n, h.len)[0]

    # Compute exact kernel
    k = h.autocorrelation()

    # Normalise
    h /= h.energy ** .5
    k /= k.max
    f /= f.std

    return f, k, h


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
    k /= k.max
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
    tk = np.linspace(-k_stretch * tau_w, k_stretch * tau_w, 301)
    k = akm.k(tk)
    h = akm.h(tk)

    # Normalise
    f -= f.mean
    f /= f.std
    h /= h.energy_causal ** .5
    k /= k.max

    return f, k, h


def load_seaice():
    import pandas as pd

    import datetime

    def year_fraction(date):
        start = datetime.date(date.year, 1, 1).toordinal()

        year_length = datetime.date(date.year + 1, 1, 1).toordinal() - start

        decimal_date = date.year + float(
            date.toordinal() - start) / year_length

        return decimal_date

    ### Read in daily data

    names = ['Year', 'Month', 'Day', 'Extent', 'Missing', 'Source Data']

    df = pd.read_csv('/home/scott/DATA/N_seaice_extent_daily_v2.1.csv',
                     skiprows=2, names=names)

    dec_date = np.zeros(len(df))

    for j, i in enumerate(df.index):
        df1 = df[df.index == i]

        dec_date[j] = year_fraction(
            datetime.date(df1.Year, df1.Month, df1.Day))

    df['decimal_date'] = dec_date
