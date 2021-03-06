from collections import namedtuple
import scipy.io as sio
import scipy.io.wavfile as sio_wav
import scipy.signal as signal
from datetime import datetime
from sklearn import linear_model
import csv
import operator

from tf_util import *
from util import is_iterable
import util
import cgpcm
import kernel


class Data(object):
    """
    A one-dimensional time series.

    :param x: times
    :param y: values, set to None in case of missing data
    """

    def __init__(self, x, y=None):
        self.x = np.squeeze(x)
        self._evenly_spaced = np.isclose(max(np.abs(np.diff(self.x, n=2))), 0)
        self.n = len(self.x)
        if y is None:
            # Data is missing
            self.y = np.zeros(shape(x))
            self.y.fill(np.nan)
        else:
            self.y = np.squeeze(y)

    def _assert_evenly_spaced(self):
        if not self._evenly_spaced:
            raise AssertionError('data must be evenly spaced')

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

        :param length: length(s) of fragment(s)
        :param start: start(s) of fragment(s)
        :return: fragmented data
        """
        # Make length and start into iterables
        if is_iterable(length) != is_iterable(start):
            if is_iterable(length):
                start = [start] * len(length)
            else:
                length = [length] * len(start)
        elif not is_iterable(length):
            length, start = [length], [start]

        # Verify input
        if len(length) != len(start):
            raise ValueError('number of elements of `length` and `start` must '
                             'match')

        # Generate indices of fragments
        inds = [range(s, s + l) for s, l in zip(start, length)]

        # Reduce and split
        return self._split(reduce(operator.add, inds, []))

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

    def autocorrelation(self, substract_mean=False, normalise=False,
                        unbiased=False):
        """
        Compute autocorrelation of the data.

        :param substract_mean: substract mean from data
        :param normalise: normalise to unity at zero lag
        :param unbiased: unbiased estimate
        :return: autocorrelation
        """
        self._assert_evenly_spaced()

        # Set NaNs to zero
        y = self.y
        y[np.isnan(y)] = 0

        if substract_mean:
            y -= np.mean(y)

        if unbiased:
            # Triangular window
            window_half = np.arange(1, self.n + 1)
            window = np.concatenate((window_half[:-1], window_half[::-1]))
        else:
            window = self.n

        ac = Data(np.linspace(-self.max_lag,
                              self.max_lag,
                              2 * self.n - 1),
                  np.convolve(y[::-1], y) / window)

        # Return with appropriate normalisation
        if normalise:
            return ac / ac.max
        else:
            return ac

    def fft_db(self, *args, **kw_args):
        """
        Alias for `core.data.Data.fft` that afterwards applies
        `core.data.Data.db`.
        """
        res = self.fft(*args, **kw_args)
        if type(res) == tuple:
            return res[0].db(), res[1]
        else:
            return res.db()

    def fft(self, split_freq=False, zero_pad=2000, normalise=True):
        """
        Compute the FFT.
        
        :param split_freq: return spectrum in relative frequency and
                           additionally return sampling frequency, else
                           return spectrum in absolute frequency
        :param zero_pad: zero padding
        :param normalise: normalise to preserve power
        :return: log spectrum and possibly sampling frequency
        """
        self._assert_evenly_spaced()

        # Check whether to normalise
        if normalise:
            scale = self.dx
        else:
            scale = 1

        # Perform Fourier transform
        spec = scale * util.fft(util.zero_pad(self.y, zero_pad))

        if split_freq:
            return Data(util.fft_freq(len(spec)), spec), 1 / self.dx
        else:
            return Data(util.fft_freq(len(spec)) / self.dx, spec)

    def abs(self):
        """
        Convert to modulus.
        """
        return Data(self.x, np.abs(self.y))

    def real(self):
        """
        Convert to real.
        """
        return Data(self.x, np.real(self.y))

    def db(self):
        """
        Convert to decibel.
        """
        return Data(self.x, 10 * np.log10(np.abs(self.y)))

    def equals_approx(self, other):
        """
        Check whether data set is approximately equal to another data set.

        :param other: other data set
        :return: equal
        """
        try:
            return np.allclose(self.x, other.x) and np.allclose(self.y,
                                                                other.y)
        except ValueError:
            # If e.g. `other` has is of another shape
            return False

    def not_at(self, other_x):
        """
        Find the data not at some x values

        :param other_x: x values to avoid
        :return: new data
        """
        if type(other_x) == Data:
            other_x = other_x.x
        return self.at(sorted(set(self.x) - set(other_x)))

    def at(self, other_x):
        """
        Find the data at some new x values through linear interpolation.

        :param other_x: new x values
        :return: new data
        """
        if type(other_x) == Data:
            other_x = other_x.x
        return Data(other_x, np.interp(other_x, self.x, self.y))

    def zero_phase(self):
        """
        Transform signal to zero-phase form.

        :return: zero-phase form
        """
        self._assert_evenly_spaced()
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
        self._assert_evenly_spaced()
        x_app = self.dx * np.arange(1, n + 1)
        y_app = np.zeros(n)
        return Data(np.concatenate((self.x[0] - x_app[::-1],
                                    self.x,
                                    self.x[-1] + x_app)),
                    np.concatenate((y_app, self.y, y_app)))

    def minimum_phase(self):
        """
        Transform signal to minimum-phase form.

        :return: minimum-phase form
        """
        self._assert_evenly_spaced()
        mag = np.abs(np.fft.fft(self.y))
        spec = np.exp(signal.hilbert(np.log(mag)).conj())
        y = np.real(np.fft.ifft(spec))
        return Data(self.x, y)

    def interpolate_fft(self, factor=2):
        """
        Use the FFT to interpolate.
        
        :param factor: factor by which to interpolate
        :return: interpolated data
        """
        self._assert_evenly_spaced()
        spec = np.fft.fftshift(np.fft.fft(self.y))
        spec = util.zero_pad(spec, (factor - 1) * (self.n / 2))
        y = np.real(np.fft.ifft(np.fft.fftshift(spec)))
        scale = float(len(y)) / self.n
        return Data(np.linspace(self.x[0], self.x[-1], len(y)), scale * y)

    def window(self, name='hamming'):
        """
        Apply a window.
        
        :param name: name of window
        :return: windowed data
        """
        if name == 'none':
            return self
        w = signal.get_window(name, Nx=self.len)
        return Data(self.x, self.y * w)

    @property
    def evenly_spaced(self):
        """
        Check whether the data is evenly spaced.
        """
        return self._evenly_spaced

    @property
    def dx(self):
        """
        Spacing of data.
        """
        self._assert_evenly_spaced()
        return self.x[1] - self.x[0]

    @property
    def max_lag(self):
        """
        Maximum lag in data.
        """
        self._assert_evenly_spaced()
        return self.x[-1] - self.x[0]

    @property
    def energy(self):
        """
        Energy of data.
        """
        return np.trapz(self.y ** 2, self.x)

    @property
    def mean(self):
        """
        Mean of data.
        """
        return np.nanmean(self.y)

    @property
    def std(self):
        """
        Standard deviation of data.
        """
        return np.nanstd(self.y, ddof=1)

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
            raise ValueError('data objects must be compatible')

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

    def __pow__(self, power, modulo=None):
        return Data(self.x, self.y.__pow__(power, modulo))

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
        K = reg(kernel(t))
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


# Named tuple for bundling predictions
UncertainData = namedtuple('UncertainData', 'mean lower upper std')


def load_hrir(n=1000, h_wav_fn='./data/KEMAR/elev20/R20e280a.wav', resample=0):
    """
    Load HRIR data.

    :param n: number of response data points
    :param h_wav_fn: HRIR WAV file
    :param resample: number of times to resample
    :return: data for function, filter, and noise
    """
    # Load data
    h = Data.from_wav(h_wav_fn)

    # Clean filter using heuristics
    h = h[h.x <= 4e-3]  # Get rid of noisy tail

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


def load_timit_tobar2015(n=350):
    """
    Load TIMIT data set from Tobar et al. (2015).

    :param n: number of points to subsample; the paper uses 350
    :return: subsampled data and full data
    """
    mat = sio.loadmat('data/TIMIT_unknown.mat')
    y = np.squeeze(mat['audio'])
    fs = np.squeeze(mat['Fs']).astype(float)
    t = np.arange(shape(y)[0]) / fs
    f = Data(t, y).fragment(1750, start=11499)[0]
    f = (f - f.mean) / f.std
    e = f.subsample(n)[0]
    return e, f


def load_timit_voiced_fricative():
    """
    Load a voiced fricative.
    
    :return: data
    """
    d = Data.from_wav('data/TIMIT_SX174_2.915_2.975.wav')
    d = (d - d.mean) / d.std
    return d


def load_gp_exp(sess, n=250, k_len=.1):
    """
    Sample from a GP with an exponential kernel.

    :param sess: TensorFlow session
    :param n: number of time points
    :param k_len: time constant of kernel
    :return: data for function and kernel
    """
    t = util.vlinspace(0, 1, n)
    tk = util.vlinspace(0, 1, 301)
    k_fun = kernel.Exponential(s2=1., gamma=1 / k_len)
    f = Data.from_gp(sess, k_fun, t)
    k = Data(tk, sess.run(k_fun(tk, np.array([[0.]]))))

    # Normalise
    f -= f.mean
    f /= f.std
    k /= k.max
    return f, k


def load_gp_eq(sess, n=250, k_len=.1):
    """
    Sample from a GP with an exponentiated quadratic kernel.

    :param sess: TensorFlow session
    :param n: number of time points
    :param k_len: time constant of kernel
    :return: data for function and kernel
    """
    t = util.vlinspace(0, 1, n)
    tk = util.vlinspace(0, 1, 301)
    k_fun = kernel.DEQ(s2=1.,
                       gamma=util.length_scale(k_len),
                       alpha=0.)
    f = Data.from_gp(sess, k_fun, t)
    k = Data(tk, sess.run(k_fun(tk, np.array([[0.]]))))

    # Normalise
    f -= f.mean
    f /= f.std
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

    # Discard negative part of filter if model is causal
    if causal:
        h = h.positive_part()

    # Normalise
    f -= f.mean
    f /= f.std
    h /= h.energy ** .5
    k /= k.max

    return f, k, h


def load_seaice():
    """
    Load the sea ice data set.
    
    :return: data
    """
    x, y = [], []

    with open('data/seaice_extent_daily_v2.1.csv') as f:
        it = iter(csv.reader(f))

        # First two lines are headers
        next(it), next(it)

        for line in it:
            year, month, day, extent, _, _ = line

            # Convert date to decimal year
            date = datetime(int(year), int(month), int(day))
            decimal_year = util.date_to_decimal_year(date)

            # Add values
            x.append(decimal_year)
            y.append(float(extent))

    # Convert to data and normalise
    d = Data(x, y)
    d -= d.mean
    d /= d.std

    return d


def load_co2():
    """
    Load CO2 data.
    
    :return: CO2 data
    """
    x, y = [], []

    # Load file
    with open('data/co2_mm_mlo.txt') as f:
        for line in f:
            if not line.strip().startswith('#'):
                _, _, date, _, interpolated, _, _ = line.strip().split()
                x.append(float(date))
                y.append(float(interpolated))

    x, y = np.array(x), np.array(y)

    # Correct x: they are just not numerically evenly spaced
    x = np.linspace(x[0], x[-1], len(x))

    # Detrend
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(x[:, None], y)
    y -= lin_reg.predict(x[:, None])

    # Normalise
    d = Data(x, y)
    d = (d - d.mean) / d.std

    return d


def load_hydrochem():
    """
    Load four hydrochemical data sets that describe the concentration of Cl
    over time.
    
    :return: data sets
    """
    feature = 'cl mg/l'
    loc_sites = [('bcl', 'lower hafren daily chloride'),
                 ('ccl', 'rainfall daily chloride'),
                 ('kcl', 'tanllwyth daily chloride'),
                 ('a', 'lower hore')]

    # Read file
    with open('data/PlynlimonResearchCatchmentHydrochemistryData.csv') as f:
        it = iter(f)
        header = [x.lower() for x in next(it).strip().split(',')]
        lines = [line.strip().split(',') for line in it]

    # Convert lines to values
    data = {field: serie for field, serie in zip(header, zip(*lines))}

    # Convert date times and create decimal years
    data['date_time'] = [datetime.strptime(x, '%d/%m/%y %H:%M')
                         for x in data['date_time']]
    # Create data and normalise
    xs_full = [util.date_to_decimal_year(x) for x in data['date_time']]
    ys_full = [float(x) for x in data[feature] if util.is_numeric(x)]

    ds = []
    for loc, site in loc_sites:
        # Filter by location and site
        xs, ys = zip(*[(x, y)
                       for x, y, l, s
                       in zip(xs_full,
                              ys_full,
                              data['location'],
                              data['site name'])
                       if l.lower() == loc and s.lower() == site])

        # Create data and normalise
        d = Data(xs, ys)
        d = (d - d.mean) / d.std
        ds.append(d)

    return ds


def load_crude_oil():
    """
    Load crude oil prices from 2000 to 2017.
    
    :return: data
    """
    xs, ys = [], []

    # Read file
    with open('data/crude_oil.csv') as f:
        reader = iter(csv.reader(f))
        header = next(reader)

        for row in reader:
            month, day, year, price, _, _, _, _, _ = row

            # Day has a comma appended
            day = day[:-1]

            # Convert to decimal year
            x = datetime.strptime('{}-{}-{}'.format(year, month, day),
                                  '%Y-%b-%d')
            xs.append(util.date_to_decimal_year(x))

            # Append price
            ys.append(float(price))

    # Reverse data
    xs, ys = xs[::-1], ys[::-1]

    # Convert to data
    d = Data(xs, ys)

    # Extract relatively flat period
    d = d[np.logical_and(d.x >= 2010, d.x <= 2014)]

    # Normalise and return
    d = (d - d.mean) / d.std

    return d


def load_eeg():
    """
    Load EEGs from healty person.
    
    :return: dictionary with data sets
    """
    ys = {}

    # Read file
    with open('data/co2c0000337.rd.000') as f:
        it = iter(f)

        # Eat header lines
        [next(it) for _ in range(5)]

        # Parse data
        for line in it:
            trial, sensor, num, value = line.strip().split()
            try:
                ys[sensor].append(float(value))
            except KeyError:
                ys[sensor] = [float(value)]

    # Convert to data and normalise
    ds = {}

    for k, v in ys.items():
        d = Data(np.arange(len(v)).astype(float), v)
        d -= d.mean
        d /= d.std
        ds[k] = d

    return ds


def load_currency():
    """
    Load exchange rates.
    
    :return: dictionary with data sets
    """
    ys = {}
    xs = []

    # Read file
    with open('data/currency.csv') as f:
        it = iter(csv.reader(f))

        next(it)  # First line is title
        header = next(it)[3:]  # Second line is header

        # Initialise columns
        for col in header:
            ys[col] = []

        # Parse data
        for line in it:
            day, date = line[:2]

            # Convert date to decimal year
            date = datetime.strptime(date, '%Y/%m/%d')
            dec_year = util.date_to_decimal_year(date)

            # Append x value
            xs.append(dec_year)

            # Append y values
            for col, val in zip(header, line[3:]):
                ys[col].append(float(val) if util.is_numeric(val) else np.nan)

    # Convert to data
    ds = {}
    for k, y in ys.items():
        d = Data(xs, y)

        # Remove missing values
        d = d[~np.isnan(d.y)]

        # Normalise
        d -= d.mean
        d /= d.std

        ds[k] = d

    return ds


def load_solar():
    head = 70
    x, y, y_bg = [], [], []

    # Parse file
    with open('data/lean2000_irradiance.txt') as f:
        reader = csv.reader(f, delimiter=' ')

        # Skip header
        for i in range(head):
            next(reader)

        # Read
        for line in reader:
            # Skip empty lines
            if not line:
                continue

            # Append current line
            year, mean, mean_bg = map(float, filter(None, line))
            x.append(year)
            y.append(mean)
            y_bg.append(mean_bg)

    # Convert to data
    d, d_bg = Data(x, y), Data(x, y_bg)

    # Take years after 1750
    d = d[d.x >= 1750]
    d_bg = d_bg[d_bg.x >= 1750]

    # Normalise
    d -= d.mean
    d /= d.std
    d_bg -= d_bg.mean
    d_bg /= d_bg.std

    return d, d_bg