import util
import eq
from par import Parametrisable
from dist import Normal
from kernel import DEQ
from sample import ESS
from tfutil import *
from util import *


def cgpcm_pars(sess,
               t,
               y,
               nx,
               nh,
               k_len,
               k_wiggles,
               causal,
               causal_id=False):
    """
    Create parameters for the CGPCM.

    :param sess: TensorFlow session
    :param t: times
    :param y: observations
    :param nx: number of inducing points for noise
    :param nh: number of inducing points for filter
    :param k_len: length of kernel
    :param k_wiggles: number of wiggles in kernel
    :param causal: boolean that indicates whether the model is causal
    :param causal_id: boolean that indicates whether the interdomain
                      transformation is causal
    :return: parameters
    """
    # Equal prior power
    if causal:
        k_len *= 2

    # Hyperparameters
    alpha = length_scale(k_len)
    gamma = length_scale(float(k_len) / k_wiggles)

    # Hyperparameter for x and inducing points for x
    if nx > 0:
        omega = length_scale((max(t) - min(t)) / nx)
        tx = np.linspace(min(t), max(t), nx)

    # Make sure of an odd number of inducing points for h, so as to have one
    # at zero
    if nh % 2 == 0:
        nh += 1

    # Inducing points for h
    if causal:
        th = np.linspace(0, 2 * k_len, nh)
        dth = th[1] - th[0]
        th -= dth * 3
    else:
        th = np.linspace(-2 * k_len, 2 * k_len, nh)

    pars = {'sess': sess,
            't': tf.constant(t),
            'y': tf.constant(y),
            'th': tf.constant(th),
            's2': var_pos(to_float(1e-4), 's2'),
            's2_f': var_pos(to_float(1), 's2_f'),
            'alpha': var_pos(to_float(alpha), 'alpha'),
            'gamma': var_pos(to_float(gamma), 'gamma'),
            'causal': causal,
            'causal_id': causal_id}
    if nx > 0:
        pars['tx'] = tf.constant(tx)
        pars['omega'] = var_pos(to_float(omega), 'omega')

    initialise_uninitialised_variables(sess)
    return pars


class CGPCM(Parametrisable):
    """
    Causal Gaussian Process Convolution Model.
    """

    _required_pars = ['sess', 't', 'y',
                      'th', 'tx',
                      's2', 's2_f', 'alpha', 'gamma', 'omega',
                      'causal', 'causal_id']

    def __init__(self, **kw_args):
        Parametrisable.__init__(self, **kw_args)
        self.nh, self.nx = shape(self.th)[0], shape(self.tx)[0]
        self._precomputed = False
        self._init_kernels()
        self._init_var_map()
        self._init_vars()
        self._init_expressions()

    def _init_expressions(self):
        kh = eq.kh_constructor(self.alpha, self.gamma)
        kxs = eq.kxs_constructor(self.omega)
        self.expq_a = kh(self.t1 - self.tau1, self.t2 - self.tau1)
        self.expq_Ahh = kh(self.t1 - self.tau1, self.th1) \
                        * kh(self.th2, self.t2 - self.tau1)
        self.expq_Axx = kh(self.t1 - self.tau1, self.t2 - self.tau2) \
                        * kxs(self.tau1, self.tx1) \
                        * kxs(self.tx2, self.tau2)
        self.expq_Ahx = kh(self.t1 - self.tau1, self.th1) \
                        * kxs(self.tau1, self.tx1)

    def _init_vars(self):
        for var in ['tau1', 'tau2', 't1', 't2', 'th1', 'th2',
                    'tx1', 'tx2', 'min_t1_t2', 'min_t1_0',
                    'min_t1_tx1', 'min_t1_tx2']:
            setattr(self, var, eq.var(var))

    def _init_var_map(self):
        self.var_map = {'th1': expand_dims(self.th, 2, 3),
                        'th2': expand_dims(self.th, 3, 2),
                        'tx1': expand_dims(self.tx, 4, 1),
                        'tx2': expand_dims(self.tx, 5, 0)}

    def _generate_var_map(self, t):
        var_map = self.var_map
        var_map['t1'] = expand_dims(t, 0, 5)
        var_map['t2'] = expand_dims(t, 1, 4)
        var_map['min_t1_t2'] = tf.minimum(var_map['t1'], var_map['t2'])
        var_map['min_t1_0'] = tf.minimum(var_map['t1'], 0)
        var_map['min_t1_tx1'] = tf.minimum(var_map['t1'], var_map['tx1'])
        var_map['min_t1_tx2'] = tf.minimum(var_map['t1'], var_map['tx2'])
        return var_map

    def _int_tau(self, t, exp, t1, t2, upper):
        return exp.substitute('t1', t1).substitute('t2', t2) \
            .integrate_box(('tau1', -inf, upper),
                           **self._generate_var_map(t))

    def _int_tau2(self, t, exp, t1, t2, upper1, upper2):
        return exp.substitute('t1', t1).substitute('t2', t2) \
            .integrate_box(('tau1', -inf, upper1),
                           ('tau2', -inf, upper2),
                           **self._generate_var_map(t))

    def _a(self, t):
        return self._int_tau(t, self.expq_a, self.t1, self.t2,
                             upper=self.min_t1_t2 if self.causal else inf)

    def _a_diag(self, t):
        return self._int_tau(t, self.expq_a, self.t1, self.t1,
                             upper=self.t1 if self.causal else inf)

    def _a_center(self, t):
        return self._int_tau(t, self.expq_a, self.t1, eq.const(0),
                             upper=self.min_t1_0 if self.causal else inf)

    def _Axx_diag(self, t):
        if self.causal:
            if self.causal_id:
                upper1 = self.min_t1_tx1
                upper2 = self.min_t1_tx2
            else:
                upper1 = self.t1
                upper2 = self.t2
        else:
            upper1 = inf
            upper2 = inf
        return self._int_tau2(t, self.expq_Axx, self.t1, self.t1,
                              upper1=upper1, upper2=upper2)

    def _Ahh(self, t):
        return self._int_tau(t, self.expq_Ahh, self.t1, self.t2,
                             upper=self.min_t1_t2 if self.causal else inf)

    def _Ahh_diag(self, t):
        return self._int_tau(t, self.expq_Ahh, self.t1, self.t1,
                             upper=self.t1 if self.causal else inf)

    def _Ahh_center(self, t):
        return self._int_tau(t, self.expq_Ahh, self.t1, eq.const(0),
                             upper=self.min_t1_0 if self.causal else inf)

    def _Ahx(self, t):
        # Variable t2 is not present in Ahx
        if self.causal:
            if self.causal_id:
                upper = self.min_t1_tx1
            else:
                upper = self.t1
        else:
            upper = inf
        return self._int_tau(t, self.expq_Ahx, self.t1, self.t2, upper=upper)

    def _run(self, *args, **kw_args):
        return self.sess.run(*args, **kw_args)

    def _construct(self):
        # Some frequently accessed quantities
        self.n = shape(self.t)[0]
        self.sum_y2 = sum(self.y ** 2)
        self.mats = self._construct_model_matrices(self.t, self.y)
        # Precompute stuff that is not going to change
        self.sum_y2 = self._run(self.sum_y2)

    def _init_kernels(self):
        # Stuff related to h
        self.kernel_h = DEQ(s2=self.s2, alpha=self.alpha, gamma=self.gamma)
        kh = reg(self.kernel_h(self.th[:, None]))
        self.Lh = tf.cholesky(kh)
        self.iKh = cholinv(self.Lh)
        self.h_prior = Normal(self.iKh)
        # Stuff related to x
        self.kernel_x = DEQ(s2=(.5 * np.pi / self.omega) ** .5,
                            alpha=0,
                            gamma=.5 * self.omega)
        self.Kx = reg(self.kernel_x(self.tx[:, None]))
        self.Lx = tf.cholesky(self.Kx)
        self.iKx = cholinv(self.Lx)
        self.x_prior = Normal(self.iKx)
        # Precompute stuff that is not going to change
        self.Kx, self.Lx, self.iKx = self._run([self.Kx, self.Lx, self.iKx])

    def _construct_model_matrices(self, t, y):
        n = shape(t)[0]
        iKx_t = tile(self.iKx, n)
        iKh_t = tile(self.iKh, n)
        mats = dict()
        mats['a'] = self._a_diag(t)[0]  # Not per data point
        mats['Axx'] = self._Axx_diag(t)
        mats['Ahh'] = self._Ahh_diag(t)[0, :, :]  # Also not per data point
        mats['Ahx'] = self._Ahx(t)
        mats['sum_a'] = n * mats['a']
        mats['sum_Axx'] = sum(mats['Axx'], 0)
        mats['sum_Ahh'] = n * mats['Ahh']
        if y is not None:
            mats['sum_Ahx_y'] = sum(y[:, None, None] * mats['Ahx'], 0)
        mats['b'] = (mats['a']
                     - trmul(self.iKh, mats['Ahh'])
                     - trmul(iKx_t, mats['Axx'])
                     + trmul(mul(iKh_t, mats['Ahx']),
                             mul(mats['Ahx'], iKx_t)))
        mats['Bxx'] = (mats['Axx'] - mul3(mats['Ahx'],
                                          iKh_t,
                                          mats['Ahx'], adj_a=True))
        mats['Bhh'] = (tile(mats['Ahh'], n) - mul3(mats['Ahx'],
                                                   iKx_t,
                                                   mats['Ahx'], adj_c=True))
        mats['sum_b'] = (mats['sum_a']
                         - trmul(self.iKh, mats['sum_Ahh'])
                         - trmul(self.iKx, mats['sum_Axx'])
                         + sum(trmul(mul(iKh_t, mats['Ahx']),
                                     mul(mats['Ahx'], iKx_t))))
        mats['sum_Bxx'] = (mats['sum_Axx']
                           - sum(mul3(mats['Ahx'],
                                      iKh_t,
                                      mats['Ahx'], adj_a=True), 0))
        mats['sum_Bhh'] = (mats['sum_Ahh']
                           - sum(mul3(mats['Ahx'],
                                      iKx_t,
                                      mats['Ahx'], adj_c=True), 0))
        return mats

    def precompute(self, recompute=False):
        """
        Precompute matrices.

        :param recompute: boolean that indicates whether to recompute matrices
                          if they are already precomputed
        """
        if recompute and self._precomputed:
            self.undo_precompute()
        if not self._precomputed:
            self.mats_symbolic = dict(self.mats)
            self.mats = {k: self._run(v) for k, v in self.mats.items()}

    def undo_precompute(self):
        """
        Revert precomputation of matrices.
        """
        if self._precomputed:
            self.mats = self.mats_symbolic
            self._precomputed = False


class AKM(CGPCM):
    """
    Approximate Kernel Model.
    """
    _required_pars = ['sess',
                      'th',
                      's2', 's2_f', 'alpha', 'gamma',
                      'causal', 'causal_id']

    def prior_kernel(self, t, iters=1000, psd=False, granularity=2.5):
        """
        Prior distribution over kernels or PSDs.

        :param t: point to evaluate kernel at
        :param iters: number of Monte-Carlo iterations
        :param psd: compute PSD instead
        :param granularity: delta probability at which to generate contours of
                            marginal probability
        :return: mean, list of lower bounds, and list of upper bounds
        """
        n = shape(t)[0]
        Ahh, a = self._run([self._Ahh_center(t),
                            self._a_center(t)])

        h = self.h_prior.sample()
        k = (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        if psd:
            ks = np.concatenate([util.psd(self._run(k), axis=0)
                                 for i in range(iters)], axis=1)
        else:
            ks = np.concatenate([self._run(k) for i in range(iters)], axis=1)

        mu = np.mean(ks, axis=1)
        lowers, uppers = [], []
        for perc in np.arange(granularity, 50 - granularity, granularity):
            lowers.append(np.percentile(ks, perc, axis=1))
            uppers.append(np.percentile(ks, 100 - perc, axis=1))

        return mu, lowers, uppers

    def sample_h(self):
        """
        Sample filter.
        """
        self.h_draw = self._run(self.h_prior.sample())

    def sample_f(self, t):
        """
        Sample function

        :param t: points to evaluate process at
        """
        self.t = t
        self.e_draw = self._run(randn([shape(t)[0], 1]))

    def sample(self, t):
        """
        Sample filter and function.

        :param t: points to evaluate function at
        """
        self.sample_h()
        self.sample_f(t)

    def f(self):
        """
        Construct function

        :return: function
        """
        n = shape(self.t)[0]
        Ahh = self._Ahh(self.t)
        a = self._a(self.t)
        K = reg(a + trmul(tile(outer(self.h_draw) - self.iKh, [n, n]), Ahh))
        f = tf.squeeze(mul(tf.cholesky(K), self.e_draw))
        return self._run(f)

    def h(self, t, assert_positive_at_index=None):
        """
        Construct filter.

        :param t: points to evaluate process at
        :param assert_positive_at_index: index at which to assert that the
                                         process is positive
        :return: filter
        """
        Kfu = self.kernel_h(t, self.th[:, None])
        h = self._run(mul(Kfu, self.h_draw))

        # Set h to correct sign
        if assert_positive_at_index is None:
            assert_positive_at_index = (shape(h)[0] - 1) / 2
        h = h * np.sign(h[assert_positive_at_index])

        return h

    def k(self, t):
        """
        Construct the kernel.

        :param t: points to evaluate kernel at
        :return: kernel
        """
        nk = shape(t)[0]
        Ahh_k = self._Ahh_center(t)
        a_k = self._a_center(t)
        k = a_k + trmul(tile(outer(self.h_draw) - self.iKh, nk), Ahh_k)
        return self._run(k)


class VCGPCM(CGPCM):
    """
    Variational inference in the CGPCM.
    """

    def __init__(self, **kw_args):
        CGPCM.__init__(self, **kw_args)
        self._construct()
        self._init_inducing_points()

    def _init_inducing_points(self):
        mean_init = tf.Variable(self.h_prior.sample(), name='muh')
        var_init = tf.Variable(tf.cholesky(self.h_prior.variance), name='Sh')
        tf.variables_initializer([mean_init, var_init])
        self.h = Normal(mul(var_init, var_init, adj_a=True), mean_init)

    def _qz_natural(self, h_mean, h_m2):
        mu = self.s2_f ** .5 / self.s2 * mul(self.mats['sum_Ahx_y'], h_mean,
                                             adj_a=True)
        S = self.mats['sum_Bxx'] + sum(mul3(self.mats['Ahx'],
                                            tile(h_m2, self.n),
                                            self.mats['Ahx'], adj_a=True), 0)
        return mu, self.Kx + self.s2_f / self.s2 * S

    def elbo(self, smf=False):
        """
        Construct the ELBO.

        :param smf: stochastic approximation of SMF approximation via
                    reparametrisation
        :return: ELBO
        """
        if smf:
            # Stochastic approximation via reparametrisation
            h_samp = self.h.sample()
            mu, S = self._qz_natural(h_samp, outer(h_samp))
        else:
            mu, S = self._qz_natural(self.h.mean, self.h.m2)
        L = tf.cholesky(reg(S))

        elbo = (-self.n * tf.log(2 * np.pi * self.s2)
                - log_det(self.Lx)
                - log_det(L)
                + sum(trisolve(L, mu) ** 2)
                - self.sum_y2 / self.s2
                - self.s2_f / self.s2 * (self.mats['sum_b']
                                         + trmul(self.mats['sum_Bhh'],
                                                 self.h.m2))) / 2. \
               - self.h.kl(self.h_prior)

        return tf.Print(elbo,
                        [self.s2, self.s2_f, self.gamma],
                        message='s2, s2_f, gamma = ')

    def predict_k(self, t, samples_h=200, psd=False):
        """
        Predict kernel.

        :param t: points to predict kernel at
        :param samples_h: samples in Monte-Carlo estimation
        :param psd: predict PSD instead
        :return: predicted kernel
        """
        n = shape(t)[0]
        Ahh, a = self._run([self._Ahh_center(t),
                            self._a_center(t)])

        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        # Compute via MC
        h = placeholder(shape(self.h.sample()))
        k = (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        samples_k = [self._run(k, feed_dict={h: sample_h})
                     for sample_h in samples_h]

        # Check whether to predict kernel or PSD
        if psd:
            samples = [util.psd(x / max(x) / len(x), axis=0)
                       for x in samples_k]
        else:
            samples = samples_k

        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        if psd:
            return mu, lower, upper, std
        else:
            s2_f = self._run(self.s2_f)
            return s2_f * mu, s2_f * lower, s2_f * upper, s2_f * std

    def predict_h(self, t, samples_h=200, assert_positive_at_index=None):
        """
        Predict filter.

        :param t: point to predict filter at
        :param samples_h: samples in Monte-Carlo estimation
        :param assert_positive_at_index: index at which to assert that
                                         the filter is positive
        :return: predicted filter
        """
        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        h = placeholder(shape(self.h.sample()))
        Kfu = self.kernel_h(t, self.th[:, None])
        mu = mul(Kfu, h)
        samples = np.concatenate([self._run(mu, feed_dict={h: sample_h})
                                  for sample_h in samples_h], axis=1)

        # Set samples to correct sign
        if assert_positive_at_index is None:
            assert_positive_at_index = (shape(samples[0])[0] - 1) / 2
        samples = [sample * np.sign(sample[assert_positive_at_index])
                   for sample in samples]

        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)
        return mu, lower, upper, std

    def predict_f(self, t, samples_h=50, smf=False):
        """
        Predict function.

        :param t: point at which to predict function
        :param samples_h: samples in Monte-Carlo estimation
        :param smf: boolean that indicates whether to use the SMF approximation
        :return: predicted function
        """
        n = shape(t)[0]
        mats = self._construct_model_matrices(t, None)
        mats = {k: self._run(mats[k]) for k in ['a', 'Ahh', 'Ahx', 'Axx']}

        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        # Construct optimal q(z|u) or q(z)
        h = tf.placeholder(config.dtype, shape(self.h.sample()))
        if smf:
            mu, S = self._qz_natural(h, outer(h))
        else:
            mu, S = self._qz_natural(self.h.mean, self.h.m2)
        L = tf.cholesky(S)
        x = Normal(cholinv(L), tf.cholesky_solve(L, mu))

        # Construct mean
        mu = tf.squeeze(mul3(tile(h, n),
                             mats['Ahx'],
                             tile(x.mean, n), adj_a=True))[:, None]

        # Construct variance
        mh = outer(h) - self.iKh
        mh_t = tile(mh, n)
        mx_t = tile(x.m2 - self.iKx, n)
        m2 = tf.squeeze(mats['a']
                        + trmul(mats['Ahh'], mh)
                        + trmul(mats['Axx'], mx_t)
                        + trmul(mul(mh_t, mats['Ahx']),
                                mul(mats['Ahx'], mx_t)))[:, None]
        var = m2 - mu ** 2

        # Compute via MC
        samples = [self._run([mu, var], feed_dict={h: sample_h})
                   for sample_h in samples_h]
        samples_mu, samples_var = zip(*samples)
        mu = np.mean(np.concatenate(samples_mu, axis=1), axis=1)
        var = np.mean(np.concatenate(samples_var, axis=1), axis=1)

        s2_f = self._run(self.s2_f)
        mu *= np.sqrt(s2_f)
        var *= s2_f
        return mu, mu - 2 * var ** .5, mu + 2 * var ** .5, var ** .5

    def sample(self, iters=200, burn=50, display=False):
        """
        Sample from the posterior over filters.

        :param iters: number of samples
        :param burn: number of samples to burn
        :param display: boolean that indicates whether to display progress
        :return: samples
        """
        h = placeholder(shape(self.h_prior.sample()))
        mu, S = self._qz_natural(h, outer(h))
        L = tf.cholesky(S)

        prior_sample = self.h_prior.sample()
        log_lik = tf.squeeze(-.5 * log_det(L)
                             + .5 * sum(trisolve(L, mu) ** 2)
                             - .5 * self.s2_f / self.s2
                             * mul3(h, self.mats['sum_Bhh'], h, adj_a=True))

        ess = ESS(lambda x: self._run(log_lik, feed_dict={h: x}),
                  lambda: self._run(prior_sample))
        ess.move(self._run(self.h.mean))
        ess.sample(burn, display=display)
        return ess.sample(iters, display=display)
