from learn import map_progress
from parametrisable import Parametrisable
from distribution import Normal
from kernel import DEQ
from sample import ESS
from tf_util import *
from util import inf, length_scale, zero_pad, fft_freq, is_numeric, \
    lower_perc, upper_perc
import data
import util
import exponentiated_quadratic as eq


class CGPCM(Parametrisable):
    """
    Causal Gaussian Process Convolution Model.
    """

    _required_pars = ['sess', 'e',
                      'th', 'tx',
                      's2', 's2_f', 'alpha', 'gamma', 'omega', 'vars',
                      'causal', 'causal_id']

    def __init__(self, **kw_args):
        Parametrisable.__init__(self, **kw_args)
        self._precomputed = False
        self._init_kernels()
        self._init_var_map()
        self._init_vars()
        self._init_expressions()

    @classmethod
    def from_recipe(cls, sess, e, nx, nh, tau_w, tau_f, causal,
                    causal_id=False, noise_init=1e-4):
        """
        Generate parameters for the CGPCM and construct afterwards.

        :param sess: TensorFlow session
        :param e: observations
        :param nx: number of inducing points for noise
        :param nh: number of inducing points for filter
        :param tau_w: length of kernel window
        :param tau_f: length scale of function prior
        :param causal: causal model
        :param causal_id: causal interdomain transformation
        :param noise_init: initialisation of noise
        :return: :class:`core.cgpcm.CGPCM` instance
        """
        # Trainable variables
        vars = {}

        # Config
        k_stretch = 3
        causal_extra_points = 2

        # Acausal parameters
        alpha = length_scale(tau_w)
        gamma = length_scale(tau_f) - .5 * alpha
        s2_f, vars['s2_f'] = var_pos(to_float((2 * alpha / np.pi) ** .5))

        # Update in the causal case
        if causal:
            gamma += 3. * alpha / 8.
            alpha /= 4.

        alpha, vars['alpha'] = var_pos(to_float(alpha))
        gamma, vars['gamma'] = var_pos(to_float(gamma))

        # Hyperparameter and inducing points for noise
        if nx > 0:
            omega = .5 * length_scale((max(e.x) - min(e.x)) / nx)
            tx = constant(np.linspace(min(e.x), max(e.x), nx))
        else:
            omega = to_float(np.nan)
            tx = constant([])
        omega, vars['omega'] = var_pos(to_float(omega))

        # If acausal, ensure odd number of inducing points so as to have one at
        # zero
        if not causal and nh % 2 == 0:
            nh += 1

        # Inducing points for filter
        if causal:
            th = np.linspace(0, 2 * k_stretch * tau_w, nh)
            dth = th[1] - th[0]
            # Extra inducing points to account for derivatives at zero
            th -= dth * causal_extra_points
            th = constant(th)
        else:
            th = constant(np.linspace(-k_stretch * tau_w,
                                      k_stretch * tau_w, nh))

        # Initial observation noise
        s2, vars['s2'] = var_pos(to_float(noise_init))

        # Initialise variables and construct
        initialise_uninitialised_variables(sess)

        return cls(sess=sess, e=e, th=th, tx=tx, s2=s2, s2_f=s2_f, alpha=alpha,
                   gamma=gamma, omega=omega, vars=vars, causal=causal,
                   causal_id=causal_id,
                   # Also set some handy variables
                   tau_w=tau_w, tau_f=tau_f, nh=nh, nx=nx)

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
                upper2 = self.t1
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
        if self.causal:
            if self.causal_id:
                upper = self.min_t1_tx1
            else:
                upper = self.t1
        else:
            upper = inf
        # Variable t2 is not present in Ahx, so substitution can be anything
        return self._int_tau(t, self.expq_Ahx, self.t1, self.t2, upper=upper)

    def _run(self, *args, **kw_args):
        return self.sess.run(*args, **kw_args)

    def _construct(self):
        # Some frequently accessed quantities
        self.n = shape(self.e.x)[0]
        self.sum_y2 = sum(self.e.y ** 2)
        self.mats = self._construct_model_matrices(self.e)

        # Precompute stuff that is not going to change
        self.sum_y2 = self._run(self.sum_y2)

    def _init_kernels(self):
        # Stuff related to h
        self.kernel_h = DEQ(s2=1., alpha=self.alpha, gamma=self.gamma)
        kh = reg(self.kernel_h(self.th))
        self.Lh = tf.cholesky(kh)
        self.iKh = cholinv(self.Lh)
        self.h_prior = Normal(self.iKh)

        # Stuff related to x
        self.kernel_x = DEQ(s2=(.5 * np.pi / self.omega) ** .5,
                            alpha=0,
                            gamma=.5 * self.omega)
        self.Kx = reg(self.kernel_x(self.tx))
        self.Lx = tf.cholesky(self.Kx)
        self.iKx = cholinv(self.Lx)
        self.x_prior = Normal(self.iKx)

        # Precompute stuff that is not going to change
        self.Kx, self.Lx, self.iKx = self._run([self.Kx, self.Lx, self.iKx])

    def _construct_model_matrices(self, e):
        n = shape(e.x)[0]
        iKx_t = tile(self.iKx, n)
        iKh_t = tile(self.iKh, n)
        mats = dict()
        mats['a'] = self._a_diag(e.x)[0]  # Not per data point
        mats['Axx'] = self._Axx_diag(e.x)
        mats['Ahh'] = self._Ahh_diag(e.x)[0, :, :]  # Also not per data point
        mats['Ahx'] = self._Ahx(e.x)
        mats['sum_a'] = n * mats['a']
        mats['sum_Axx'] = sum(mats['Axx'], 0)
        mats['sum_Ahh'] = n * mats['Ahh']
        mats['sum_Ahx_y'] = sum(e.y[:, None, None] * mats['Ahx'], 0)
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

        :param recompute: recompute matrices
                          if they are already precomputed
        """
        if recompute and self._precomputed:
            self.undo_precompute()
        if not self._precomputed:
            self.mats_symbolic = dict(self.mats)
            self.mats = {k: self._run(v) for k, v in self.mats.items()}
            self._precomputed = True

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

    def k_prior(self, t, iters=1000, psd=False, granularity=1, psd_pad=1000):
        """
        Prior distribution over kernels or PSDs.

        :param t: point to evaluate kernel at
        :param iters: number of Monte-Carlo iterations
        :param psd: compute PSD instead
        :param psd_pad: zero padding in the case of PSD
        :param granularity: delta probability at which to generate contours of
                            marginal probability
        :return: mean, list of lower bounds, and list of upper bounds
        """
        n = shape(t)[0]
        Ahh, a = self._run([self._Ahh_center(t),
                            self._a_center(t)])

        # Sample kernel
        h = self.h_prior.sample()
        k = self.s2_f * (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        samples = map_progress(lambda x: self._run(k), range(iters),
                               name='sampling kernel')

        # Check whether PSD must be computed
        if psd:
            samples = map_progress(lambda x: util.fft_db(zero_pad(x, psd_pad),
                                                         axis=0),
                                   samples,
                                   name='transforming samples')
        samples = np.concatenate(samples, axis=1)

        # Compute bounds
        mu = np.mean(samples, axis=1)
        bounds = map_progress(lambda x: (np.percentile(samples, x, axis=1),
                                         np.percentile(samples, 100 - x,
                                                       axis=1)),
                              np.arange(granularity,
                                        50 - granularity,
                                        granularity),
                              name='computing bounds')
        lowers, uppers = zip(*bounds)

        # Determine x axis
        if psd:
            x = fft_freq(shape(samples)[0])
        else:
            x = t

        return data.Data(x, mu), \
               [data.Data(x, lower) for lower in lowers], \
               [data.Data(x, upper) for upper in uppers]

    def sample_h(self, h=None):
        """
        Sample filter.

        :param h: draw for filter, note the parametrisation of the filter
        """
        if h is None:
            self.h_draw = self._run(self.h_prior.sample())
        else:
            self.h_draw = h

    def sample_f(self, t):
        """
        Sample function

        :param t: points to evaluate process at
        """
        self.t = t
        self.e_draw = self._run(randn([shape(t)[0], 1]))

    def sample(self, t, h=None):
        """
        Sample filter and function.

        :param t: points to evaluate function at
        :param h: draw for filter, not the parametrisation of the filter
        """
        self.sample_h(h)
        self.sample_f(t)

    def f(self):
        """
        Construct function.

        :return: function
        """
        n = shape(self.t)[0]
        Ahh = self._Ahh(self.t)
        a = self._a(self.t)
        K = reg(a + trmul(tile(outer(self.h_draw) - self.iKh, [n, n]), Ahh))
        f = self.s2_f ** .5 * tf.squeeze(mul(tf.cholesky(K), self.e_draw))
        return data.Data(self.t, self._run(f))

    def h(self, t):
        """
        Construct filter.

        :param t: points to evaluate process at
        :return: filter
        """
        Kfu = self.kernel_h(t, self.th)
        h = self._run(tf.squeeze(mul(Kfu, self.h_draw)))
        return data.Data(t, h)

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
        return data.Data(t, self._run(self.s2_f * k))


class VCGPCM(CGPCM):
    """
    Variational inference in the CGPCM.
    """

    def __init__(self, **kw_args):
        CGPCM.__init__(self, **kw_args)
        self._construct()
        self._init_inducing_points()

    def _init_inducing_points(self):
        # Mean
        mean_init = tf.Variable(self.h_prior.sample())
        self.vars['mu'] = mean_init

        # Variance
        var_init = tf.Variable(tf.cholesky(self.h_prior.var), name='Sh')
        self.vars['var'] = var_init

        tf.variables_initializer([mean_init, var_init])
        self.h = Normal(mul(var_init, var_init, adj_a=True), mean_init)

    def _qz_natural(self, h_mean, h_m2):
        mu = self.s2_f ** .5 / self.s2 * mul(self.mats['sum_Ahx_y'], h_mean,
                                             adj_a=True)
        S = self.mats['sum_Bxx'] + sum(mul3(self.mats['Ahx'],
                                            tile(h_m2, self.n),
                                            self.mats['Ahx'], adj_a=True), 0)
        return mu, self.Kx + self.s2_f / self.s2 * S

    def elbo(self, smf=False, sample_h=None):
        """
        Construct the ELBO.

        :param smf: stochastic approximation of SMF approximation
        :param sample_h: sample to use in SMF approximation
        :return: ELBO
        """
        if smf:
            # Stochastic approximation of SMF approximation
            if sample_h is None:
                sample_h = self.h.sample()
            mu, S = self._qz_natural(sample_h, outer(sample_h))
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
        return elbo

    def elbo_smf(self, samples_h):
        """
        Estimate the ELBO for the SMF approximation.

        :param samples_h: samples for the filter
        :return: ELBO for the SMF approximation
        """
        sample_h = placeholder(shape(self.h.sample()))
        elbo = self.elbo(smf=True, sample_h=sample_h)
        elbos = map_progress(lambda x: self._run(elbo,
                                                 feed_dict={sample_h: x}),
                             samples_h,
                             name='estimating ELBO for the SMF approximation '
                                  'using MC')
        return np.mean(elbos), np.std(elbos) / len(sample_h) ** .5

    def predict_k(self, t, samples_h=200, psd=False, normalise=True,
                  psd_pad=1000):
        """
        Predict kernel.

        :param t: points to predict kernel at
        :param samples_h: samples in Monte-Carlo estimation
        :param psd: predict PSD instead
        :param normalise: normalise prediction
        :param psd_pad: zero padding in the case of PSD
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
        k = self.s2_f * (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        samples = map_progress(lambda x: self._run(k, feed_dict={h: x}),
                               samples_h,
                               name='{} prediction using '
                                    'MC'.format('PSD' if psd else 'kernel'))

        # Check whether to normalise predictions
        if normalise:
            samples = [sample / max(sample) for sample in samples]

        # Check whether to predict kernel or PSD
        if psd:
            samples = [util.fft_db(zero_pad(sample, psd_pad), axis=0)
                       for sample in samples]

        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        # Determine x axis
        if psd:
            x = fft_freq(shape(samples)[0])
        else:
            x = t

        return data.Data(x, mu), \
               data.Data(x, lower), \
               data.Data(x, upper), \
               data.Data(x, std)

    def predict_h(self, t, samples_h=200, normalise=True, correct_signs=False):
        """
        Predict filter.

        :param t: point to predict filter at
        :param samples_h: samples in Monte-Carlo estimation
        :param normalise: normalise prediction
        :param correct_signs: correct signs of samples
        :return: predicted filter
        """
        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        h = placeholder(shape(self.h.sample()))
        Kfu = self.kernel_h(t, self.th)
        mu = mul(Kfu, h)
        samples = map_progress(lambda x: self._run(mu, feed_dict={h: x}),
                               samples_h,
                               name='filter prediction using MC')
        samples = np.concatenate(samples, axis=1)

        if correct_signs:
            # Set samples to correct sign
            samples *= util.sign_smart(samples.T)[None, :]

        # Compute statistics
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        # Check whether to normalise prediction
        if normalise:
            if self.causal:
                scale = data.Data(t, mu).energy_causal ** .5
            else:
                scale = data.Data(t, mu).energy ** .5
            mu /= scale
            lower /= scale
            upper /= scale
            std /= scale

        return data.Data(t, mu), \
               data.Data(t, lower), \
               data.Data(t, upper), \
               data.Data(t, std)

    def predict_f(self, t, samples_h=50):
        """
        Predict function.

        The SMF approximation will be used if and only if `samples_h` is
        provided.

        :param t: point at which to predict function
        :param samples_h: samples in Monte-Carlo estimation
        :return: predicted function
        """
        n = shape(t)[0]
        mats = self._construct_model_matrices(data.Data(t, None))
        mats = {k: self._run(mats[k]) for k in ['a', 'Ahh', 'Ahx', 'Axx']}

        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]
            smf = False
        else:
            smf = True

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
        mu *= self.s2_f ** .5

        # Construct variance
        mh = outer(h) - self.iKh
        mh_t = tile(mh, n)
        mx_t = tile(x.m2 - self.iKx, n)
        m2 = self.s2_f * tf.squeeze(mats['a']
                                    + trmul(mats['Ahh'], mh)
                                    + trmul(mats['Axx'], mx_t)
                                    + trmul(mul(mh_t, mats['Ahx']),
                                            mul(mats['Ahx'], mx_t)))[:, None]
        var = m2 - mu ** 2

        # Compute via MC
        samples = map_progress(lambda x: self._run([mu, var],
                                                   feed_dict={h: x}),
                               samples_h,
                               name='function prediction using MC')
        samples_mu, samples_var = zip(*samples)
        mu = np.mean(np.concatenate(samples_mu, axis=1), axis=1)
        var = np.mean(np.concatenate(samples_var, axis=1), axis=1)

        return data.Data(t, mu), \
               data.Data(t, mu - 2 * var ** .5), \
               data.Data(t, mu + 2 * var ** .5), \
               data.Data(t, var ** .5)

    def sample(self, iters=200, burn=None):
        """
        Sample from the posterior over filters.

        :param iters: number of samples
        :param burn: number of samples to burn
        :return: samples
        """
        if burn is None:
            burn = iters
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
        ess.sample(burn)
        return ess.sample(iters)
