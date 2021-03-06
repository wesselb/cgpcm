from operator import add

from learn import map_progress
from parametrisable import Parametrisable
from distribution import Normal
from kernel import DEQ
from sample import ESS
from tf_util import *
from util import inf, length_scale, is_numeric, lower_perc, upper_perc
import data
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
                    causal_id=False, noise_init=1e-4, tx_range=None):
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
        :param tx_range: range of the inducing points for x, is taken from `e`
                         by default
        :return: :class:`core.cgpcm.CGPCM` instance
        """
        # Trainable variables
        vars = {}

        # Config
        tau_ws = 1
        causal_extra_points = 2

        # Acausal parameters
        alpha = 2 * length_scale(tau_w)
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
            tx_range = (min(e.x), max(e.x)) if tx_range is None else tx_range
            dtx = (tx_range[1] - tx_range[0]) / nx
            omega = .5 * length_scale(dtx)
            tx = constant(np.linspace(tx_range[0], tx_range[1], nx))
        else:
            omega = np.nan
            tx = constant([])
        omega, vars['omega'] = var_pos(to_float(omega))

        # If acausal, ensure odd number of inducing points so as to have one at
        # zero
        if not causal and nh % 2 == 0:
            nh += 1

        # Inducing points for filter
        if causal:
            th = np.linspace(0, 2 * tau_ws * tau_w, nh)
            dth = th[1] - th[0]
            # Extra inducing points to account for derivatives at zero
            th -= dth * causal_extra_points
            th = constant(th)
        else:
            th = constant(np.linspace(-tau_ws * tau_w, tau_ws * tau_w, nh))

        # Initial observation noise
        s2, vars['s2'] = var_pos(to_float(noise_init))

        # Initialise variables and construct
        sess.run(tf.variables_initializer(vars.values()))

        return cls(sess=sess, th=th, tx=tx, s2=s2, s2_f=s2_f, alpha=alpha,
                   gamma=gamma, omega=omega, vars=vars,

                   # Store recipe for reconstruction
                   e=e, causal=causal, causal_id=causal_id, tau_w=tau_w,
                   tau_f=tau_f, nh=nh, nx=nx, noise_init=noise_init,
                   tx_range=tx_range)

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

    def _init_kernels(self):
        # Stuff related to h
        self.kernel_h = DEQ(s2=1., alpha=self.alpha, gamma=self.gamma)
        self.Kh = reg(self.kernel_h(self.th))
        self.Lh = tf.cholesky(self.Kh)
        self.iKh = cholinv(self.Lh)
        self.h_prior = Normal(reg(self.iKh))

        # Stuff related to x
        self.kernel_x = DEQ(s2=(.5 * np.pi / self.omega) ** .5,
                            alpha=0,
                            gamma=.5 * self.omega)
        self.Kx = reg(self.kernel_x(self.tx))
        self.Lx = tf.cholesky(self.Kx)
        self.iKx = cholinv(self.Lx)
        self.x_prior = Normal(reg(self.iKx))

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
            ks, vs = zip(*self.mats.items())
            vs = self._run(vs)
            self.mats = {k: v for k, v in zip(ks, vs)}
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

    def k_prior(self, t, iters=1000, psd=False, granularity=1):
        """
        Prior distribution over kernels or PSDs (via the kernel approximation).

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

        # Sample kernel
        h = self.h_prior.sample()
        k = self.s2_f * (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        samples = map_progress(lambda x: self._run(k), range(iters),
                               name='sampling kernel')

        # Check whether PSD must be computed
        if psd:
            samples = map_progress(lambda x: data.Data(t, x).fft().y[:, None],
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
            x = data.Data(t).fft().x
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

        if self.causal:
            return data.Data(t, h).positive_part()
        else:
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
        # Mean and variance of q(u)
        mean_init = tf.Variable(self.h_prior.sample())
        self.vars['mu_u'] = mean_init
        var_init = tf.Variable(tril_to_vec(tf.cholesky(self.h_prior.var)))
        self.vars['var_u'] = var_init
        self._run(tf.variables_initializer([mean_init, var_init]))

        # Construct q(u)
        var_init = vec_to_tril(var_init)
        self.h = Normal(reg(mul(var_init, var_init, adj_b=True)), mean_init)

        # Mean and variance of q(z)
        mean_init = tf.Variable(self.x_prior.sample())
        self.vars['mu_z'] = mean_init
        var_init = tf.Variable(tril_to_vec(tf.cholesky(self.x_prior.var)))
        self.vars['var_z'] = var_init
        self._run(tf.variables_initializer([mean_init, var_init]))

        # Construct q(u)
        var_init = vec_to_tril(var_init)
        self.x = Normal(reg(mul(var_init, var_init, adj_b=True)), mean_init)

    def _optimal_q(self, h_mean, h_m2, z=True):
        """
        Compute the natural parameters of the optimal :math:`q(z)`
        (:math:`q(u)`).
        
        :param h_mean: mean of :math:`q(u)` (:math:`q(z)`)
        :param h_m2: second moment of :math:`q(u)` (:math:`q(z)`)
        :param z: compute parameters of of optimal :math:`q(z)`
                  (:math:`q(u)`)
        :return: natural mean and precision of optimal :math:`q(z)`
                 (:math:`q(u)`)
        """
        mu = self.s2_f ** .5 / self.s2 * mul(self.mats['sum_Ahx_y'], h_mean,
                                             adj_a=z)
        B = self.mats['sum_Bxx'] if z else self.mats['sum_Bhh']
        S = B + sum(mul3(self.mats['Ahx'],
                         tile(h_m2, self.n),
                         self.mats['Ahx'], adj_a=z, adj_c=not z), 0)
        K = self.Kx if z else self.Kh
        return mu, K + self.s2_f / self.s2 * S

    def fpi(self, num=50, z=True, high_reg=False):
        """
        Fixed-point iteration on :math:`q(u)` (:math:`q(z)`).
        
        :param num: number of iterations
        :param z: FPI on :math:`q(u)` (:math:`q(z)`)
        :param high_reg: high amount of regularisation
        """

        def scan_fn(prev, cur):
            mean, var = prev
            dist = Normal(var, mean)

            # Optimal q(z) (q(u))
            lam, P = self._optimal_q(dist.mean, dist.m2, z=z)
            if high_reg:
                P = reg(P, diag=1e-4)
            dist = Normal.from_natural(P, lam)

            # Optimal q(u) (q(z))
            lam, P = self._optimal_q(dist.mean, dist.m2, z=not z)
            if high_reg:
                P = reg(P, diag=1e-4)
            dist = Normal.from_natural(P, lam)

            return dist.mean, dist.var

        res = tf.scan(scan_fn, tf.range(num),
                      initializer=(self.h.mean if z else self.x.mean,
                                   self.h.var if z else self.x.var))

        # Compute
        mean, var = self._run([res[0][-1],
                               tril_to_vec(tf.cholesky(res[1][-1]))])

        # Assign result
        self._run([self.vars['mu_u' if z else 'mu_z'].assign(mean),
                   self.vars['var_u' if z else 'var_z'].assign(var)])

    def elbo(self, smf=False, sample=None, z=True):
        """
        Construct the ELBO.

        :param smf: stochastic approximation of SMF approximation
        :param sample: sample to use in SMF approximation
        :param z: use ELBO saturated for :math:`q(z)`
        :return: ELBO and fetches for all terms
        """
        if smf:
            # Stochastic approximation of SMF approximation
            if sample is None:
                sample = self.h.sample() if z else self.x.sample()
            lam, P = self._optimal_q(sample, outer(sample), z=z)
        else:
            lam, P = self._optimal_q(self.h.mean if z else self.x.mean,
                                    self.h.m2 if z else self.x.m2, z=z)
        L = tf.cholesky(reg(P))

        if z:
            trace_term = trmul(self.mats['sum_Bhh'], self.h.m2)
        else:
            trace_term = trmul(self.mats['sum_Bxx'], self.x.m2)

        # Terms of the ELBO
        terms = [{'name': 's2 complexity',
                  'tensor': -.5 * self.n * tf.log(2 * np.pi * self.s2)
                            - .5 * self.sum_y2 / self.s2},

                 {'name': '{} complexity'.format('p(z)' if z else 'p(u)'),
                  'tensor': .5 * log_det(self.Lx if z else self.Lh)},

                 {'name': '{} complexity'.format('q*(z)' if z else 'q*(u)'),
                  'tensor': - .5 * log_det(L)},

                 {'name': '{} fit'.format('q*(z)' if z else 'q*(u)'),
                  'tensor': .5 * sum(trisolve(L, lam) ** 2)},

                 {'name': 'general conditioning penalty',
                  'tensor': - .5 * self.s2_f / self.s2 * self.mats['sum_b']},

                 {'name': '{} conditioning penalty'.format('q(u)'
                                                           if z else 'q(z'),
                  'tensor': - .5 * self.s2_f / self.s2 * trace_term},

                 {'name': '-KL[{}||{}]'.format('q(u)' if 'z' else 'q(z)',
                                               'p(u)' if 'z' else 'p(z)'),
                  'tensor': (-self.h.kl(self.h_prior)
                             if z else -self.x.kl(self.x_prior))}]

        # Add modifiers for terms
        for term in terms:
            term['modifier'] = '.2e'

        # Compute ELBO as sum of terms
        elbo = reduce(add, [term['tensor'] for term in terms], 0)

        return elbo, terms

    def convert(self, z=True):
        """
        Assign :math:`q(z)` (:math:`q(u)`) after optimising :math:`q(u)`
        (:math:`q(z)`).
        
        :param z: assign :math:`q(z)` (:math:`q(u)`)
        """
        lam, P = self._optimal_q(self.h.mean if z else self.x.mean,
                                 self.h.m2 if z else self.x.m2,
                                 z=z)
        dist = Normal.from_natural(P, lam)
        mean, var = dist.mean, tril_to_vec(tf.cholesky(dist.var))

        # Assign conversion
        self._run([self.vars['mu_z' if z else 'mu_u'].assign(mean),
                   self.vars['var_z' if z else 'var_u'].assign(var)])

    def elbo_smf(self, samples_h):
        """
        Estimate the ELBO for the SMF approximation.

        :param samples_h: samples for the filter
        :return: ELBO for the SMF approximation
        """
        sample_h = placeholder(shape(self.h.sample()))
        elbo = self.elbo(smf=True, sample=sample_h)[0]
        elbos = map_progress(lambda x: self._run(elbo,
                                                 feed_dict={sample_h: x}),
                             samples_h,
                             name='estimating ELBO for the SMF approximation '
                                  'using MC')
        return np.mean(elbos), np.std(elbos) / len(samples_h) ** .5

    def predict_k(self, t, samples_h=200, psd=False, normalise=True):
        """
        Predict kernel or PSD (via kernel approximation).

        :param t: points to predict kernel at
        :param samples_h: samples in Monte-Carlo estimation
        :param psd: predict PSD instead
        :param normalise: normalise prediction
        :return: predicted kernel or PSD
        """

        n = shape(t)[0]
        iKh, Ahh, a = self._run([self.iKh,
                                 self._Ahh_center(t),
                                 self._a_center(t)])

        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        # Compute via MC
        h = placeholder(shape(self.h.sample()))
        k = self.s2_f * (a + trmul(tile(outer(h) - iKh, n), Ahh))[:, None]
        samples = map_progress(lambda x: self._run(k, feed_dict={h: x}),
                               samples_h,
                               name='{} prediction using '
                                    'MC'.format('PSD' if psd else 'kernel'))

        # Check whether to normalise predictions
        if normalise:
            samples = [sample / max(sample) for sample in samples]

        # Check whether to predict kernel or PSD
        if psd:
            samples = [data.Data(t, sample).fft().abs().y[:, None]
                       for sample in samples]

        # Determine x axis
        if psd:
            x = data.Data(t).fft().x
        else:
            x = t

        # Compute statistics
        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        return data.UncertainData(mean=data.Data(x, mu),
                                  lower=data.Data(x, lower),
                                  upper=data.Data(x, upper),
                                  std=data.Data(x, std))

    def predict_psd(self, t, samples_h=500, normalise=True):
        """
        Predict PSD.

        :param t: points to predict kernel at
        :param samples_h: samples in Monte-Carlo estimation
        :param normalise: normalise prediction
        :return: predicted PSD
        """
        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        h = placeholder(shape(self.h.sample()))

        # Posterior over h
        Kuh = self.kernel_h(self.th, t)
        A = trisolve(self.Lh, Kuh)
        L = tf.cholesky(reg(self.kernel_h(t) - mul(A, A, adj_a=True)))

        # Precompute matrices
        Kuh, L = self._run([Kuh, L])

        sample = mul(Kuh, h, adj_a=True) + mul(L, randn([shape(t)[0], 1]))
        samples = map_progress(lambda x: self._run(sample, feed_dict={h: x}),
                               samples_h,
                               name='PSD prediction using MC')

        # Compute kernels
        samples = [data.Data(t, sample).autocorrelation(normalise=normalise)
                   for sample in samples]

        # Fourier transform kernels
        samples = [sample.fft().abs().y[:, None] for sample in samples]

        # Compute x axis
        x = data.Data(t).autocorrelation(normalise=True).fft().x

        # Compute statistics
        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        return data.UncertainData(mean=data.Data(x, mu),
                                  lower=data.Data(x, lower),
                                  upper=data.Data(x, upper),
                                  std=data.Data(x, std))

    def predict_h(self, t, samples_h=500, normalise=True,
                  phase_transform='minimum_phase'):
        """
        Predict filter.

        :param t: point to predict filter at
        :param samples_h: samples in Monte-Carlo estimation
        :param normalise: normalise prediction
        :param phase_transform: phase transform, must be `None` or a method of
                                :class:`core.data.Data`
        :return: predicted filter
        """
        if is_numeric(samples_h):
            h = self.h.sample()
            samples_h = [self._run(h) for i in range(samples_h)]

        h = placeholder(shape(self.h.sample()))

        # Posterior over h
        Kuh = self.kernel_h(self.th, t)
        A = trisolve(self.Lh, Kuh)
        L = tf.cholesky(reg(self.kernel_h(t) - mul(A, A, adj_a=True)))

        # Precompute matrices
        Kuh, L = self._run([Kuh, L])

        sample = mul(Kuh, h, adj_a=True) + mul(L, randn([shape(t)[0], 1]))
        samples = map_progress(lambda x: self._run(sample, feed_dict={h: x}),
                               samples_h,
                               name='filter prediction using MC '
                                    '(transform: {})'.format(phase_transform))

        def process(sample):
            y = data.Data(t, sample)
            y = y.positive_part() if self.causal else y
            if phase_transform is not None:
                y = getattr(y, phase_transform)()

            # Check whether to normalise predictions
            if normalise:
                y /= y.energy ** .5

            return y.y[:, None]  # Make column vectors again

        # Process samples: normalisation and phase transformation
        samples = [process(sample) for sample in samples]

        # Find corresponding times
        t = data.Data(t).positive_part() if self.causal else data.Data(t)
        if phase_transform is not None:
            t = getattr(t, phase_transform)().x
        else:
            t = t.x

        # Compute statistics
        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        lower = np.percentile(samples, lower_perc, axis=1)
        upper = np.percentile(samples, upper_perc, axis=1)

        return data.UncertainData(mean=data.Data(t, mu),
                                  lower=data.Data(t, lower),
                                  upper=data.Data(t, upper),
                                  std=data.Data(t, std))

    def predict_f(self, t, samples_h=50, precompute=True):
        """
        Predict function.

        The SMF approximation will be used if and only if `samples_h` is
        provided.

        :param t: point at which to predict function
        :param samples_h: samples in Monte-Carlo estimation
        :param precompute: perform precomputation
        :return: predicted function
        """
        n = shape(t)[0]
        mats = self._construct_model_matrices(data.Data(t))

        # Check whether to perform precomputation
        if precompute:
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
            lam, P = self._optimal_q(h, outer(h))
        else:
            lam, P = self._optimal_q(self.h.mean, self.h.m2)
        x = Normal.from_natural(P, lam)

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

        # Compute statistics
        samples_mu, samples_var = zip(*samples)
        mu = np.mean(np.concatenate(samples_mu, axis=1), axis=1)
        var = np.mean(np.concatenate(samples_var, axis=1), axis=1)

        return data.UncertainData(mean=data.Data(t, mu),
                                  lower=data.Data(t, mu - 2 * var ** .5),
                                  upper=data.Data(t, mu + 2 * var ** .5),
                                  std=data.Data(t, var ** .5))

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
        lam, P = self._optimal_q(h, outer(h))
        L = tf.cholesky(P)

        prior_sample = self.h_prior.sample()
        log_lik = tf.squeeze(-.5 * log_det(L)
                             + .5 * sum(trisolve(L, lam) ** 2)
                             - .5 * self.s2_f / self.s2
                             * mul3(h, self.mats['sum_Bhh'], h, adj_a=True))

        ess = ESS(lambda x: self._run(log_lik, feed_dict={h: x}),
                  lambda: self._run(prior_sample))
        ess.move(self._run(self.h.mean))
        ess.sample(burn)
        return ess.sample(iters)
