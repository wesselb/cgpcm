import scipy.stats

from core import expq as expq
from core.dist import Normal
from core.kernel import DEQ
from core.samplers import ESS
import core.utils as utils
from core.utils import *


class GPCM:
    """
    Gaussian Process Convolution Model.
    """

    def __init__(self, pars, th, tx, t, y, sess):
        self.th, self.tx = th, tx
        self.nh, self.nx = shape(th)[0], shape(tx)[0]
        self.pars = pars
        self.t = t
        self.y = y
        self.sess = sess
        self._init_kernels()

        self.var_map = {'th1': expand_dims(th, 2, 3),
                        'th2': expand_dims(th, 3, 2),
                        'tx1': expand_dims(tx, 4, 1),
                        'tx2': expand_dims(tx, 5, 0)}
        self.kh = lambda t1, t2: expq.kh(pars['alpha'], pars['gamma'], t1, t2)
        self.kxs = lambda t1, t2: expq.kxs(pars['omega'], t1, t2)

        # Some variables
        self.tau1 = expq.var('tau1')
        self.tau2 = expq.var('tau2')
        self.tau = expq.var('tau')
        self.t1 = expq.var('t1')
        self.t2 = expq.var('t2')
        self.th1 = expq.var('th1')
        self.th2 = expq.var('th2')
        self.tx1 = expq.var('tx1')
        self.tx2 = expq.var('tx2')
        self.min_t1_t2 = expq.var('min_t1_t2')
        self.min_t1_0 = expq.var('min_t1_0')
        self.min_t1_tx1 = expq.var('min_t1_tx1')
        self.min_t1_tx2 = expq.var('min_t1_tx2')

        # Some expressions
        self.expq_a = self.kh(self.t1 - self.tau, self.t2 - self.tau)
        self.expq_Ahh = self.kh(self.t1 - self.tau, self.th1) \
                        * self.kh(self.th2, self.t2 - self.tau)
        self.expq_Axx = self.kh(self.t1 - self.tau1, self.t2 - self.tau2) \
                        * self.kxs(self.tau1, self.tx1) \
                        * self.kxs(self.tx2, self.tau2)
        self.expq_Ahx = self.kh(self.t1 - self.tau, self.th1) \
                        * self.kxs(self.tau, self.tx1)

    def generate_var_map(self, t):
        var_map = self.var_map
        var_map['t1'] = expand_dims(t, 0, 5)
        var_map['t2'] = expand_dims(t, 1, 4)
        var_map['min_t1_t2'] = tf.minimum(var_map['t1'], var_map['t2'])
        var_map['min_t1_0'] = tf.minimum(var_map['t1'], 0)
        var_map['min_t1_tx1'] = tf.minimum(var_map['t1'], var_map['tx1'])
        var_map['min_t1_tx2'] = tf.minimum(var_map['t1'], var_map['tx2'])
        return var_map

    def _int_tau(self, t, expq, t1, t2, upper):
        return expq.substitute('t1', t1).substitute('t2', t2) \
            .integrate_box(('tau', -inf, upper),
                           **self.generate_var_map(t))

    def _int_tau2(self, t, expq, t1, t2, upper1, upper2):
        return expq.substitute('t1', t1).substitute('t2', t2) \
            .integrate_box(('tau1', -inf, upper1),
                           ('tau2', -inf, upper2),
                           **self.generate_var_map(t))

    def a(self, t, causal):
        return self._int_tau(t, self.expq_a, self.t1, self.t2,
                             upper=self.min_t1_t2 if causal else inf)

    def a_diag(self, t, causal):
        return self._int_tau(t, self.expq_a, self.t1, self.t1,
                             upper=self.t1 if causal else inf)

    def a_center(self, t, causal):
        return self._int_tau(t, self.expq_a, self.t1, expq.const(0),
                             upper=self.min_t1_0 if causal else inf)

    def Axx_diag(self, t, causal, causal_id):
        upper1 = (self.min_t1_tx1 if causal_id else self.t1) if causal else inf
        upper2 = (self.min_t1_tx2 if causal_id else self.t1) if causal else inf
        return self._int_tau2(t, self.expq_Axx, self.t1, self.t1,
                              upper1=upper1, upper2=upper2)

    def Ahh(self, t, causal):
        return self._int_tau(t, self.expq_Ahh, self.t1, self.t2,
                             upper=self.min_t1_t2 if causal else inf)

    def Ahh_diag(self, t, causal):
        return self._int_tau(t, self.expq_Ahh, self.t1, self.t1,
                             upper=self.t1 if causal else inf)

    def Ahh_center(self, t, causal):
        return self._int_tau(t, self.expq_Ahh, self.t1, expq.const(0),
                             upper=self.min_t1_0 if causal else inf)

    def Ahx(self, t, causal, causal_id):
        # Variable t2 is not present in Ahx
        upper = (self.min_t1_tx1 if causal_id else self.t1) if causal else inf
        return self._int_tau(t, self.expq_Ahx, self.t1, self.t2, upper=upper)

    def run(self, *args, **kw_args):
        return self.sess.run(*args, **kw_args)

    def preconstruct(self, causal, causal_id):
        # Some frequently accessed quantities
        self.n = shape(self.t)[0]
        self.s2 = self.pars['s2']
        self.s2_f = self.pars['s2_f']
        self.sum_y2 = sum(self.y ** 2)
        self.mats = self.construct_model_matrices(self.t,
                                                  self.y,
                                                  causal,
                                                  causal_id)

        # Precompute stuff that is not going to change
        self.sum_y2 = self.run(self.sum_y2)

    def _init_kernels(self):
        self.kernel_h = DEQ({'alpha': self.pars['alpha'],
                             'gamma': self.pars['gamma'],
                             's2': 1.})
        self.Kh = reg(self.kernel_h(self.th[:, None]))
        self.Lh = tf.cholesky(self.Kh)
        self.iKh = cholinv(self.Lh)
        self.h_prior = Normal(self.iKh)

        self.kernel_x = DEQ({'alpha': 0,
                             'gamma': .5 * self.pars['omega'],
                             's2': (.5 * np.pi / self.pars['omega']) ** .5})
        self.Kx = reg(self.kernel_x(self.tx[:, None]))
        self.Lx = tf.cholesky(self.Kx)
        self.iKx = cholinv(self.Lx)
        self.x_prior = Normal(self.iKx)

        # Precompute stuff that is not going to change
        self.Kx, self.Lx, self.iKx = self.run([self.Kx, self.Lx, self.iKx])

    def construct_model_matrices(self,
                                 t,
                                 y,
                                 causal,
                                 causal_id):
        n = shape(t)[0]
        iKx_t = tile(self.iKx, n)
        iKh_t = tile(self.iKh, n)
        mats = dict()
        mats['a'] = self.a_diag(t, causal)[0]  # Not per data point
        mats['Axx'] = self.Axx_diag(t, causal, causal_id)
        mats['Ahh'] = self.Ahh_diag(t, causal)[0, :, :]  # Also not per data
        # point
        mats['Ahx'] = self.Ahx(t, causal, causal_id)

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

    def precompute(self):
        self.mats_ops = dict(self.mats)
        self.mats = {k: self.run(v) for k, v in self.mats.items()}

    def undo_precompute(self):
        if hasattr(self, 'mats_ops'):
            self.mats = self.mats_ops


class AKM(GPCM):
    def __init__(self, *args, **kw_args):
        GPCM.__init__(self, *args, **kw_args)

    def prior_kernel(self, t, causal, iters=1000, psd=False):
        n = shape(t)[0]
        Ahh = self.Ahh_center(t, causal)
        a = self.a_center(t, causal)
        # Precompute
        Ahh, a = self.run([Ahh, a])

        h = self.h_prior.sample()
        k = (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        if psd:
            ks = np.concatenate([utils.psd(self.run(k), axis=0)
                                 for i in range(iters)], axis=1)
        else:
            ks = np.concatenate([self.run(k) for i in range(iters)], axis=1)

        gran = 2.5
        percentiles = np.arange(gran, 50 - gran, gran)

        mu = np.mean(ks, axis=1)
        lowers, uppers = [], []
        for perc in percentiles:
            lowers.append(np.percentile(ks, perc, axis=1))
            uppers.append(np.percentile(ks, 100 - perc, axis=1))
        return mu, lowers, uppers


    def generate_sample(self, t, tk):
        self.sample = {'h': self.run(self.h_prior.sample()),
                       'e': self.run(randn([shape(t)[0], 1])),
                       't': t,
                       'tk': tk}

    def construct_sample(self, causal, mean=False):
        h, e = self.sample['h'], self.sample['e']
        t, tk = self.sample['t'], self.sample['tk']
        nk = shape(tk)[0]
        n = shape(t)[0]
        # Kernel function
        Ahh_k = self.Ahh_center(tk, causal)
        a_k = self.a_center(tk, causal)
        if mean:
            k = a_k
        else:
            k = a_k + trmul(tile(outer(h) - self.iKh, nk), Ahh_k)
        # Kernel matrix
        Ahh = self.Ahh(t, causal)
        a = self.a(t, causal)
        if mean:
            K = reg(a)
        else:
            K = reg(a + trmul(tile(outer(h) - self.iKh, [n, n]), Ahh))
        # Function
        f = tf.squeeze(mul(tf.cholesky(K), e))
        return self.run([k, f, K])

    def construct_filter(self, t):
        h = mul(self.Kh, self.sample['h'])
        Kfu = self.kernel_h(t, self.th[:, None])
        h = self.run(mul(Kfu, tf.cholesky_solve(self.Lh, h)))
        # Assume t is symmetric around zero
        return h * np.sign(h[(shape(t)[0] - 1) / 2])

    def construct_kernel(self, causal):
        h, e = self.sample['h'], self.sample['e']
        t, tk = self.sample['t'], self.sample['tk']
        nk = shape(tk)[0]
        Ahh_k = self.Ahh_center(tk, causal)
        a_k = self.a_center(tk, causal)
        k = a_k + trmul(tile(outer(h) - self.iKh, nk), Ahh_k)
        return self.run(k)


class VGPCM(GPCM):
    def __init__(self, *args, **kw_args):
        self.causal = kw_args['causal']
        self.causal_id = kw_args['causal_id']
        del kw_args['causal']
        del kw_args['causal_id']
        GPCM.__init__(self, *args, **kw_args)
        self.preconstruct(self.causal, self.causal_id)
        self._init_inducing_points()

    def _init_inducing_points(self):
        mean_init = tf.Variable(self.h_prior.sample(), name='muh')
        var_init = tf.Variable(tf.cholesky(self.h_prior.variance), name='Sh')
        self.h = Normal(mul(var_init, var_init, adj_a=True), mean_init)

    def q_z_natural(self, h_mean, h_m2):
        mu = self.s2_f ** .5 / self.s2 * mul(self.mats['sum_Ahx_y'], h_mean,
                                             adj_a=True)
        S = self.mats['sum_Bxx'] + sum(mul3(self.mats['Ahx'],
                                            tile(h_m2, self.n),
                                            self.mats['Ahx'], adj_a=True), 0)
        return mu, self.Kx + self.s2_f / self.s2 * S

    def elbo(self, smf=False):
        if smf:
            # Stochastic approximation via reparametrisation
            h_samp = self.h.sample()
            mu, S = self.q_z_natural(h_samp, outer(h_samp))
        else:
            mu, S = self.q_z_natural(self.h.mean, self.h.m2)
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

        return tf.Print(elbo, [smf, self.s2, self.s2_f, self.pars['gamma'],
                               self.pars['omega']])

    def predict_k(self,
                  t,
                  samples_h=None,
                  iters=200,
                  percentiles=True,
                  psd=False):
        n = shape(t)[0]
        Ahh = self.Ahh_center(t, self.causal)
        a = self.a_center(t, self.causal)

        # Precompute
        Ahh, a = self.run([Ahh, a])

        if samples_h is None:
            h = self.h.sample()
            samples_h = [self.run(h) for i in range(iters)]

        # Compute via MC
        h = ph(shape(self.h.sample()))
        k = (a + trmul(tile(outer(h) - self.iKh, n), Ahh))[:, None]
        samples_k = [self.run(k, feed_dict={h: sample_h})
                     for sample_h in samples_h]

        # Check whether to predict kernel or PSD
        if psd:
            samples = [utils.psd(x / max(x) / len(x), axis=0)
                       for x in samples_k]
        else:
            samples = samples_k

        samples = np.concatenate(samples, axis=1)
        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        if percentiles:
            lower = np.percentile(samples,
                                  100 * scipy.stats.norm.cdf(-2),
                                  axis=1)
            upper = np.percentile(samples,
                                  100 * scipy.stats.norm.cdf(2),
                                  axis=1)
        else:
            lower = mu - 2 * std
            upper = mu + 2 * std
        if psd:
            return mu, lower, upper, std
        else:
            s2_f = self.run(self.s2_f)
            return s2_f * mu, s2_f * lower, s2_f * upper, s2_f * std

    def predict_h(self, t, samples_h=None, iters=50, percentiles=True, offset=0):
        if samples_h is None:
            h = self.h.sample()
            samples_h = [self.run(h) for i in range(iters)]

        h = ph(shape(self.h.sample()))

        # Transform samples back into h space
        h_transformed = mul(self.Kh, h)
        samples_h = [self.run(h_transformed, feed_dict={h: sample_h})
                     for sample_h in samples_h]

        # Sign samples the right way, assuming that th is symmetric around zero
        i_zero = (self.nh - 1 + offset) / 2
        samples_h = [sample_h * np.sign(sample_h[i_zero])
                     for sample_h in samples_h]

        Kfu = self.kernel_h(t, self.th[:, None])
        mu = mul(Kfu, tf.cholesky_solve(self.Lh, h))
        samples = np.concatenate([self.run(mu, feed_dict={h: sample_h})
                                  for sample_h in samples_h], axis=1)

        mu = np.mean(samples, axis=1)
        std = np.std(samples, axis=1)
        if percentiles:
            lower = np.percentile(samples,
                                  100 * scipy.stats.norm.cdf(-2),
                                  axis=1)
            upper = np.percentile(samples,
                                  100 * scipy.stats.norm.cdf(2),
                                  axis=1)
        else:
            lower = mu - 2 * std
            upper = mu + 2 * std
        return mu, lower, upper, std

    def predict_f(self, t, samples_h=None, iters=50, smf=False):
        n = shape(t)[0]
        mats = self.construct_model_matrices(t,
                                             None,
                                             causal=self.causal,
                                             causal_id=self.causal_id)

        # Precompute
        mats = {k: self.run(mats[k]) for k in ['a',
                                               'Ahh',
                                               'Ahx',
                                               'Axx']}

        if samples_h is None:
            h = self.h.sample()
            samples_h = [self.run(h) for i in range(iters)]

        # Construct optimal q(z|u) or q(z)
        h = tf.placeholder(config.dtype, shape(self.h.sample()))
        if smf:
            mu, S = self.q_z_natural(h, outer(h))
        else:
            mu, S = self.q_z_natural(self.h.mean, self.h.m2)
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
        samples = [self.run([mu, var], feed_dict={h: sample_h})
                   for sample_h in samples_h]
        samples_mu, samples_var = zip(*samples)
        mu = np.mean(np.concatenate(samples_mu, axis=1), axis=1)
        var = np.mean(np.concatenate(samples_var, axis=1), axis=1)

        s2_f = self.run(self.s2_f)
        mu *= np.sqrt(s2_f)
        var *= s2_f
        return mu, mu - 2 * np.sqrt(var), mu + 2 * np.sqrt(var), var ** .5

    def sample(self, iters=200, burn=50, display=False):
        h = tf.placeholder(config.dtype, shape(self.h_prior.sample()))
        mu, S = self.q_z_natural(h, outer(h))
        L = tf.cholesky(S)

        prior_sample = self.h_prior.sample()
        log_lik = tf.squeeze(-.5 * log_det(L)
                             + .5 * sum(trisolve(L, mu) ** 2)
                             - .5 * self.s2_f / self.s2
                             * mul3(h, self.mats['sum_Bhh'], h, adj_a=True))

        ess = ESS(lambda x: self.run(log_lik, feed_dict={h: x}),
                  lambda: self.run(prior_sample))
        ess.move(self.run(self.h.mean))
        ess.sample(burn, display=display)
        return ess.sample(iters, display=display)
