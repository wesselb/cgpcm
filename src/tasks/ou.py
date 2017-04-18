import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


import scipy.signal as sp


class Experiment(Task):
    """
    OU.
    """

    def generate_config(self, args):
        options = Options('ou')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='TIMIT',
                          seed=1,
                          fp=options.fp(),

                          # Training options
                          iters_pre=200,
                          iters=500,
                          iters_post=0,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=500,
                          nx=250,
                          nh=31,
                          noise_init=1e-3,
                          tau_w=0.8,
                          tau_f=0.3,

                          # Experiment options
                          n_predict=100,
                          noise=0)

    def load(self, sess):
        # Load data
        n_total = self.config.n + self.config.n_predict
        f, k = data.load_gp_exp(sess, n=n_total, k_len=.5)

        # Predict on
        f_train, self.data['f_pred'] = f.fragment(self.config.n)
        e = f_train.make_noisy(self.config.noise)

        # Store data
        self._set_data(f=f, e=e, k=k,
                       h=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)))

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 tx_range=(min(f.x), max(f.x)),
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 tau_w=self.config.tau_w,
                                 tau_f=self.config.tau_f,
                                 causal=self.config.causal_model,
                                 noise_init=self.config.noise_init)
        self._set_model(mod)

