import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


import scipy.signal as sp


class Experiment(Task):
    """
    Currency.
    """

    def generate_config(self, args):
        options = Options('currency')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='Currency',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_fpi=20,
                          iters_pre=50,
                          iters=50,
                          iters_post=0,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=500,
                          nx=150,
                          nh=51,
                          noise_init=1e-2,
                          tau_w=.5,
                          tau_f=.1)

    def load(self, sess):
        # Load data
        f = data.load_crude_oil()
        e = f.subsample(self.config.n)[0]

        # Store data
        self._set_data(f=f, e=f,
                       k=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)),
                       h=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)))

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=f,
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 tau_w=self.config.tau_w,
                                 tau_f=self.config.tau_f,
                                 causal=self.config.causal_model,
                                 noise_init=self.config.noise_init)
        self._set_model(mod)

