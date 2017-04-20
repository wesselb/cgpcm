import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    TIMIT 2.
    """

    def generate_config(self, args):
        options = Options('timit2')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='TIMIT 2',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_fpi_pre=0,
                          iters_pre=100,
                          iters=100,
                          iters_post=0,
                          iters_fpi_post=500,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=800,
                          nx=400,
                          nh=150,
                          noise_init=1e-3,
                          tau_w=1.5e-2,
                          tau_f=.05e-3)

    def load(self, sess):
        # Load data
        f = data.load_timit_voiced_fricative()
        e = f.subsample(self.config.n)[0]

        # Store data
        self._set_data(f=f, e=e,
                       k=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)),
                       h=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)))

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 tau_w=self.config.tau_w,
                                 tau_f=self.config.tau_f,
                                 causal=self.config.causal_model,
                                 noise_init=self.config.noise_init)
        self._set_model(mod)

