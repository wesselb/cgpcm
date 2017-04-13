import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    TIMIT.
    """

    def generate_config(self, args):
        options = Options('timit')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='TIMIT',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_pre=200,
                          iters=500,
                          iters_post=50,
                          samps=200,

                          # Model options
                          causal_model=options['causal-model'],
                          nx=150,
                          nh=150,
                          noise_init=1e-3,
                          tau_w=10e-3,
                          tau_f=1e-3)

    def load(self, sess):
        # Load data
        e, f = data.load_timit_tobar2015()
        self._set_data(f=f, e=f,
                       k=data.Data(np.linspace(-4 * self.config.tau_w,
                                               4 * self.config.tau_w,
                                               300)),
                       h=data.Data(np.linspace(0,
                                               2 * self.config.tau_w,
                                               300)))

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

