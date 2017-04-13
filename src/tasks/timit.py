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
                          iters_pre=50,
                          iters=50,
                          iters_post=0,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          nx=180,
                          nh=180,
                          noise_init=1e-3,
                          tau_w=3e-2,
                          tau_f=.3e-3)

    def load(self, sess):
        # Load data
        e, f = data.load_timit_tobar2015()
        self._set_data(f=f, e=e,
                       k=data.Data(np.linspace(-4 * self.config.tau_w,
                                               4 * self.config.tau_w,
                                               1000)),
                       h=data.Data(np.linspace(0,
                                               2 * self.config.tau_w,
                                               1000)))

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

