import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    HRIR.
    """

    def generate_config(self, args):
        options = Options('hrir')
        options.add_option('causal-model', 'use the causal model')
        options.add_value_option('resample', value_type=int, default=0,
                                 desc='number of times to resample')
        options.parse(args)

        return TaskConfig(name='HRIR',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_pre=50,
                          iters=100,
                          iters_post=100,
                          samps=500,

                          # Model options
                          causal_model=options['causal-model'],
                          n=300,
                          nx=150,
                          nh=151,
                          noise_init=1e-3,
                          tau_w=1e-3,
                          tau_f=.15e-3,
                          resample=options['resample'])

    def load(self, sess):
        # Load data
        f, k, h = data.load_hrir(n=self.config.n,
                                 resample=self.config.resample)
        e = f
        self._set_data(h=h, f=f, e=f, k=k)

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
