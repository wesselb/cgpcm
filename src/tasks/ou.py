import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data
import config


config.reg = 1e-6


class Experiment(Task):
    """
    OU.
    """

    def generate_config(self, args):
        options = Options('ou')
        options.add_option('causal-model', 'use the causal model')
        options.add_value_option('seed', int, default=0)
        options.parse(args)

        return TaskConfig(name='OU',
                          seed=options['seed'],
                          fp=options.fp([['seed']]),

                          # Training options
                          iters_fpi=50,
                          iters_pre=100,
                          iters=500,
                          iters_post=100,
                          samps=500,

                          # Model options
                          causal_model=options['causal-model'],
                          n=600,
                          nx=200,
                          nh=51,
                          noise_init=1e-3,
                          tau_w=0.1,
                          tau_f=0.05,

                          # Experiment options
                          k_len=0.025,
                          noise=0,
                          fragment_start=200,
                          fragment_length=100)

    def load(self, sess):
        # Load data
        f, k = data.load_gp_exp(sess,
                                n=self.config.n,
                                k_len=self.config.k_len)
        e = f

        # Store data
        self._set_data(f=f, e=e, k=k,
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
