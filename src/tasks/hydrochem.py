import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data
import config


config.reg = 1e-5


class Experiment(Task):
    """
    Hydrochemical Data.
    """

    def generate_config(self, args):
        options = Options('hydrochem')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='Hydrochemical Data',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_fpi_pre=0,
                          iters_pre=20,
                          iters=50,
                          iters_post=0,
                          iters_fpi_post=500,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=600,
                          nx=400,
                          nh=151,
                          noise_init=1e-1,
                          tau_w=5,
                          tau_f=0.1)

    def load(self, sess):
        # Load data: take last data set
        f = data.load_hydrochem()[3].subsample(self.config.n)[0]
        e = f

        # Store data
        t = data.Data(np.linspace(-2 * self.config.tau_w,
                                  2 * self.config.tau_w,
                                  501))
        self._set_data(f=f, e=e, k=t, h=t)

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
