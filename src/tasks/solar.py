import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data
import config


class Experiment(Task):
    """
    Solar irradiance.
    """

    def generate_config(self, args):
        options = Options('solar')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='Solar irradiance',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_fpi_pre=0,
                          iters_pre=50,
                          iters=1500,
                          iters_post=500,
                          iters_fpi_post=500,
                          samps=1500,

                          # Model options
                          causal_model=options['causal-model'],
                          n=251,
                          nx=200,
                          nh=121,
                          noise_init=5e-3,
                          tau_w=50.,
                          tau_f=.5,

                          # Experiment options
                          n_pred=20)

    def load(self, sess):
        # Load data
        f, _ = data.load_solar()
        self.data['f_out'], e = f.fragment(self.config.n_pred,
                                           start=[20, 120, 220])

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
