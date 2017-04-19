import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


import scipy.signal as sp


class Experiment(Task):
    """
    EEG.
    """

    def generate_config(self, args):
        options = Options('eeg')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='EEG',
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
                          n=257,  # Full length of data set
                          nx=150,
                          nh=150,
                          noise_init=1e-2,
                          tau_w=100,
                          tau_f=5)

    def load(self, sess):
        # Load data
        f = data.load_eeg()['F3']
        e = f

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

