import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data

import scipy.signal as sp


class Experiment(Task):
    """
    Hydrochemical Data.
    """

    def generate_config(self, args):
        options = Options('hydrochem')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='Hydrochemical Data',
                          seed=1,
                          fp=options.fp(),

                          # Training options
                          iters_fpi=50,
                          iters_pre=20,
                          iters=100,
                          iters_post=0,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=500,
                          nx=250,
                          nh=51,
                          noise_init=1e-2,
                          tau_w=0.2,
                          tau_f=0.05,

                          # Experiment options; data set is 1139 long
                          noise=0,
                          fragment_start=500,
                          fragment_length=100)

    def load(self, sess):
        # Load data: take last data set
        f = data.load_hydrochem()[2]

        # Predict on
        self.data['f_pred'], f_train = f.fragment(self.config.fragment_length,
                                                  self.config.fragment_start)
        # Further downsample
        e = f_train.subsample(self.config.n)[0].make_noisy(self.config.noise)

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
