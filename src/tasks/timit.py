import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


import scipy.signal as sp


class Experiment(Task):
    """
    TIMIT.
    """

    def generate_config(self, args):
        options = Options('timit')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='TIMIT',
                          seed=10,
                          fp=options.fp(),

                          # Training options
                          iters_pre=50,
                          iters=50,
                          iters_post=50,
                          samps=200,

                          # Model options
                          causal_model=options['causal-model'],
                          n=350,
                          nx=150,
                          nh=150,
                          noise_init=1e-2,
                          tau_w=3e-2,
                          tau_f=.5e-3)

    def load(self, sess):
        # Load data
        e, f = data.load_timit_tobar2015(self.config.n)

        # Subsample function to prevent graph explosion during prediction
        f_sub = f[::3]

        # Do, however, store accurate emperical estimates of the kernel and PSD
        k_emp = f.autocorrelation(normalise=True)
        self.data['k_emp'] = k_emp
        self.data['psd_emp'] = k_emp.fft()

        # Store data
        self._set_data(f=f_sub, e=e,
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

