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
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_pre=200,
                          iters=5000,
                          iters_post=200,
                          samps=200,

                          # Model options
                          causal_model=options['causal-model'],
                          n=350,
                          nx=150,
                          nh=50,
                          noise_init=1e-3,
                          tau_w=2e-2,
                          tau_f=1e-3)

    def load(self, sess):
        # Load data
        e, f = data.load_timit_tobar2015(self.config.n)

        # Subsample function to prevent graph explosion during prediction
        f_sub = f[::3]

        # Do, however, store accurate emperical estimates of the kernel and PSD
        k_emp = f.autocorrelation()
        k_emp /= k_emp.max
        self.data['k_emp'] = k_emp
        self.data['psd_emp'] = k_emp.fft_db(split_freq=False)
        # x_psd, y_psd = sp.periodogram(f.y, fs=1 / f.dx)
        # self.data['psd_emp'] = data.Data(x_psd, 10 * np.log(y_psd / scale))


        # Store data
        self._set_data(f=f_sub, e=e,
                       k=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               1501)),
                       h=data.Data(np.linspace(-2 * self.config.tau_w,
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

