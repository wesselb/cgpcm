import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    CO2.
    """

    def generate_config(self, args):
        options = Options('co2')
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='Mauna Loa CO2',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_pre=50,
                          iters=100,
                          iters_post=100,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=709,  # All data
                          nx=300,
                          nh=300,
                          noise_init=1e-2,
                          tau_w=30,
                          tau_f=0.1,

                          # Experiment options
                          n_predict=100,
                          noise=0)

    def load(self, sess):
        # Load data
        f = data.load_co2()
        k = f.autocorrelation(normalise=True)

        # Predict on
        f_train, self.data['f_pred'] = f.fragment(self.config.n
                                                  - self.config.n_predict)
        e = f_train.make_noisy(self.config.noise)

        # Store data
        self._set_data(f=f, e=e, k=k,
                       h=data.Data(np.linspace(-2 * self.config.tau_w,
                                               2 * self.config.tau_w,
                                               501)))

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 tx_range=(min(f.x), max(f.x)),
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 tau_w=self.config.tau_w,
                                 tau_f=self.config.tau_f,
                                 causal=self.config.causal_model,
                                 noise_init=self.config.noise_init)
        self._set_model(mod)
