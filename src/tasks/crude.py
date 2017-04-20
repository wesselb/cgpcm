import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data
import config


config.reg = 5e-4


class Experiment(Task):
    """
    Crude oil.
    """

    def generate_config(self, args):
        options = Options('crude')
        options.add_option('causal-model', 'use the causal model')
        options.add_value_option('offset', int, required=True)
        options.add_value_option('length', int, required=True)
        options.add_value_option('n', int, required=True)
        options.parse(args)

        return TaskConfig(name='Crude oil',
                          seed=0,
                          fp=options.fp([['offset', 'length', 'n']]),

                          # Training options
                          iters_fpi_pre=0,
                          iters_pre=100,
                          iters=500,
                          iters_post=200,
                          iters_fpi_post=500,
                          samps=500,

                          # Model options
                          causal_model=options['causal-model'],
                          n=options['n'],
                          nx=300,
                          nh=51,
                          noise_init=5e-3,
                          tau_w=1.,
                          tau_f=.1,
                          
                          # Experiment options
                          offset=options['offset'],
                          length=options['length'])

    def load(self, sess):
        # Load data
        f = data.load_crude_oil()
        self.data['starts'] = self.config.offset + np.arange(500, 900, 80)
        self.data['f_pred'], f_train = f.fragment(self.config.length,
                                                  self.data['starts'])
        e = f_train

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

