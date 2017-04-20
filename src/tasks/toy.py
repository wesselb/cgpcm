from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data
import config


config.reg = 1e-6


class Experiment(Task):
    """
    Toy experiment.
    """

    def generate_config(self, args):
        options = Options('toy')
        options.add_option('causal-sample', 'generate a causal sample')
        options.add_option('causal-model', 'use the causal model')
        options.add_option('test', 'small test experiment')
        options.add_value_option('resample', int,
                                 'number of times to resample',
                                 default=0)
        options.parse(args)

        if options['test']:
            tau_f = .25 if options['causal-sample'] else .125
        else:
            tau_f = .1 if options['causal-sample'] else .05

        return TaskConfig(name='Toy Experiment',
                          seed=1005 if options['causal-sample'] else 1030,
                          fp=options.fp(groups=[['causal-sample']],
                                        ignore=['test']),

                          # Training options
                          iters_fpi_pre=0,
                          iters_pre=200 if options['test'] else 400,
                          iters=500 if options['test'] else 2000,
                          iters_post=50 if options['test'] else 200,
                          samps=200 if options['test'] else 500,
                          iters_fpi_post=500,

                          # Sample options
                          causal=options['causal-sample'],
                          causal_model=options['causal-model'],
                          resample=options['resample'],
                          n=150 if options['test'] else 400,
                          nx=60 if options['test'] else 150,
                          nh=41,
                          noise=0 if options['causal-sample'] else .5,
                          noise_init=1e-4 if options['causal-sample'] else 1e-2,

                          tau_w=0.25 if options['test'] else 0.1,
                          tau_f=tau_f,
                          data_scale=.5)

    def load(self, sess):
        # Load data
        f, k, h = data.load_akm(sess=sess,
                                n=self.config.n,
                                nh=self.config.nh,
                                tau_w=self.config.tau_w \
                                      * self.config.data_scale,
                                tau_f=self.config.tau_f \
                                      * self.config.data_scale,
                                causal=self.config.causal,
                                resample=self.config.resample)
        e = f.make_noisy(self.config.noise)
        self._set_data(h=h, k=k, f=f, e=e)

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 tau_w=self.config.tau_w,
                                 tau_f=self.config.tau_f \
                                       * self.config.data_scale,
                                 causal=self.config.causal_model,
                                 noise_init=self.config.noise_init)
        self._set_model(mod)
