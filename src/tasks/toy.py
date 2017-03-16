from core.exp import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    Toy experiment.
    """

    def generate_config(self, args):
        options = Options('toy')
        options.add_option('causal-sample', 'generate a causal sample')
        options.add_option('causal-model', 'use the causal model')
        options.add_value_option('resample', int,
                                 'number of times to resamples',
                                 required=True)
        options.parse(args)

        return TaskConfig(name='Toy Experiment',
                          seed=1025 if options['causal-sample'] else 1030,
                          fp=options.fp(groups=[['causal-sample']]),

                          # Training options
                          iters_pre=400,
                          iters=2000,
                          iters_post=0,
                          samps=400,

                          # Sample options
                          causal=options['causal-sample'],
                          causal_model=options['causal-model'],
                          resample=options['resample'],
                          n=400,
                          nx=120,
                          nh=41,
                          noise=1e-4 if options['causal-sample'] else .5,
                          noise_init=1e-4 if options['causal-sample'] else 1e-2,

                          tau_w=0.12 if options['causal-sample'] else 0.04,
                          tau_f=0.12 if options['causal-sample'] else 0.04,
                          data_scale=.75)

    def load(self, sess):
        # Load data
        f, k, h, psd = data.load_akm(sess=sess,
                                     n=self.config.n,
                                     nh=self.config.nh,
                                     tau_w=self.config.tau_w \
                                           * self.config.data_scale,
                                     tau_f=self.config.tau_f \
                                           * self.config.data_scale,
                                     causal=self.config.causal,
                                     resample=self.config.resample)
        e = f.make_noisy(self.config.noise)
        self._set_data(h=h, k=k, f=f, e=e, psd=psd)

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
