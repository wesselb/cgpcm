from core.exp import Task, TaskConfig, TaskOptions
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    Toy experiment.
    """

    def generate_config(self, args):
        options = TaskOptions()
        options.add_option('causal', 'generate a causal sample')
        options.add_option('causal-model', 'use the causal model')
        options.add_option('post', 'use posttraining')
        options.add_option('no-pre', 'no pretraining')
        options.add_option('noisy', 'noisy sample')
        options.add_option('small', 'small sample')
        options.add_option('extensive', 'extensive training')
        options.add_value_option('resample', int,
                                 'number of times to resamples',
                                 required=True)
        options.add_value_option('seed', int, 'seed', required=True)
        options.parse(args)

        return TaskConfig(name='Toy Experiment',
                          seed=options['seed'],
                          fn=options.fn(group_by=['causal', 'seed', 'small']),

                          # Training options
                          iters_pre=0 if options['no-pre'] else 250,
                          iters=10000 if options['extensive'] else 2000,
                          iters_post=50 if options['post'] else 0,
                          samps=400,

                          # Sample options
                          causal=options['causal'],
                          causal_model=options['causal-model'],
                          resample=options['resample'],
                          n=150 if options['small'] else 400,
                          nx=50 if options['small'] else 120,
                          nh=41,
                          noise=.5 if options['noisy'] else 1e-4,
                          noise_init=1e-4 if options['causal'] else 1e-2,

                          tau_w=0.12 if options['causal'] else 0.04,
                          tau_f=0.12 if options['causal'] else 0.04,
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
