from core.exp import Task, TaskConfig, TaskOptions
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    HRIR.
    """

    def generate_config(self, args):
        options = TaskOptions()
        options.add_option('causal-model', 'use the causal model')
        options.parse(args)

        return TaskConfig(name='HRIR',
                          seed=0,
                          fn=options.fn(prefix='hrir'),

                          # Training options
                          iters_pre=50,
                          iters=500,
                          iters_post=0,
                          samps=0,

                          # Model options
                          causal_model=options['causal-model'],
                          n=500,
                          nx=200,
                          nh=101,
                          noise_init=1e-4,
                          tau_w=1e-3,
                          tau_f=1e-4)

    def load(self, sess):
        # Load data
        f, k, h = data.load_hrir(n=self.config.n)
        e = f
        self._set_data(h=h, f=f, e=f, k=k)

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
