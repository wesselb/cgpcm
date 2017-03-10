from core.exp import Task, TaskConfig
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    """
    Toy experiment.

    The first option is

    Options:

        - resample: number of times to resample, this option is required and
                    must be first,
        - causal: generate a causal sample,
        - causal-model: use the GPCM instead of the CGPCM, and
        - post: use posttraining.
    """

    def generate_config(self, options):
        # Parse options
        if len(options) == 0:
            raise RuntimeError('first option is required')
        resample = int(options[0])
        causal = 'causal' in options
        causal_mod = 'causal-model' in options
        post = 'post' in options

        fn = 'toy_{}_{}_sample_{}{}'.format(resample,
                                            'causal' if causal else 'acausal',
                                            'cgpcm' if causal_mod else 'gpcm',
                                            '_post' if post else '')

        return TaskConfig(name='Toy Experiment',
                          seed=85,
                          fn=fn,

                          # Training options
                          iters_pre=200,
                          iters=500,
                          iters_post=50 if post else 0,
                          samps=300,

                          # Sample options
                          causal=causal,
                          causal_model=causal_mod,
                          resample=resample,
                          n=500,
                          nx=120,
                          nh=31,
                          k_len=.15,
                          k_wiggles=2,
                          noise=1e-4,

                          # Learner options
                          data_scale=.3,
                          learner_scale=.6)

    def load(self, sess):
        # Load data
        f, k, h, psd = data.load_akm(sess=sess,
                                     n=self.config.n,
                                     nh=self.config.nh,
                                     k_len=self.config.data_scale
                                           * self.config.k_len,
                                     k_wiggles=self.config.k_wiggles,
                                     causal=self.config.causal,
                                     resample=self.config.resample)
        e = f.make_noisy(self.config.noise)
        self._set_data(h=h, k=k, f=f, e=e, psd=psd)

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 k_len=self.config.k_len,
                                 k_wiggles=self.config.k_wiggles
                                           / self.config.learner_scale,
                                 causal=self.config.causal_model)
        self._set_model(mod)
