from core.learn import Task, TaskConfig
from core.cgpcm import VCGPCM
import core.data as data


class Experiment(Task):
    def generate_config(self, options):
        causal = 'causal' in options
        causal_model = 'causal-model' in options
        small = 'small' in options
        fn = 'toy_{}_sample_{}'.format('causal' if causal else 'acausal',
                                       'cgpcm' if causal_model else 'gpcm')
        return TaskConfig(name='Toy Experiment',
                          seed=51,
                          fn=fn,
                          iters_pre=400,
                          iters=2000,
                          samps=500,

                          causal=causal,
                          causal_model=causal_model,
                          resample=0,
                          n=150 if small else 400,
                          nx=50 if small else 100,
                          nh=31,
                          k_len=0.1,
                          k_wiggles=1.5,
                          data_scale=.5,
                          noise=1e-4)

    def load(self, sess):
        # Load data
        f, k, h = data.load_akm(sess=sess,
                                n=self.config.n,
                                nh=self.config.nh,
                                k_len=self.config.data_scale
                                      * self.config.k_len,
                                k_wiggles=self.config.k_wiggles,
                                causal=self.config.causal,
                                resample=self.config.resample)
        e = f.make_noisy(self.config.noise)
        self._set_data(h=h, k=k, f=f, e=e)

        # Construct model
        mod = VCGPCM.from_recipe(sess=sess,
                                 e=e,
                                 nx=self.config.nx,
                                 nh=self.config.nh,
                                 k_len=self.config.k_len,
                                 k_wiggles=self.config.k_wiggles
                                           / self.config.data_scale,
                                 causal=self.config.causal_model)
        self._set_model(mod)
