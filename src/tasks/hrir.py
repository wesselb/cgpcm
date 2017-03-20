import numpy as np

from core.experiment import Task, TaskConfig, Options
from core.cgpcm import VCGPCM
from core.experiment import correct_filter
import core.data as data


class Experiment(Task):
    """
    HRIR.
    """

    def generate_config(self, args):
        options = Options('hrir')
        options.add_option('causal-model', 'use the causal model')
        options.add_value_option('resample', value_type=int, default=0,
                                 desc='number of times to resample')
        options.parse(args)

        return TaskConfig(name='HRIR',
                          seed=0,
                          fp=options.fp(),

                          # Training options
                          iters_pre=200,
                          iters=2000,
                          iters_post=0,
                          samps=1000,

                          # Model options
                          causal_model=options['causal-model'],
                          n=500,
                          nx=200,
                          nh=121,
                          noise_init=2e-3,
                          tau_w=1e-3,
                          tau_f=.2e-3,
                          resample=options['resample'])

    def load(self, sess):
        # Load data
        f, k, h = data.load_hrir(n=self.config.n,
                                 resample=self.config.resample)
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

    def report(self):
        report = {'parameters': {'s2': self.data['s2'],
                                 's2_f': self.data['s2_f'],
                                 'alpha': self.data['alpha'],
                                 'gamma': self.data['gamma'],
                                 'ELBO MF': self.data['elbo_mf']}}
        if 'elbo_smf' in self.data:
            if len(self.data['elbo_smf']) == 3:
                report['parameters']['ELBO SMF'] = self.data['elbo_smf'][0]
            else:
                report['parameters']['ELBO SMF'] = \
                    {'estimate': self.data['elbo_smf'][0],
                     'estimated std': self.data['elbo_smf'][1]}

        def report_key(key_base, filter=lambda x: x):
            key = key_base + '_pred'
            ref = filter(self.data[key_base])
            report = {'SMSE': {'MF': ref.smse(filter(self.data[key][0]))},
                      'MLL': {'MF': ref.mll(filter(self.data[key][0]),
                                            filter(self.data[key][3]))}}
            if key + '_smf' in self.data:
                report['SMSE']['SMF'] = \
                    ref.smse(filter(self.data[key + '_smf'][0]))
                report['MLL']['SMF'] = \
                    ref.mll(filter(self.data[key + '_smf'][0]),
                            filter(self.data[key + '_smf'][3]))
            return report

        def filter_twosided(x):
            return x[np.abs(x.x) <= 2 * self.config.tau_w]

        def filter_onesided(x):
            return x[np.logical_and(x.x >= 0, x.x <= 2 * self.config.tau_w)]

        # Correct filters
        correct_filter(self, 'h_pred', 'h')
        if 'h_pred_smf' in self.data:
            correct_filter(self, 'h_pred_smf', 'h')

        report.update({'prediction': {'function': report_key('f'),
                                      'kernel': report_key('k',
                                                           filter_twosided),
                                      'filter': report_key('h',
                                                           filter_onesided)}})

        return report
