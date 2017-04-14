import abc

from parametrisable import Parametrisable
from plot import Plotter2D
from options import Options
from data import Data, UncertainData
from cgpcm import VCGPCM
from tf_util import *
import out
import learn


class Task(object):
    """
    Learning task.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, options):
        self.data = {}  # To save useful data in
        self.config = self.generate_config(options)

    @abc.abstractmethod
    def generate_config(self, options):
        """
        Generate config for task.

        :param options: list of strings
        :return: :class:`core.exp.TaskConfig` instance
        """
        pass

    @abc.abstractmethod
    def load(self, sess):
        """
        Load data and model.

        This method should call `self._set_data` and `self._set_model`.

        :param sess: TensorFlow session
        """
        pass

    def _set_data(self, h, k, f, e):
        self.data['h'] = h
        self.data['k'] = k
        self.data['f'] = f
        self.data['e'] = e

    def _set_model(self, mod):
        self.mod = mod

    def make_pickleable(self, sess):
        """
        Save useful stuff in the property `data` and make the object
        pickleable.
        """
        self.data.update({'tx': sess.run(self.mod.tx),
                          'th': sess.run(self.mod.th),
                          's2': sess.run(self.mod.s2),
                          's2_f': sess.run(self.mod.s2_f),
                          'alpha': sess.run(self.mod.alpha),
                          'gamma': sess.run(self.mod.gamma),
                          'omega': sess.run(self.mod.omega),
                          'causal': self.mod.causal,
                          'causal_id': self.mod.causal_id,
                          'mu': sess.run(self.mod.h.mean),
                          'var': sess.run(self.mod.h.var)})

        # Store variables
        self.data['vars'] = {}
        for var_name in self.mod.vars.keys():
            self.data['vars'][var_name] = sess.run(self.mod.vars[var_name])

        # Store recipe
        self.data['recipe'] = {}
        for attr in ['e', 'nx', 'nh', 'tau_w', 'tau_f', 'causal', 'causal_id',
                     'noise_init']:
            self.data['recipe'][attr] = getattr(self.mod, attr)

        del self.mod

    def report(self):
        """
        Generate a report of the task.
        
        :return: report in the form of a dictionary 
        """

        def from_data(key):
            return self.data[key] if key in self.data else 'missing'

        report = {'parameters': {'s2': from_data('s2'),
                                 's2_f': from_data('s2_f'),
                                 'alpha': from_data('alpha'),
                                 'gamma': from_data('gamma'),
                                 'ELBO MF': from_data('elbo_mf')},
                  'training': {'iters pre': self.config.iters_pre,
                               'iters': self.config.iters,
                               'iters post': self.config.iters_post,
                               'samples': self.config.samps,
                               'n': self.config.n,
                               'nx': self.config.nx,
                               'nh': self.config.nh,
                               'tau_w': self.config.tau_w,
                               'tau_f': self.config.tau_f}}
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
            return x[np.abs(x.x) <= self.config.tau_w]

        def filter_onesided(x):
            return x[np.logical_and(x.x >= 0, x.x <= self.config.tau_w)]

        self.data['h_mp'] = self.data['h'].minimum_phase()
        self.data['h_zp'] = self.data['h'].zero_phase()
        report.update({'prediction': {'function': report_key('f'),
                                      'kernel':
                                          report_key('k', filter_onesided),
                                      'filter (zero phase)':
                                          report_key('h_zp', filter_onesided),
                                      'filter (minimum phase)':
                                          report_key('h_mp', filter_onesided)
                                      }})

        return report


class TaskConfig(Parametrisable):
    """
    Configuration for a task.
    """
    _required_pars = ['fp', 'seed', 'iters_post', 'iters', 'iters_pre',
                      'samps', 'name']


def train(sess, task, debug_options):
    """
    Perform training for task.

    :param sess: TensorFlow session
    :param task: task
    :param debug_options: debug options
    :return: trained model
    """
    # Load task
    out.section('loading task')
    task.load(sess)
    mod = task.mod
    out.section_end()

    # Check whether to debug kernel
    if debug_options['kernel']:
        out.state('plotting kernel and filter and then exiting')
        p = Plotter2D()
        p.subplot(2, 2, 1)
        p.title('Data')
        p.plot(task.data['f'])
        p.subplot(2, 2, 2)
        p.title('Kernel')
        p.plot(task.data['k'])
        p.subplot(2, 2, 3)
        p.title('Filter')
        p.plot(task.data['h'])
        p.subplot(2, 2, 4)
        p.title('PSD')
        p.plot(task.data['k'].fft_db(split_freq=False))
        p.show()
        exit()

    # Precomputation
    out.section('precomputing')
    mod.precompute()
    out.section_end()

    # Train MF
    out.section('training MF')
    elbo = mod.elbo()
    fetches_config = [{'name': 'ELBO', 'tensor': elbo, 'modifier': '.2e'},
                      {'name': 's2', 'tensor': mod.s2, 'modifier': '.2e'},
                      {'name': 's2_f', 'tensor': mod.s2_f, 'modifier': '.2e'},
                      {'name': 'gamma', 'tensor': mod.gamma,
                       'modifier': '.2e'},
                      {'name': 'alpha', 'tensor': mod.alpha,
                       'modifier': '.2e'}]
    learn.minimise_lbfgs(sess, -elbo,
                         vars=[mod.vars['mu'], mod.vars['var']],
                         iters=task.config.iters_pre,
                         fetches_config=fetches_config,
                         name='pretraining using L-BFGS')
    learn.minimise_lbfgs(sess, -elbo,
                         vars=[mod.vars['mu'],
                               mod.vars['var'],
                               mod.vars['s2'],
                               mod.vars['s2_f']],
                         iters=task.config.iters,
                         fetches_config=fetches_config,
                         name='training using L-BFGS')
    # Check number of iterations to prevent unnecessary precomputation
    if task.config.iters_post > 0:
        mod.undo_precompute()
        elbo = mod.elbo()
        fetches_config[0]['tensor'] = elbo
        learn.minimise_lbfgs(sess, -mod.elbo(),
                             vars=[mod.vars['mu'],
                                   mod.vars['var'],
                                   mod.vars['s2'],
                                   mod.vars['s2_f'],
                                   mod.vars['gamma'],
                                   mod.vars['alpha']],
                             iters=task.config.iters_post,
                             fetches_config=fetches_config,
                             name='posttraining using L-BFGS')
        out.section('precomputing')
        mod.precompute()
        out.section_end()
    out.section_end()

    if task.config.samps > 0:
        # Train SMF
        out.section('training SMF')
        task.data['samples'] = mod.sample(iters=task.config.samps)
        out.section_end()

    return mod


def mod_from_task(sess, task, debug_options):
    """
    Construct the trained model from a task that was trained previously.
    
    :param sess: TensorFlow session
    :param task: task
    :param debug_options: debug options
    :return: trained model
    """

    # Construct model from recipe
    mod = VCGPCM.from_recipe(sess=sess, **task.data['recipe'])
    initialise_uninitialised_variables(sess)

    # Restore variables
    for var_name, var_value in task.data['vars'].items():
        sess.run(mod.vars[var_name].assign(var_value))

    # Precompute matrices
    out.state('precomputing')
    mod.precompute()

    return mod


def predict(sess, task, mod, debug_options):
    """
    Perform training for task.

    :param sess: TensorFlow session
    :param task: task
    :param mod: trained model
    :param debug_options: debug options
    """
    # Predict MF
    out.section('predicting MF')
    task.data['f_pred'] = mod.predict_f(task.data['f'].x)
    task.data['k_pred'] = mod.predict_k(task.data['k'].x)
    task.data['psd_pred'] = mod.predict_k(task.data['k'].x, psd=True)
    task.data['psd_pred_fs'] = 1. / task.data['k'].dx
    task.data['h_mp_pred'] = mod.predict_h(task.data['h'].x,
                                           phase_transform='minimum_phase')
    task.data['h_zp_pred'] = mod.predict_h(task.data['h'].x,
                                           phase_transform='zero_phase')
    out.section_end()

    # ELBO for MF
    task.data['elbo_mf'] = sess.run(mod.elbo())

    if task.config.samps > 0:
        # Predict SMF
        out.section('predicting SMF')
        samples = task.data['samples']
        task.data['f_pred_smf'] = mod.predict_f(task.data['f'].x,
                                                samples_h=samples)
        task.data['k_pred_smf'] = mod.predict_k(task.data['k'].x,
                                                samples_h=samples)
        task.data['psd_pred_smf'] = mod.predict_k(task.data['k'].x,
                                                  samples_h=samples,
                                                  psd=True)
        task.data['psd_pref_smf_fs'] = 1. / task.data['k'].dx
        task.data['h_mp_pred_smf'] = mod.predict_h(task.data['h'].x,
                                                   samples_h=samples,
                                                   phase_transform='minimum_'
                                                                   'phase')
        task.data['h_zp_pred_smf'] = mod.predict_h(task.data['h'].x,
                                                   samples_h=samples,
                                                   phase_transform='zero_'
                                                                   'phase')
        out.section_end()

        # ELBO for SMF
        task.data['elbo_smf'] = mod.elbo_smf(samples)


class TaskPlotter(object):
    """
    Performs common plotting exercises for a task.

    :param p: :class:`core.plot.Plotter2D` instance
    :param task: task
    """
    _styles = {'truth': {'colour': '#7b3294',
                         'line_style': '-.',
                         'marker_style': 'o'},
               'observation': {'colour': '#008837',
                               'line_style': ':',
                               'marker_style': '^'},
               'task1': {'colour': '#0571b0',
                         'line_style': '-',
                         'marker_style': '*'},
               'task2': {'colour': '#ca0020',
                         'line_style': '--',
                         'marker_style': '+'},
               'alt1': {'colour': 'cyan',
                        'line_style': '-',
                        'marker_style': 'o'},
               'alt2': {'colour': 'orange',
                        'line_style': '-',
                        'marker_style': 'o'},
               'alt3': {'colour': 'green',
                        'line_style': '-',
                        'marker_style': 'o'},
               'inducing_points': {'colour': 'k',
                                   'line_style': '-',
                                   'marker_style': 's'}}

    def __init__(self, p, task):
        self._p = p
        self._task = task
        self._x_min = None
        self._x_max = None
        p.config(line_width=1.5,
                 marker_size=2.5,
                 fill_alpha=.25)

    @property
    def model_name(self):
        """
        Get model name associated with task.
        """
        if 'causal' in self._task.data:
            cgpcm = self._task.data['causal']
        else:
            # Then assume config.causal_model exists
            cgpcm = self._task.config.causal_model
        return 'CGPCM' if cgpcm else 'GPCM'

    def bound(self, x_min=None, x_max=None, key=None):
        """
        Bound the data plotted.
        
        :param x_min: minimum x value
        :param x_max: maximum x value
        :param key: bound any plotted data to the domain of the data associated
                    with this key
        """
        self._x_min, self._x_max = None, None
        if key:
            self._x_min = min(self._task[key].x)
            self._x_max = max(self._task[key].x)
        if x_min is not None:
            self._x_min = x_min
        if x_max is not None:
            self._x_max = x_max

    def _process_fun(self, x_unit=1):
        def process(d):
            d = Data(d.x * x_unit, d.y)
            if self._x_min is not None:
                d = d[d.x >= self._x_min]
            if self._x_max is not None:
                d = d[d.x <= self._x_max]
            return d

        return process

    def fill(self, key, style, label=None, x_unit=1):
        """
        Plot a fill.
        
        :param key: key of data
        :param style: style
        :param label: label
        :param x_unit: unit of x axis
        """
        mean, lower, upper, std = map(self._process_fun(x_unit),
                                      self._task.data[key])
        self._p.fill(lower.x, lower.y, upper.y,
                     fill_colour=self._styles[style]['colour'])
        self._p.plot(mean.x, mean.y,
                     line_colour=self._styles[style]['colour'],
                     line_style=self._styles[style]['line_style'],
                     label=label)

    def line(self, key, style, label=None, x_unit=1):
        """
        Plot a line
        
        :param key: key of data
        :param style: style
        :param label: label
        :param x_unit: unit of x axis
        """
        d = self._process_fun(x_unit)(self._task.data[key])
        self._p.plot(d.x, d.y,
                     line_colour=self._styles[style]['colour'],
                     line_style=self._styles[style]['line_style'],
                     label=label)

    def marker(self, key, style, label=None, x_unit=1):
        """
        Plot markers.
        
        :param key: key of data
        :param style: style
        :param label: label
        :param x_unit: unit of x axis
        """
        d = self._process_fun(x_unit)(self._task.data[key])
        self._p.plot(d.x, d.y,
                     line_style='none',
                     marker_style=self._styles[style]['marker_style'],
                     marker_colour=self._styles[style]['colour'],
                     label=label)


def plot_compare(tasks, args):
    """
    Compare tasks.
    
    :param tasks: tasks
    :param args: arguments
    :return: `core.plot.Plotter2D` instance and file path
    """
    options = Options('compare')
    options.add_value_option('index0', value_type=int, default=0,
                             desc='index of first plot in comparison')
    options.add_value_option('index1', value_type=int, default=None,
                             desc='index of second plot in comparison')
    options.add_option('big', 'show big plot')
    options.add_option('ms', 'rescale to milliseconds')
    options.add_option('no-psd', 'no PSD')
    options.add_option('mf0', 'plot instead MF prediction for first task')
    options.add_option('mf1', 'plot instead MF prediction for second task')
    options.add_option('zp', 'plot zero-phase prediction of filter')
    options.add_value_option('tau-ws', desc='number of tau_w\'s to plot',
                             value_type=float, default=2)
    options.parse(args)

    p = Plotter2D(figure_size=(20, 10) if options['big'] else (12, 6),
                  font_size=12,
                  figure_toolbar='toolbar2' if options['big'] else 'none',
                  grid_colour='none')
    p.figure()

    tau_ws = options['tau-ws']

    # Process first task
    task1 = tasks[options['index0']]
    data1, pt1 = task1.data, TaskPlotter(p, task1)

    # Possibly process second task
    if options['index1'] is not None:
        task2 = tasks[options['index1']]
        data2, pt2 = task2.data, TaskPlotter(p, task2)
    else:
        task2 = None

    if options['mf0']:
        add1 = ''
        name1 = pt1.model_name + ' (MF)'
    else:
        add1 = '_smf'
        name1 = pt1.model_name + ' (SMF)'

    if task2:
        if options['mf1']:
            add2 = ''
            name2 = pt2.model_name + ' (MF)'
        else:
            add2 = '_smf'
            name2 = pt2.model_name + ' (SMF)'

    # Set units for time plots
    if options['ms']:
        p.x_scale = 1e3
        x_unit = 'ms'
    else:
        p.x_scale = 1
        x_unit = 's'

    data1['tx_data'] = Data(data1['tx'], 0 * data1['tx'])
    data1['th_data'] = Data(data1['th'], 0 * data1['th'])
    if task2:
        data2['tx_data'] = Data(data2['tx'], 0 * data2['tx'])
        data2['th_data'] = Data(data2['th'], 0 * data2['th'])

    # Function
    p.subplot2grid((2, 6), (0, 0), colspan=6)
    p.x_shift = -data1['f'].x[0]
    pt1.marker('tx_data', 'inducing_points')
    pt1.marker('f', 'truth', 'Truth')
    if not data1['f'].equals_approx(data1['e']):
        pt1.marker('e', 'observation', 'Observations')

    pt1.fill('f_pred' + add1, 'task1', name1)
    if task2:
        pt2.fill('f_pred' + add2, 'task2', name2)
    p.lims(x=data1['f'].domain)
    p.show_legend()
    p.labels(y='$f\,|\,h$', x='$t$ ({})'.format(x_unit))
    p.x_shift = 0

    pt1.bound(x_min=-tau_ws * task1.config.tau_w, x_max=tau_ws * task1.config.tau_w)
    if task2:
        pt2.bound(x_min=-tau_ws * task1.config.tau_w, x_max=tau_ws * task1.config.tau_w)

    # Kernel
    if options['no-psd']:
        p.subplot2grid((2, 6), (1, 0), colspan=3)
    else:
        p.subplot2grid((2, 6), (1, 0), colspan=2)
    p.lims(x=(-tau_ws * task1.config.tau_w, tau_ws * task1.config.tau_w))
    pt1.marker('th_data', 'inducing_points')
    if task2:
        pt2.marker('th_data', 'inducing_points')
    pt1.line('k', 'truth', 'Truth')
    if 'k_emp' not in data1:
        data1['k_emp'] = data1['f'].autocorrelation()
    data1['k_emp'] /= data1['k_emp'].max
    pt1.line('k_emp', 'observation', 'Autocorrelation')
    pt1.fill('k_pred' + add1, 'task1', name1)
    if task2:
        pt2.fill('k_pred' + add2, 'task2', name2)
    p.labels(y='$k_{f\,|\,h}$', x='$t$ ({})'.format(x_unit))
    p.show_legend()

    # Filter
    if options['no-psd']:
        p.subplot2grid((2, 6), (1, 3), colspan=3)
    else:
        p.subplot2grid((2, 6), (1, 2), colspan=2)
    p.lims(x=(-tau_ws * task1.config.tau_w, tau_ws * task1.config.tau_w))
    pt1.marker('th_data', 'inducing_points')
    data1['h_mp'] = data1['h'].minimum_phase()
    data1['h_zp'] = data1['h'].zero_phase()
    pt1.line('h_{}'.format('zp' if options['zp'] else 'mp'), 'truth', 'Truth')
    pt1.fill('h_{}_pred{}'.format('zp' if options['zp'] else 'mp', add1),
             'task1', name1)
    if task2:
        pt2.fill('h_{}_pred{}'.format('zp' if options['zp'] else 'mp', add2),
                 'task2', name2)
    p.labels(y='$h$', x='$t$ ({})'.format(x_unit))
    p.show_legend()

    if not options['no-psd']:
        def extract_fs(data, psd_key, k_key):
            # This function is needed due to compatibility issues
            if psd_key + '_fs' in data:
                return data[psd_key + '_fs']
            else:
                # Then assume that PSD prediction is associated to kernel
                # prediction
                return 1. / data[k_key][0].dx

        # Extract sampling frequencies
        fs1 = extract_fs(data1, 'psd_pred' + add1, 'k_pred' + add1)
        if task2:
            fs2 = extract_fs(data2, 'psd_pred' + add2, 'k_pred' + add2)

        pt1.bound(x_min=0, x_max=1.5 / task1.config.tau_f)
        if task2:
            pt2.bound(x_min=0, x_max=1.5 / task1.config.tau_f)

        # Compute PSD of truth
        data1['psd'], data1['psd_fs'] = data1['k'].fft_db()

        # Set units to frequency plots
        if options['ms']:
            p.x_scale = 1e-3
            x_unit = 'kHz'
        else:
            p.x_scale = 1
            x_unit = 'Hz'

        # PSD
        p.subplot2grid((2, 6), (1, 4), colspan=2)
        p.lims(x=(0, 1 / task1.config.tau_f))
        pt1.line('psd', 'truth', x_unit=data1['psd_fs'], label='Truth')
        if 'psd_emp' not in data1:
            data1['psd_emp'] = data1['k_emp'].fft_db(split_freq=False)
        pt1.line('psd_emp', 'observation', label='Periodogram')

        ac1 = data1['f_pred'][0].autocorrelation()
        data1['psd2'] = (ac1 / ac1.max).fft_db(split_freq=False)
        pt1.line('psd2', 'alt1', label='From f pred')

        ac2 = data1['h_mp_pred'][0].autocorrelation()
        data1['psd3'] = (ac2 / ac2.max).fft_db(split_freq=False)
        pt1.line('psd3', 'alt2', label='From h pred')

        ac3 = data1['k_pred'][0]
        data1['psd4'] = (ac3 / ac3.max).fft_db(split_freq=False)
        pt1.line('psd4', 'alt3', label='From k pred')

        pt1.fill('psd_pred' + add1, 'task1', x_unit=fs1, label=name1)
        if task2:
            pt2.fill('psd_pred' + add2, 'task2', x_unit=fs2, label=name2)
        p.labels(y='PSD of $f\,|\,h$ (dB)',
                 x='Frequency ({})'.format(x_unit))
        # p.ax.set_ylim(bottom=-30)
        p.show_legend()

    # Return `Plotter2D` instance and file path
    fp = options.fp(
        ignore=['index0', 'index1', 'ms', 'no-psd', 'mf0', 'mf1',
                'zp', 'big', 'tau-ws'])
    if task2:
        fp += task1.config.fp & task2.config.fp
    else:
        fp += task1.config.fp
    return p, fp


lot_choices = ['compare']
plot_calls = {'compare': plot_compare}


def plot(choice, tasks, args):
    """
    Plot a task.

    Choice must be in `plot_choices`.

    :param choice: choice
    :param tasks: tasks
    :param args: plot arguments
    :return: :class:`core.plot.Plotter2D` instance and file path
    """
    return plot_calls[choice](tasks, args)
