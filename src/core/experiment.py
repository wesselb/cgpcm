import abc
import numpy as np

from parametrisable import Parametrisable
from plot import Plotter2D
from options import Options
from data import Data
import out
import learn
import util


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

    def _set_data(self, h=None, k=None, f=None, e=None):
        self.data['h'] = h
        self.data['k'] = k
        self.data['f'] = f
        self.data['e'] = e

    def _set_model(self, mod):
        self.mod = mod

    def make_pickleable(self, sess):
        """
        Save useful stuff in the property `data` and make the object pickleable.
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
        del self.mod

    def report(self):
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
            return x[np.abs(x.x) <= 2 * self.config.tau_w]

        def filter_onesided(x):
            return x[np.logical_and(x.x >= 0, x.x <= 2 * self.config.tau_w)]

        # Correct reference filter
        self.data['h'] = self.data['h'].minimum_phase()

        report.update({'prediction': {'function': report_key('f'),
                                      'kernel': report_key('k',
                                                           filter_twosided),
                                      'filter': report_key('h',
                                                           filter_onesided)}})

        return report


class TaskConfig(Parametrisable):
    """
    Configuration for a task.
    """
    _required_pars = ['fp', 'seed', 'iters_post', 'iters', 'iters_pre',
                      'samps', 'name']


def train(sess, task, debug_options):
    """
    Train task.

    :param sess: TensorFlow session
    :param task: task
    :param debug_options: debug options
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
        p.figure('Data')
        p.plot(task.data['f'].x, task.data['f'].y)
        p.figure('Kernel')
        p.plot(task.data['k'].x, task.data['k'].y)
        p.figure('Filter')
        p.plot(task.data['h'].x, task.data['h'].y)
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
    if task.config.iters_post > 0:
        mod.undo_precompute()
        elbo = mod.elbo()
        fetches_config[0]['tensor'] = elbo
        learn.minimise_lbfgs(sess, -mod.elbo(),
                             vars=[mod.vars['mu'],
                                   mod.vars['var'],
                                   mod.vars['s2'],
                                   mod.vars['s2_f'],
                                   mod.vars['gamma']],
                             iters=task.config.iters_post,
                             fetches_config=fetches_config,
                             name='posttraining using L-BFGS')
        out.section('precomputing')
        mod.precompute()
        out.section_end()
    out.section_end()

    # Predict MF
    out.section('predicting MF')
    task.data['f_pred'] = mod.predict_f(task.data['f'].x)
    task.data['k_pred'] = mod.predict_k(task.data['k'].x)
    task.data['psd_pred'] = mod.predict_k(task.data['k'].x, psd=True)
    task.data['psd_pred_fs'] = 1. / task.data['k'].dx
    task.data['h_pred'] = mod.predict_h(task.data['h'].x)
    out.section_end()

    # ELBO for MF
    task.data['elbo_mf'] = sess.run(mod.elbo())

    if task.config.samps > 0:
        # Train SMF
        out.section('training SMF')
        task.data['samples'] = mod.sample(iters=task.config.samps)
        out.section_end()

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
        task.data['h_pred_smf'] = mod.predict_h(task.data['h'].x,
                                                samples_h=samples,
                                                correct_signs=True)
        out.section_end()

        # ELBO for SMF
        task.data['elbo_smf'] = mod.elbo_smf(samples)


def correct_filter(task, key, out_key, reference_key, soft=False):
    """
    Correct the filter in a task.

    :param task: task
    :param key: key of predicted filter
    :param out_key: key of corrected predicted filter
    :param reference_key: key of true filter
    :param soft: soft correction: only adjust sign
    """
    mean, lower, upper, std = task.data[key]
    x_orig = mean.x
    ref = task.data[reference_key].positive_part()

    # Correct sign
    sign = util.sign_reference(mean, ref)
    mean *= sign
    lower, upper = (lower, upper) if sign > 0 else (-upper, -lower)

    # Shift
    if not soft:
        delta = util.optimal_shift(mean, ref, bounds=(-10 * ref.dx,
                                                      10 * ref.dx))
    else:
        delta = 0

    # Save
    task.data[out_key] = mean.shift(delta).at(x_orig), \
                         lower.shift(delta).at(x_orig), \
                         upper.shift(delta).at(x_orig), \
                         std.shift(delta).at(x_orig)


class TaskPlotter(object):
    """
    Performs common plotting exercises for a task.

    :param p: :class:`core.plot.Plotter2D` instance
    :param task: task
    """
    _colours = {'truth': '#7b3294',
                'observation': '#008837',
                'task1': '#0571b0',
                'task2': '#ca0020'}

    def __init__(self, p, task):
        self._p = p
        self._task = task
        self._x_min = None
        self._x_max = None
        p.config(line_width=1,
                 marker_size=2,
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

    def colour(self, colour):
        if colour in self._colours:
            return self._colours[colour]
        else:
            return colour

    def bound(self, x_min=None, x_max=None, key=None):
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

    def fill(self, key, colour, label=None, x_unit=1):
        mean, lower, upper, std = map(self._process_fun(x_unit),
                                      self._task.data[key])
        self._p.fill(lower.x, lower.y, upper.y,
                     fill_colour=self.colour(colour))
        self._p.plot(mean.x, mean.y,
                     line_colour=self.colour(colour),
                     label=label)

    def line(self, key, colour, label=None, x_unit=1):
        d = self._process_fun(x_unit)(self._task.data[key])
        self._p.plot(d.x, d.y,
                     line_colour=self.colour(colour),
                     label=label)

    def marker(self, key, colour, label=None, x_unit=1):
        d = self._process_fun(x_unit)(self._task.data[key])
        self._p.plot(d.x, d.y,
                     line_style='none',
                     label=label,
                     marker_style='o',
                     marker_colour=self.colour(colour))


def plot_full(tasks, args):
    """
    Fully plot a single task.

    :param tasks: tasks
    :param args: arguments
    :return: :class:`core.plot.Plotter2D` instance
    """
    options = Options('full')
    options.add_value_option('index', value_type=int, default=0,
                             desc='index of task to plot')
    options.add_option('flip', 'flip the filter for the SMF prediction')
    options.parse(args)

    task = tasks[options['index']]

    def do_plot(p, x, x_pred, x_noisy=None, inducing_points=None):
        mu, lower, upper, std = x_pred
        p.plot(x.x, x.y, label='Truth', line_colour='r')
        if x_noisy is not None:
            p.plot(x_noisy.x, x_noisy.y,
                   label='Observed',
                   line_style='none',
                   marker_style='o',
                   marker_colour='g',
                   marker_size=3)
        p.fill(lower.x, lower.y, upper.y, fill_colour='b')
        p.plot(mu.x, mu.y, label='Learned', line_colour='b')
        if inducing_points is not None:
            p.plot(inducing_points, inducing_points * 0,
                   line_style='none',
                   marker_style='o',
                   marker_colour='k',
                   marker_size=3)
        p.show_legend()

    p = Plotter2D(figure_size=(20, 7))
    p.figure()

    p.subplot(2, 3, 1)
    p.title('Filter')
    sign = util.sign_reference(task.data['h_pred'][0], task.data['h'])
    delta = util.optimal_shift(task.data['h'],
                               sign * task.data['h_pred'][0].positive_part(),
                               bounds=(-10 * task.data['h'].dx,
                                       10 * task.data['h'].dx))
    do_plot(p, task.data['h'].shift(delta),
            map(lambda x: sign * x, task.data['h_pred']),
            inducing_points=task.data['th'])

    p.subplot(2, 3, 2)
    p.title('Kernel')
    k_emp = task.data['e'].autocorrelation()
    k_emp /= k_emp.max
    p.plot(k_emp.x, k_emp.y, line_colour='green', label='Autocorrelation')
    do_plot(p, task.data['k'], task.data['k_pred'],
            inducing_points=task.data['th'])

    p.subplot(2, 3, 3)
    p.title('Function')
    do_plot(p, task.data['f'], task.data['f_pred'], x_noisy=task.data['e'],
            inducing_points=task.data['tx'])

    if 'h_pred_smf' in task.data:
        p.subplot(2, 3, 4)
        p.title('Filter')
        sign = util.sign_reference(task.data['h_pred_smf'][0], task.data['h'])
        sign = -sign if options['flip'] else sign
        delta = util.optimal_shift(task.data['h'],
                                   sign * task.data['h_pred_smf'][
                                       0].positive_part(),
                                   bounds=(-10 * task.data['h'].dx,
                                           10 * task.data['h'].dx))
        do_plot(p, task.data['h'].shift(delta),
                map(lambda x: sign * x, task.data['h_pred_smf']),
                inducing_points=task.data['th'])

        p.subplot(2, 3, 5)
        p.title('Kernel')
        p.plot(k_emp.x, k_emp.y, line_colour='green', label='Autocorrelation')
        do_plot(p, task.data['k'], task.data['k_pred_smf'],
                inducing_points=task.data['th'])

        p.subplot(2, 3, 6)
        p.title('Function')
        do_plot(p, task.data['f'], task.data['f_pred_smf'],
                x_noisy=task.data['e'], inducing_points=task.data['tx'])

    return p, options.fp(ignore=['index', 'flip']) + task.config.fp


def plot_compare2(tasks, args):
    options = Options('compare')
    options.add_value_option('index1', value_type=int, default=0,
                             desc='index of first plot in comparison')
    options.add_value_option('index2', value_type=int, default=None,
                             desc='index of second plot in comparison')
    options.add_option('big', 'show big plot')
    options.add_option('correct-h', 'correct prediction for h')
    options.add_option('soft-correct-h', 'correct sign of prediction for h')
    options.add_option('ms', 'rescale to milliseconds')
    options.add_option('no-psd', 'no-psd')
    options.add_option('mf1', 'plot instead MF prediction for first task')
    options.add_option('mf2', 'plot instead MF prediction for second task')
    options.add_option('mp', 'zero phase the reference')
    options.add_option('2side-h', 'plot both sides of the filter')
    options.parse(args)

    p = Plotter2D(figure_size=(20, 10) if options['big'] else (12, 6),
                  font_size=12,
                  figure_toolbar='toolbar2' if options['big'] else 'none',
                  grid_colour='none')
    p.figure()

    tau_ws = 2

    # Process first task
    task1 = tasks[options['index1']]
    data1, pt1 = task1.data, TaskPlotter(p, task1)

    # Possibly process second task
    if options['index2'] is not None:
        task2 = tasks[options['index2']]
        data2, pt2 = task2.data, TaskPlotter(p, task2)
    else:
        task2 = None

    if options['mf1']:
        add1 = ''
        name1 = pt1.model_name + ' (MF)'
    else:
        add1 = '_smf'
        name1 = pt1.model_name + ' (SMF)'

    if task2:
        if options['mf2']:
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

    # Function
    p.subplot2grid((2, 6), (0, 0), colspan=6)
    p.x_shift = -data1['f'].x[0]
    pt1.marker('tx_data', 'k')
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

    if options['2side-h']:
        pt1.bound(x_min=-tau_ws * task1.config.tau_w,
                  x_max=tau_ws * task1.config.tau_w)
        if task2:
            pt2.bound(x_min=-tau_ws * task1.config.tau_w,
                      x_max=tau_ws * task1.config.tau_w)
    else:
        pt1.bound(x_min=0, x_max=tau_ws * task1.config.tau_w)
        if task2:
            pt2.bound(x_min=0, x_max=tau_ws * task1.config.tau_w)

    # Kernel
    if options['no-psd']:
        p.subplot2grid((2, 6), (1, 0), colspan=3)
    else:
        p.subplot2grid((2, 6), (1, 0), colspan=2)
    if options['2side-h']:
        p.lims(x=(-tau_ws * task1.config.tau_w, tau_ws * task1.config.tau_w))
    else:
        p.lims(x=(0, tau_ws * task1.config.tau_w))
    pt1.marker('th_data', 'k')
    pt1.line('k', 'truth')
    data1['k_emp'] = data1['f'].autocorrelation()
    data1['k_emp'] /= data1['k_emp'].max
    pt1.line('k_emp', 'observation')
    pt1.fill('k_pred' + add1, 'task1')
    if task2:
        pt2.fill('k_pred' + add2, 'task2')
    p.labels(y='$k_{f\,|\,h}$', x='$t$ ({})'.format(x_unit))

    # Filter
    if options['no-psd']:
        p.subplot2grid((2, 6), (1, 3), colspan=3)
    else:
        p.subplot2grid((2, 6), (1, 2), colspan=2)
    if options['2side-h']:
        p.lims(x=(-tau_ws * task1.config.tau_w, tau_ws * task1.config.tau_w))
    else:
        p.lims(x=(0, tau_ws * task1.config.tau_w))
    pt1.marker('th_data', 'k')
    # if options['mp']:
    data1['h'] = data1['h'].minimum_phase()
    #     data1['h_mp'] /= data1['h_mp'].energy ** .5
    #     pt1.line('h_mp', 'truth')
    # else:
    pt1.line('h', 'truth')
    data1['h_pred_smf'] = list(data1['h_pred_smf'])
    data1['h_pred_smf'][0] = data1['h_pred_smf'][0].minimum_phase()
    if options['correct-h']:
        correct_filter(task1, 'h_pred' + add1, 'h2_pred' + add1, 'h',
                       soft=options['soft-correct-h'])
        pt1.fill('h2_pred' + add1, 'task1')
    else:
        pt1.fill('h_pred' + add1, 'task1')
    if task2:
        if options['correct-h']:
            correct_filter(task2, 'h_pred' + add2, 'h2_pred' + add2, 'h',
                           soft=options['soft-correct-h'])
            pt2.fill('h2_pred' + add2, 'task2')
        else:
            pt2.fill('h_pred' + add2, 'task2')
    p.labels(y='$h$', x='$t$ ({})'.format(x_unit))

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
        p.lims(x=(0, 1.5 / task1.config.tau_f))
        pt1.line('psd', 'truth', x_unit=data1['psd_fs'])
        pt1.fill('psd_pred_smf', 'task1', x_unit=fs1)
        if task2:
            pt2.fill('psd_pred_smf', 'task2', x_unit=fs2)
        p.labels(y='PSD of $f\,|\,h$ (dB)', x='Frequency ({})'.format(x_unit))
        p.ax.set_ylim(bottom=-15)

    # Return `Plotter2D` instance and file path
    fp = options.fp(ignore=['index1', 'index2', 'big', 'correct-h', 'ms',
                            'no-psd', 'mf1', 'mf2', 'soft-correct-h',
                            '2side-h', 'mp'])
    if task2:
        fp += task1.config.fp & task2.config.fp
    else:
        fp += task1.config.fp
    return p, fp


def plot_compare(tasks, args):
    """
    Compare the GPCM and CGPCM.

    :param tasks: tasks
    :param args: arguments
    :return: :class:`core.plot.Plotter2D` instance
    """
    options = Options('compare')
    options.add_value_option('index1', value_type=int, default=0,
                             desc='index of first plot in comparison')
    options.add_value_option('index2', value_type=int, default=1,
                             desc='index of second plot in comparison')
    options.add_option('big', 'show big plot')
    options.parse(args)

    task1, task2 = tasks[options['index1']], tasks[options['index2']]

    # Config
    truth_colour = '#7b3294'
    observation_colour = '#008837'
    task1_colour = '#0571b0'
    task2_colour = '#ca0020'
    marker_size = 2
    inducing_point_size = 2

    p = Plotter2D(figure_size=(24, 5) if options['big'] else (12, 6),
                  font_size=12, figure_toolbar='none', grid_colour='none')
    p.figure()

    def plot_pred(p, pred, label, colour):
        mu, lower, upper, std = pred
        p.plot(mu.x, mu.y, label=label, line_width=1, line_colour=colour)
        p.fill(lower.x, lower.y, upper.y,
               fill_colour=colour,
               fill_alpha=.25,
               zorder=3)

    # Function
    p.subplot2grid((5, 3), (0, 0), colspan=3, rowspan=3)
    p.plot(task1.data['tx'], task1.data['tx'] * 0,
           line_style='none',
           marker_style='o',
           marker_size=inducing_point_size,
           marker_colour='k')
    p.plot(task1.data['e'].x, task1.data['e'].y,
           line_style='none',
           label='Observations',
           marker_style='o',
           marker_size=marker_size,
           marker_colour=observation_colour)
    p.plot(task1.data['f'].x, task1.data['f'].y,
           line_style='none',
           label='Truth',
           marker_style='o',
           marker_size=marker_size,
           marker_colour=truth_colour)
    plot_pred(p, task1.data['f_pred_smf'],
              'CGPCM' if task1.config.causal_model else 'GPCM', task1_colour)
    plot_pred(p, task2.data['f_pred_smf'],
              'CGPCM' if task2.config.causal_model else 'GPCM', task2_colour)
    p.labels(y='$f\,|\,h$', x='$t$ (s)')
    p.ticks_arange(x=(min(task1.data['e'].x),
                      max(task1.data['e'].x) + 0.1,
                      0.1))
    p.lims(x=(min(task1.data['e'].x),
              max(task1.data['e'].x)))
    p.show_legend()

    def plot_inducing_points(p):
        p.plot(task1.data['th'], task1.data['th'] * 0,
               line_style='none',
               marker_style='o',
               marker_size=inducing_point_size,
               marker_colour=task1_colour)
        p.plot(task2.data['th'], task2.data['th'] * 0,
               line_style='none',
               marker_style='o',
               marker_size=inducing_point_size,
               marker_colour=task2_colour)

    # Kernel
    p.subplot2grid((5, 3), (3, 0), rowspan=2)
    plot_inducing_points(p)
    plot_pred(p, map(lambda x: x.positive_part(), task1.data['k_pred_smf']),
              'CGPCM' if task1.config.causal_model else 'GPCM', task1_colour)
    plot_pred(p, map(lambda x: x.positive_part(), task2.data['k_pred_smf']),
              'CGPCM' if task2.config.causal_model else 'GPCM', task2_colour)
    p.plot(task1.data['k'].positive_part().x,
           task1.data['k'].positive_part().y,
           label='Truth',
           line_colour=truth_colour,
           line_width=1)
    p.labels(y='$k_{f\,|\,h}$', x='$t$ (s)')
    p.lims(x=(0, .5 * max(task1.data['k'].x)))

    # Filter
    p.subplot2grid((5, 3), (3, 1), rowspan=2)
    plot_inducing_points(p)
    plot_pred(p, map(lambda x: x.positive_part(), task1.data['h_pred_smf']),
              'CGPCM' if task1.config.causal_model else 'GPCM', task1_colour)
    plot_pred(p, map(lambda x: x.positive_part(), task2.data['h_pred_smf']),
              'CGPCM' if task2.config.causal_model else 'GPCM', task2_colour)
    p.plot(task1.data['h'].positive_part().x,
           task1.data['h'].positive_part().y,
           label='Truth',
           line_colour=truth_colour,
           line_width=1)
    p.labels(y='$h$', x='$t$ (s)')
    p.lims(x=(0, .5 * max(task1.data['h'].x)))

    # Build a filter to limit x axis and convert to absolute frequencies,
    # assuming that the sampling frequency for both tasks is equal, and equal
    # to that of the kernel
    freq_max = 0.1
    psd, Fs = task1.data['k'].fft_db()

    def freq_filter(d):
        d = d[np.logical_and(d.x >= 0, d.x <= freq_max)]
        d.x *= Fs
        return d

    # PSD
    p.subplot2grid((5, 3), (3, 2), rowspan=2)
    plot_pred(p, map(freq_filter, task1.data['psd_pred_smf']),
              'CGPCM' if task1.config.causal_model else 'GPCM', task1_colour)
    plot_pred(p, map(freq_filter, task2.data['psd_pred_smf']),
              'CGPCM' if task2.config.causal_model else 'GPCM', task2_colour)
    p.plot(freq_filter(psd).x,
           freq_filter(psd).y,
           label='Truth',
           line_colour=truth_colour,
           line_width=1)
    p.labels(y='PSD of $f\,|\,h$ (dB)', x='Frequency (Hz)')
    p.lims(x=(0, max(freq_filter(psd).x)))
    p.lims(y=(-10, 20))

    return p, \
           options.fp(groups=[['big']], ignore=['index1', 'index2']) \
           + (task1.config.fp & task2.config.fp)


plot_choices = ['full', 'compare', 'compare2']
plot_calls = {'full': plot_full,
              'compare': plot_compare,
              'compare2': plot_compare2}


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
