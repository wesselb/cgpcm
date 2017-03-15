import abc
import numpy as np

from par import Parametrisable
from plot import Plotter2D
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
        :return: instance of `TaskConfig`
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

    def _set_data(self, h=None, k=None, f=None, e=None, psd=None):
        self.data['h'] = h
        self.data['k'] = k
        self.data['f'] = f
        self.data['e'] = e
        self.data['psd'] = psd

    def _set_model(self, mod):
        self.mod = mod

    def make_pickleable(self, sess):
        """
        Save useful data in the property `data` and make the object pickleable.
        """
        self.data.update({'tx': sess.run(self.mod.tx),
                          'th': sess.run(self.mod.th),
                          's2': sess.run(self.mod.s2),
                          's2_f': sess.run(self.mod.s2_f),
                          'mu': sess.run(self.mod.h.mean),
                          'var': sess.run(self.mod.h.var)})
        del self.mod


class TaskOptions(object):
    """
    Options for task.
    """

    def __init__(self):
        self._options = []

    def add_option(self, name, desc='no description available'):
        """
        Add a boolean option.

        :param name: name of options
        :param desc: description of option
        """
        self._options.append({'has_value': False,
                              'name': name.lower(),
                              'value': False,
                              'description': desc,
                              'required': False})

    def add_value_option(self, name, value_type,
                         desc='no description available', required=False):
        """
        Add an option with a value.

        :param name: name of option
        :param value_type: type of option, should be callable
        :param desc: description of option
        :param required: option is required
        """
        self._options.append({'has_value': True,
                              'name': name.lower(),
                              'value_type': value_type,
                              'value': None,
                              'description': desc,
                              'required': required})

    def _get_option(self, name):
        for option in self._options:
            if name == option['name']:
                return option
        raise RuntimeError('option "{}" not found'.format(name))

    def parse(self, args):
        """
        Parse arguments.

        :param args: arguments
        """
        self._options = sorted(self._options, key=lambda x: x['name'])
        self._parse_help(args)
        self._parse_args(args)
        self._parse_required()

    def _parse_required(self):
        missing = []
        for option in self._options:
            if option['required'] and option['value'] is None:
                missing.append(option['name'])
        if len(missing) == 1:
            raise RuntimeError('missing option "{}"'.format(missing[0]))
        elif len(missing) > 1:
            missing_string = ', '.join(['"{}"'.format(x) for x in missing])
            raise RuntimeError('missing options {}'.format(missing_string))

    def _parse_args(self, args):
        it = iter(args)
        for arg in it:
            option = self._get_option(arg)
            if option['has_value']:
                option['value'] = option['value_type'](next(it))
            else:
                option['value'] = True

    def _parse_help(self, args):
        if 'help' in args:
            out.section('options for task')
            for option in self._options:
                out.section(option['name'])
                out.kv('description', option['description'])
                out.kv('type', 'value' if option['has_value'] else 'bool')
                out.kv('required', 'yes' if option['required'] else 'no')
                out.section_end()
            out.section_end()
            exit()

    def __getitem__(self, name):
        return self._get_option(name)['value']

    def fn(self, group_by=None, prefix=''):
        group_names = [] if group_by is None else sorted(group_by)
        fn_names = sorted(set([x['name'] for x in self._options])
                          - set(group_names))

        if prefix:
            prefix += '_'  # Add separator

        group_part = self._fn_opts_to_str(map(self._get_option, group_names))
        if group_part:
            prefix += group_part + '/'

        return prefix + self._fn_opts_to_str(map(self._get_option, fn_names))

    def _fn_opts_to_str(self, xs):
        def to_str(x): return ('y' if x else 'n') if type(x) == bool \
            else str(x)

        return ','.join(['{}={}'.format(x['name'], to_str(x['value']))
                         for x in xs])


class TaskConfig(Parametrisable):
    """
    Configuration for a task.
    """
    _required_pars = ['fn', 'seed', 'iters', 'iters_pre', 'samps', 'name']


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
    if 'kernel' in debug_options:
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
    pos_i = util.nearest_index(task.data['h'].x, .2 * mod.tau_w)
    task.data['h_pred'] = mod.predict_h(task.data['h'].x,
                                        assert_positive_at_index=pos_i)
    out.section_end()

    if task.config.samps > 0:
        # Train SMF
        out.section('training SMF')
        task.data['samples'] = mod.sample(iters=task.config.samps)
        out.section_end()

        # Predict SMF
        out.section('predicting SMF')
        task.data['f_pred_smf'] = mod.predict_f(task.data['f'].x,
                                                samples_h=task.data['samples'])
        task.data['k_pred_smf'] = mod.predict_k(task.data['k'].x,
                                                samples_h=task.data['samples'])
        task.data['psd_pred_smf'] = mod.predict_k(task.data['k'].x,
                                                  samples_h=task.data['samples'],
                                                  psd=True)
        task.data['h_pred_smf'] = mod.predict_h(task.data['h'].x,
                                                samples_h=task.data['samples'],
                                                assert_positive_at_index=pos_i)
        out.section_end()


def plot_full(tasks, options):
    """
    Fully plot a single task.

    :param tasks: tasks
    :param options: options
    :return: `Plotter2D` instance
    """

    if len(tasks) != 1:
        raise RuntimeError('can only plot a single task')
    else:
        task = tasks[0]

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
    p.figure('Results')

    p.subplot(2, 3, 1)
    p.title('Filter')
    do_plot(p, task.data['h'], task.data['h_pred'],
            inducing_points=task.data['th'])

    p.subplot(2, 3, 2)
    p.title('Kernel')
    do_plot(p, task.data['k'], task.data['k_pred'],
            inducing_points=task.data['th'])

    p.subplot(2, 3, 3)
    p.title('Function')
    do_plot(p, task.data['f'], task.data['f_pred'], x_noisy=task.data['e'],
            inducing_points=task.data['tx'])

    if 'h_pref_smf' in task.data:
        p.subplot(2, 3, 4)
        p.title('Filter')
        do_plot(p, task.data['h'], task.data['h_pred_smf'],
                inducing_points=task.data['th'])

        p.subplot(2, 3, 5)
        p.title('Kernel')
        do_plot(p, task.data['k'], task.data['k_pred_smf'],
                inducing_points=task.data['th'])

        p.subplot(2, 3, 6)
        p.title('Function')
        do_plot(p, task.data['f'], task.data['f_pred_smf'],
                x_noisy=task.data['e'], inducing_points=task.data['tx'])

    return p, task.config.fn


def plot_compare(tasks, options):
    """
    Compare the GPCM and CGPCM.

    :param tasks: tasks
    :param options: options
    :return: `Plotter2D` instance
    """
    if len(tasks) != 2:
        raise RuntimeError('can only compare two tasks')
    else:
        task1, task2 = tasks

    # Config
    truth_colour = '#7b3294'
    observation_colour = '#008837'
    task1_colour = '#0571b0'
    task2_colour = '#ca0020'
    marker_size = 2
    inducing_point_size = 2

    p = Plotter2D(figure_size=(24, 5) if 'big' in options else (12, 6),
                  font_size=12, figure_toolbar='none', grid_colour='none')

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
    p.title('$f\,|\,h$')
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
    p.title('$k_{f\,|\,h}$')
    p.lims(x=(0, max(task1.data['k'].x)))

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
    p.title('$h$')
    p.lims(x=(0, max(task1.data['h'].x)))

    # Build a filter to limit x axis
    freq_max = 0.1

    def freq_filter(d):
        return d.filter(lambda x: x[np.logical_and(x >= 0, x <= freq_max)])[0]

    # PSD
    p.subplot2grid((5, 3), (3, 2), rowspan=2)
    plot_pred(p, map(freq_filter, task1.data['psd_pred_smf']),
              'CGPCM' if task1.config.causal_model else 'GPCM', task1_colour)
    plot_pred(p, map(freq_filter, task2.data['psd_pred_smf']),
              'CGPCM' if task2.config.causal_model else 'GPCM', task2_colour)
    p.plot(freq_filter(task1.data['psd']).x,
           freq_filter(task1.data['psd']).y,
           label='Truth',
           line_colour=truth_colour,
           line_width=1)
    p.title('PSD of $f\,|\,h$')
    p.lims(x=(0, max(freq_filter(task1.data['psd']).x)))
    p.lims(y=(-10, 20))

    group1, fn1 = task1.config.fn.split('/')
    group2, fn2 = task2.config.fn.split('/')

    return p, '{}/{}_versus_{}'.format(group1, fn1, fn2)


plot_choices = ['full', 'compare']
plot_calls = {'full': plot_full,
              'compare': plot_compare}


def plot(choice, tasks, options):
    """
    Plot a task.

    Choice must be in `plot_choices`.

    :param choice: choice
    :param tasks: tasks
    :param options: plot options
    :return `Plotter2D` instance and file name
    """
    return plot_calls[choice](tasks, options)
