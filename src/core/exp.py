import abc
import numpy as np

from par import Parametrisable
from plot import Plotter2D
from options import Options
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


class TaskConfig(Parametrisable):
    """
    Configuration for a task.
    """
    _required_pars = ['fp', 'seed', 'iters', 'iters_pre', 'samps', 'name']


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
                                                  samples_h=task.data[
                                                      'samples'],
                                                  psd=True)
        task.data['h_pred_smf'] = mod.predict_h(task.data['h'].x,
                                                samples_h=task.data['samples'],
                                                assert_positive_at_index=pos_i)
        out.section_end()


def plot_full(tasks, args):
    """
    Fully plot a single task.

    :param tasks: tasks
    :param args: arguments
    :return: `Plotter2D` instance
    """
    options = Options('full')
    options.add_value_option('index', value_type=int, default=0,
                             desc='index of task to plot')
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

    if 'h_pred_smf' in task.data:
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

    return p, options.fp(ignore=['index']) + task.config.fp


def plot_compare(tasks, args):
    """
    Compare the GPCM and CGPCM.

    :param tasks: tasks
    :param args: arguments
    :return: `Plotter2D` instance
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

    return p, options.fp(groups=[['big']], ignore=['index1', 'index2']) \
           + (task1.config.fp & task2.config.fp)


plot_choices = ['full', 'compare']
plot_calls = {'full': plot_full,
              'compare': plot_compare}


def plot(choice, tasks, args):
    """
    Plot a task.

    Choice must be in `plot_choices`.

    :param choice: choice
    :param tasks: tasks
    :param args: plot arguments
    :return `Plotter2D` instance and file name
    """
    return plot_calls[choice](tasks, args)
