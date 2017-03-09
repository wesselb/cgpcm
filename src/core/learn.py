from tensorflow.contrib.opt.python.training.external_optimizer import \
    ScipyOptimizerInterface as SciPyOpt
import time
import abc
import sys

from tfutil import *
from par import Parametrisable
import out


class Progress(object):
    """
    Progress display.

    A fetch is specified through a dictionary where the key `name` specifies
    the name of the fetch, the key `modifier` specifies the modifier of the
    fetch, and the key `unit` specifies the unit of the fetch.

    :param name: name of task
    :param iters: total number of iterations
    :param fetches_config: list of dictionaries that specify the fetches
    """

    def __init__(self, name, iters=None, fetches_config=None):
        self._started = False
        self._iters = iters
        self._name = name
        if fetches_config is None:
            fetches_config = []
        self._fetches_config = fetches_config
        self._fetches_cache = [None for i in range(len(fetches_config))]
        self._printed = 0

    def __call__(self, fetches=None, step=True):
        """
        Show progress display.

        :param fetches: fetches, set to `None` if none are fetched
        :param step: increase number of executed iterations increased by one
        """
        if not self._started:
            self._started = True
            self._start_time = time.time()
            self._iter = 1
        elif step:
            self._iter += 1
        fetches = self._cache_fetches(fetches)

        # Show log
        self._erase_previous_output()
        self._section(self._name)
        self._print_iters()
        self._print_time_elapsed()
        self._print_time_left()
        self._print_fetches(fetches)
        out.section_end()

    def _kv(self, *args, **kw_args):
        self._printed += 1
        out.kv(*args, **kw_args)

    def _section(self, *args, **kw_args):
        self._printed += 1
        out.section(*args, **kw_args)

    def _erase_previous_output(self):
        out.eat(self._printed)
        self._printed = 0

    def _cache_fetches(self, fetches):
        if fetches is None:
            fetches = self._fetches_cache
        else:
            self._fetches_cache = fetches
        return fetches

    def _print_iters(self):
        if self._iters is not None:
            status = '{}/{}'.format(self._iter, self._iters)
        else:
            status = '{}'.format(self._iter)
        self._kv('iteration', status)

    def _print_time_left(self):
        if self._iters is None:
            return
        time_it = (time.time() - self._start_time) / self._iter
        time_left = time_it * (self._iters - self._iter)
        self._kv('time left', time_left, mod='.1f', unit='s')

    def _print_time_elapsed(self):
        time_elapsed = time.time() - self._start_time
        self._kv('time elapsed', time_elapsed, mod='.1f', unit='s')

    def _print_fetches(self, fetches):
        for conf, fetch in zip(self._fetches_config, fetches):
            if conf is not None:
                unit = conf['unit'] if 'unit' in conf else ''
                mod = conf['modifier'] if 'modifier' in conf else ''
                self._kv(conf['name'], fetch, mod=mod, unit=unit)


def minimise_lbfgs(sess, objective, vars, iters, fetches_config=None):
    """
    Minimise some objective using `scipy.optimise.fmin_l_bfgs_b`.

    :param sess: TensorFlow session
    :param objective: objective
    :param vars: list of variables to optimise
    :param iters: number of iterations
    :param fetches_config: config for fetches as specified in `Progress` and
                           with an additional key `tensor` that gives the
                           tensor of the fetch
    """
    if iters == 0:
        return
    if fetches_config is None:
        fetches_config = []

    # SciPy's fmin_l_bfgs_b tends to perform two extra iterations
    opt = SciPyOpt(objective,
                   options={'disp': False,
                            'maxiter': iters - 2},
                   var_list=vars)
    initialise_uninitialised_variables(sess)

    progress = Progress(name='minimisation using L-BFGS',
                        iters=iters,
                        fetches_config=fetches_config)

    opt.minimize(sess,
                 loss_callback=lambda *fetches: progress(fetches,
                                                         step=False),
                 step_callback=lambda x: progress(step=True),
                 fetches=[x['tensor'] for x in fetches_config])


def map_progress(f, xs, name):
    """
    Map whilst showing progress.

    :param f: function
    :param xs: list
    :param name: name of operation
    :return: transformed list
    """
    progress = Progress(name=name, iters=len(xs))

    def mapping_fun(x):
        progress()
        return f(x)

    return map(mapping_fun, xs)


class TaskConfig(Parametrisable):
    """
    Configuration for a task.
    """
    _required_pars = ['fn', 'seed', 'iters', 'iters_pre', 'samps', 'name']


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

    def _set_data(self, h=None, k=None, f=None, e=None):
        self.data['h'] = h
        self.data['k'] = k
        self.data['f'] = f
        self.data['e'] = e

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
