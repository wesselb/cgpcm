from tensorflow.contrib.opt.python.training.external_optimizer import \
    ScipyOptimizerInterface as SciPyOpt
import time

import out
from tfutil import *


class Progress(object):
    """
    Progress display.

    A fetch is specified through a dictionary where the key `name` specifies
    the name of the fetch, and the key `modifier` specifies the modifier of the
    fetch.

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

    # fmin_l_bfgs_b tends to perform two extra iterations
    opt = SciPyOpt(objective,
                   options={'disp': False,
                            'maxiter': iters - 2},
                   var_list=vars)
    initialise_uninitialised_variables(sess)

    progress = Progress(
        name='minimisation using scipy.optimize.fmin_l_bfgs_b',
        iters=iters,
        fetches_config=fetches_config)

    opt.minimize(sess,
                 loss_callback=lambda *fetches: progress(fetches,
                                                         step=False),
                 step_callback=lambda x: progress(step=True),
                 fetches=[x['tensor'] for x in fetches_config])
