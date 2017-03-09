#!/usr/bin/env python
import sys
import importlib
import pickle
import argparse

from core.tfutil import *
from core.plot import Plotter2D
import core.util as util
import core.out as out
import core.learn as learn

parser = argparse.ArgumentParser(description='Run a task.')
parser.add_argument('--task', required=True, type=str, help='task to run')
parser.add_argument('--train', action='store_true', help='train model')
parser.add_argument('--debug-kernel', action='store_true',
                    help='show the kernel and exit before the model '
                         'is trained')
parser.add_argument('--show', action='store_true', help='show plots')
parser.add_argument('--options', action='store', nargs='*',
                    help='options for task')

args = parser.parse_args()

# Fetch task
module = importlib.import_module('tasks.{}'.format(args.task))
task = module.Experiment(options=args.options)

# Initialise
sess = Session()
util.seed(task.config.seed)
out.kv('argv', ' '.join(sys.argv))
out.kv('task name', task.config.name)

if args.train:
    # Load task
    out.section('loading task')
    task.load(sess)
    mod = task.mod
    out.section_end()

    # Check whether to debug kernel
    if args.debug_kernel:
        out.state('plotting kernel and then exiting')
        p = Plotter2D()
        p.figure('Kernel')
        p.plot(task.data['k'].x, task.data['k'].y)
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
                         fetches_config=fetches_config)
    learn.minimise_lbfgs(sess, -elbo,
                         vars=[mod.vars['mu'],
                               mod.vars['var'],
                               mod.vars['s2'],
                               mod.vars['s2_f']],
                         iters=task.config.iters,
                         fetches_config=fetches_config)
    out.section_end()

    # Predict MF
    out.section('predicting MF')
    task.data['f_pred'] = mod.predict_f(task.data['f'].x)
    task.data['k_pred'] = mod.predict_k(task.data['k'].x)
    pos_i = util.nearest_index(task.data['h'].x, .2 * mod.k_len)
    task.data['h_pred'] = mod.predict_h(task.data['h'].x,
                                        assert_positive_at_index=pos_i)
    out.section_end()

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
    task.data['h_pred_smf'] = mod.predict_h(task.data['h'].x,
                                            samples_h=task.data['samples'],
                                            assert_positive_at_index=pos_i)
    out.section_end()

    # Save resulting task
    out.section('saving')
    task.make_pickleable(sess)
    with open('tasks/cache/{}.pickle'.format(task.config.fn), 'w') as f:
        pickle.dump(task, f)
    out.section_end()
else:
    # Load task
    out.section('loading')
    with open('tasks/cache/{}.pickle'.format(task.config.fn)) as f:
        task = pickle.load(f)
    out.section_end()


def plot(p, x, x_pred, x_noisy=None, inducing_points=None):
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


# Plotting
out.section('plotting')
p = Plotter2D(figure_size=(20, 10))
p.figure('Results')

p.subplot(2, 3, 1)
p.title('Filter')
plot(p, task.data['h'], task.data['h_pred'], inducing_points=task.data['th'])

p.subplot(2, 3, 2)
p.title('Kernel')
plot(p, task.data['k'], task.data['k_pred'], inducing_points=task.data['th'])

p.subplot(2, 3, 3)
p.title('Function')
plot(p, task.data['f'], task.data['f_pred'], x_noisy=task.data['e'],
     inducing_points=task.data['tx'])

p.subplot(2, 3, 4)
p.title('Filter')
plot(p, task.data['h'], task.data['h_pred_smf'],
     inducing_points=task.data['th'])

p.subplot(2, 3, 5)
p.title('Kernel')
plot(p, task.data['k'], task.data['k_pred_smf'],
     inducing_points=task.data['th'])

p.subplot(2, 3, 6)
p.title('Function')
plot(p, task.data['f'], task.data['f_pred_smf'], x_noisy=task.data['e'],
     inducing_points=task.data['tx'])

p.save('output/out_{}.pdf'.format(task.config.fn))
if args.show:
    p.show()
out.section_end()
