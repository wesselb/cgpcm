#!/usr/bin/env python
import importlib
import pickle
import argparse
import matplotlib.pyplot as plt
import sys

from core.tf_util import *
from core.options import Options
import core.util as util
import core.out as out
import core.experiment as exp


# Compatibility
sys.modules['core.exp'] = exp


def fetch(specifications, remote):
    """
    Fetch tasks from their specifications.

    :param specifications: specifications
    :param remote: use remote cache
    :return: tasks
    """
    out.section('fetching tasks')
    fetched_tasks = []
    for specification in specifications:
        task_name, task_options = specification[0], specification[1:]
        out.kv('specification', ' '.join(specification))

        # Fetch task
        try:
            module = importlib.import_module('tasks.{}'.format(task_name))
            fp = module.Experiment(options=task_options).config.fp
            path = 'tasks/{}cache/'.format('remote/' if remote else '') \
                   + str(fp) + '.pickle'
            with open(path) as f:
                fetched_tasks.append(pickle.load(f))
        except IOError:
            out.state('couldn\'t fetch task: cache not found')
        except ImportError:
            out.state('couldn\'t fetch task: task not found')

    out.section_end()
    return fetched_tasks


def plot(plot_specification, tasks):
    """
    Plot a list of tasks.

    :param plot_specification: specification of plot
    :param tasks: tasks
    """
    plot_type, plot_options = plot_specification[0], plot_specification[1:]
    out.kv('plot type', plot_type)
    out.section('plotting')
    p, fp = exp.plot(plot_type, tasks, args=plot_options)
    path = 'output/' + str(fp) + '.pdf'
    util.mkdirs(path)
    p.save(path)
    out.section_end()


def learn(task_specification, debug_options):
    """
    Learn a task.

    :param task_specification: task specification
    :param debug_options: debug options
    """
    task_name, task_options = task_specification[0], task_specification[1:]

    # Fetch task
    module = importlib.import_module('tasks.{}'.format(task_name))
    task = module.Experiment(options=task_options)

    # Initialise
    out.kv('name', task.config.name)
    tf.reset_default_graph()
    with Session() as sess:
        util.seed(task.config.seed)

        # Train
        out.section('training')
        exp.train(sess, task, debug_options)
        out.section_end()

        # Save resulting task
        out.section('saving')
        task.make_pickleable(sess)
        path = 'tasks/cache/' + str(task.config.fp) + '.pickle'
        util.mkdirs(path)
        with open(path, 'w') as f:
            pickle.dump(task, f)
        out.section_end()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a task.')
    parser.add_argument('-t', '--task', type=str, action='append', nargs='+',
                        help='task to run with options', default=[])
    parser.add_argument('-d', '--debug', nargs='*', help='debug options',
                        default=[])

    parser.add_argument('-l', '--learn', type=int, action='store', nargs='*',
                        help='learn all tasks, or specify indices of tasks to '
                             'learn')
    parser.add_argument('-p', '--plot', type=str, action='append', nargs='+',
                        help='plot with options', default=[])
    parser.add_argument('-s', '--show', action='store_true', help='show plots')
    parser.add_argument('--stats', action='store_true',
                        help='show statistics of tasks')
    parser.add_argument('-r', '--remote', action='store_true',
                        help='use remote task cache')

    args = parser.parse_args()

    debug_options = Options('debug options')
    debug_options.add_option('kernel', 'plot kernel and filter and then exit')
    debug_options.parse(args.debug)

    # Fix format of learn argument
    if type(args.learn) == int:
        args.learn = [args.learn]
    if args.learn == []:
        # If no indices for learn are specified, then learn all
        args.learn = range(len(args.task))

    out.kv('number of tasks', len(args.task))
    out.kv('number of plots', len(args.plot))
    indices_str = ', '.join(map(str, args.learn)) if args.learn else 'none'
    out.kv('learning tasks', indices_str)

    # Learning of tasks
    if args.learn:
        for i in args.learn:
            task_specification = args.task[i]
            out.section('learning task')
            out.kv('task index', '{}/{}'.format(i, len(args.task) - 1))
            out.kv('specification', ' '.join(task_specification))
            learn(task_specification, debug_options)
            out.section_end()

    if args.stats or args.plot:
        fetched_tasks = fetch(args.task, remote=args.remote)

    # Showing statistics
    if args.stats:
        for i, (task_specification, task) in enumerate(zip(args.task,
                                                           fetched_tasks)):
            out.section('evaluating task')
            out.kv('task index', '{}/{}'.format(i, len(args.task) - 1))
            out.kv('specification', ' '.join(task_specification))
            report = task.report()
            out.dict_(report, numeric_mod='.3e')
            path = 'output/stats/' + str(task.config.fp) + '.pickle'
            util.mkdirs(path)
            with open(path, 'w') as f:
                pickle.dump(report, f)
            out.section_end()

    # Plotting tasks
    for i, plot_specification in enumerate(args.plot):
        out.section('plotting')
        out.kv('plot index', '{}/{}'.format(i, len(args.plot) - 1))
        out.kv('specification', ' '.join(plot_specification))
        plot(plot_specification, fetched_tasks)
        out.section_end()

    # Showing plots
    if args.plot and args.show:
        plt.show()
