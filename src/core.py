#!/usr/bin/env python
import importlib
import pickle
import argparse

from core.tfutil import *
import core.util as util
import core.out as out
import core.exp as exp


def plot(plot_specification, tasks, show):
    """
    Plot a list of tasks.

    :param plot_specification: specification of plot
    :param tasks: tasks
    :param show: show plot
    """
    plot_type, plot_options = plot_specification[0], plot_specification[1:]
    out.kv('plot type', plot_type)

    out.section('fetching tasks')
    fetched_tasks = []
    for task_args in tasks:
        out.kv('specification', ' '.join(task_args))

        # Fetch task
        module = importlib.import_module('tasks.{}'.format(task_args[0]))
        fp = module.Experiment(options=task_args[1:]).config.fp
        path = 'tasks/cache/' + str(fp) + '.pickle'
        with open(path) as f:
            fetched_tasks.append(pickle.load(f))
    out.section_end()

    out.section('plotting')
    p, fp = exp.plot(plot_type, fetched_tasks, args=plot_options)
    path = 'output/' + str(fp) + '.pdf'
    util.mkdirs(path)
    p.save(path)
    if show:
        p.show()
    out.section_end()


def learn(task_specification):
    """
    Learn a task.

    :param task_specification: task specification
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
        exp.train(sess, task, args.debug)
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
    parser.add_argument('-l', '--learn', type=int, action='store', nargs='*',
                        help='learn all tasks, or specify indices of tasks to '
                             'learn')
    parser.add_argument('-t', '--task', type=str, action='append', nargs='+',
                        help='task to run with options', default=[])
    parser.add_argument('-d', '--debug', nargs='*', help='debug options',
                        default=[])
    parser.add_argument('-p', '--plot', type=str, action='append', nargs='+',
                        help='plot with options', default=[])
    parser.add_argument('-s', '--show', action='store_true', help='show plots')

    args = parser.parse_args()

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

    if args.learn:
        for i in args.learn:
            task_specification = args.task[i]
            out.section('learning task')
            out.kv('task index', '{}/{}'.format(i, len(args.task) - 1))
            out.kv('specification', ' '.join(task_specification))
            learn(task_specification)
            out.section_end()

    for i, plot_specification in enumerate(args.plot):
        out.section('plotting')
        out.kv('plot index', '{}/{}'.format(i, len(args.plot) - 1))
        out.kv('specification', ' '.join(plot_specification))
        plot(plot_specification, args.task, args.show)
        out.section_end()
